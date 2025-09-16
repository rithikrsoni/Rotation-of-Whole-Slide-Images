import os
import argparse
import subprocess
import tempfile
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import openslide
import tifffile
from tqdm import tqdm
import uuid


def readTiffMetadata(path):
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        width = page.imagewidth
        height = page.imagelength
        dtype = page.dtype
        extraTags = [
            (t.code, t.dtype, t.count, t.value, True)
            for t in page.tags.values()
        ]
    return page, width, height, dtype, extraTags


def computeCanvas(width, height, cosA, sinA):
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    cx, cy = width / 2.0, height / 2.0
    R = np.array([[cosA, -sinA], [sinA, cosA]], dtype=np.float64)
    shifted = corners - [cx, cy]
    rotated = shifted.dot(R.T) + [cx, cy]
    xMin, yMin = rotated.min(axis=0)
    xMax, yMax = rotated.max(axis=0)
    newWidth = int(np.ceil(xMax - xMin)) + 1
    newHeight = int(np.ceil(yMax - yMin)) + 1
    xOffset = -int(np.floor(xMin))
    yOffset = -int(np.floor(yMin))
    return newWidth, newHeight, xOffset, yOffset


def rotateStripes(
    inFile,
    outFile,
    angleDeg=45,
    tileWidth=240,
    tileHeight=240,
    jpegQuality=30,
    blockWidth=2000,
    blockHeight=2000,
):
    page, width, height, dtype, extraTags = readTiffMetadata(inFile)
    theta = np.deg2rad(angleDeg)
    cosA, sinA = np.cos(theta), np.sin(theta)

   # Compute output canvas
    newWidth, newHeight, xOffset, yOffset = computeCanvas(width, height, cosA, sinA)

    channels = getattr(page, "samplesperpixel", 3)

    memmapName = f"rotated_{uuid.uuid4().hex}.dat"
    finalMap = np.memmap(
        memmapName, mode="w+", dtype=np.uint8, shape=(newHeight, newWidth, channels)
    )

    slide = openslide.OpenSlide(inFile)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inverseR = torch.tensor(
        [[cosA, sinA], [-sinA, cosA]], device=device, dtype=torch.float32
    )

    for y0 in tqdm(range(0, newHeight, blockHeight), desc="Blocks Y"):
        y1 = min(y0 + blockHeight, newHeight)
        hBlock = y1 - y0
        for x0 in tqdm(range(0, newWidth, blockWidth), desc="Blocks X", leave=False):
            x1 = min(x0 + blockWidth, newWidth)
            wBlock = x1 - x0

            ys = torch.linspace(y0, y1 - 1, steps=hBlock, device=device) - yOffset
            xs = torch.linspace(x0, x1 - 1, steps=wBlock, device=device) - xOffset
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            coords = torch.stack([gx - width / 2, gy - height / 2], dim=-1)

            # Inverse rotate
            src = coords @ inverseR + torch.tensor([width / 2, height / 2], device=device)

         
            sx = 2.0 * (src[..., 0] / (width - 1)) - 1.0
            sy = 2.0 * (src[..., 1] / (height - 1)) - 1.0
            flow = torch.stack([sx, sy], dim=-1)[None]  # 1×h×w×2

            srcX = src[..., 0].cpu().numpy()
            srcY = src[..., 1].cpu().numpy()
            xMin = int(np.floor(srcX.min()))
            xMax = int(np.ceil(srcX.max()))
            yMin = int(np.floor(srcY.min()))
            yMax = int(np.ceil(srcY.max()))
            x0Src = max(xMin, 0)
            x1Src = min(xMax, width)
            y0Src = max(yMin, 0)
            y1Src = min(yMax, height)

            
            if x1Src <= x0Src or y1Src <= y0Src:
                finalMap[y0:y1, x0:x1] = 0
                continue

            
            patch = slide.read_region((x0Src, y0Src), 0, (x1Src - x0Src, y1Src - y0Src))
            arr = np.array(patch)[..., :channels]
            tensor = (
                torch.from_numpy(arr.transpose(2, 0, 1)[None]).float().to(device) / 255.0
            )

            flow[..., 0] = 2.0 * ((src[..., 0] - x0Src) / (x1Src - x0Src - 1)) - 1.0
            flow[..., 1] = 2.0 * ((src[..., 1] - y0Src) / (y1Src - y0Src - 1)) - 1.0

            warped = F.grid_sample(
                tensor, flow, mode="bilinear", padding_mode="zeros", align_corners=True
            )[0]
            stripe = (warped.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

           
            finalMap[y0:y1, x0:x1] = stripe

    slide.close()
    finalMap.flush()

    
    base = os.path.splitext(os.path.basename(outFile))[0]
    with tempfile.TemporaryDirectory() as td:
        flatTif = os.path.join(td, f"{base}_flat.tiff")
        tifffile.imwrite(
            flatTif,
            finalMap,
            dtype=np.uint8,
            photometric="rgb",
            compression="lzw",
            extratags=extraTags,
            tile=(tileWidth, tileHeight),
            bigtiff=True,
        )

        
        vipsPyr = os.path.join(td, f"{base}_pyr_pyr.tif")
        subprocess.check_call(
            [
                "vips",
                "tiffsave",
                flatTif,
                vipsPyr,
                "--tile",
                "--pyramid",
                "--compression",
                "jpeg",
                "--Q",
                str(jpegQuality),
                "--tile-width",
                str(tileWidth),
                "--tile-height",
                str(tileHeight),
                "--bigtiff",
            ]
        )

        finalName = os.path.splitext(outFile)[0]
        
        if not finalName.endswith("_pyr"):
            finalName = finalName + "_pyr"
        finalName = finalName + ".tif"

        shutil.move(vipsPyr, finalName)

    # cleanup memmap file
    try:
        os.remove(memmapName)
    except OSError:
        pass


"""
The section below accepts user input for the following: Input Directory, Output Directory, the at which the user wants to rotate the image by.
Chunk width is whether the user wants to specify the size the rotation processes the image at. 
JPEG Quality is the quality of the JPEG for the pyramidial file format. (Higher = better)


"""

def main():
    parser = argparse.ArgumentParser(
        description="Batch GPU/CPU-accelerated rotation producing QuPath-compatible pyramid TIFFs"
    )
    parser.add_argument("inPath", help="Input SVS/TIFF file or directory")
    parser.add_argument("outDir", help="Output directory for rotated pyramid TIFFs")
    parser.add_argument("--angleDeg", type=float, default=45, help="Rotation angle")
    parser.add_argument("--chunkWidth", type=int, default=240, help="Tile width")
    parser.add_argument("--chunkHeight", type=int, default=240, help="Tile height")
    parser.add_argument("--quality", type=int, default=30, help="JPEG quality")
    parser.add_argument("--blockWidth", type=int, default=2000, help="Warp block width")
    parser.add_argument("--blockHeight", type=int, default=2000, help="Warp block height")
    args = parser.parse_args()

    os.makedirs(args.outDir, exist_ok=True)

    if os.path.isdir(args.inPath):
        inputs = sorted(
            os.path.join(args.inPath, fn)
            for fn in os.listdir(args.inPath)
            if fn.lower().endswith((".svs", ".tif"))
        )
    else:
        inputs = [args.inPath]

    for inFile in tqdm(inputs, desc="WSIs"):
        base = os.path.splitext(os.path.basename(inFile))[0]
        outPyr = os.path.join(args.outDir, base + "_pyr.tif")
        if os.path.isfile(outPyr):
            print(f"[Skip] exists {outPyr}")
            continue
        print(f"[Start] {inFile} -> {outPyr}")
        rotateStripes(
            inFile,
            outPyr,
            angleDeg=args.angleDeg,
            tileWidth=args.chunkWidth,
            tileHeight=args.chunkHeight,
            jpegQuality=args.quality,
            blockWidth=args.blockWidth,
            blockHeight=args.blockHeight,
        )


if __name__ == "__main__":
    main()
