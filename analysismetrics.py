import os, glob, warnings, json, inspect
import numpy as np
import h5py
import matplotlib.pyplot as plt

from numpy.linalg import norm as LaNorm
from datetime import datetime

from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA  

""""
This script is reduce high dimensional data and create visualisation for feautre extractors using umap, t-sne and create a mean cosine box plot  

"""
NoRotationDir  = "home/rrs3/preAugmentedCoad/h_optimus_1-03adac80" # "home/rrs3/preAugmentedCoad/xiyuewang-ctranspath-7c998680-03adac80"
RotatedDir     ="/data/rrs3/RotatedCOAD60Hoptimus/h_optimus_1-03adac80" # "/data/rrs3/preprocessedRotatedCOAD/ctranspath-7e9d3eb3"

# multiple extractor pairs for large plot with all the feature extractors grouped together

MultiPairOriginalDirs = [
    # "/data/.../Gigapath/original",
    # "/data/.../CONCH/original",
]
MultiPairRotatedDirs = [
    # "/data/.../Gigapath/rot60",
    # "/data/.../CONCH/rot60",
]
MultiPairNames = [
    # "Gigapath", "CONCH"
]

OutputDir      = "/home/rrs3/latent_space_results+cosine_and_embeddingsHoptimus"
os.makedirs(OutputDir, exist_ok=True)

RandomSeed            = 42
MaxTilesPerSlide      = 1000    #  control memory
ForceFeatureName      = None    # 
UseCosineMetric       = True    # row L2-normalise + cosine metric for UMAP

# Auto Tuning to find optimum for visualisation  
TuneMaxSamples        = 20000   # parameter search (UMAP/t-SNE)
TrustMaxSamples       = 10000   #  to avoid memory issues
KnnPreserveSamples    = 3000    # 
TsneFinalMax          = 20000   # cap for final t-SNE visualisation
KForMetrics           = 15      # neighbourhood for trustworthiness/kNN overlap
LambdaSilhouette      = 0.2     

# UMAP grid
UMAPParamGrid = [
    {"n_neighbors": 30, "min_dist": 0.05},
    {"n_neighbors": 50, "min_dist": 0.10},
    {"n_neighbors": 80, "min_dist": 0.15},
    {"n_neighbors": 100, "min_dist": 0.25},
]

# t-SNE parameter grid 
TSNEParamGrid = [
    {"perplexity": 10, "learning_rate": 200.0},
    {"perplexity": 20, "learning_rate": 200.0},
    {"perplexity": 30, "learning_rate": 200.0},
]

try:
    import umap
    HaveUMAP = True
except Exception:
    HaveUMAP = False

try:
    from sklearn.manifold import TSNE
    HaveTSNE = True
except Exception:
    HaveTSNE = False

warnings.filterwarnings("ignore", category=UserWarning, module="umap.umap_")

# to capture the feautres, madeline, eagle, prism are all slide encoders setup within stamp
CandidateFeatureKeys = [
    "features","feats","embeddings","reps","representations",
    "conch_features","gigapath_features","uni_features","uni_embeddings",
    "phikon_features","virchow_features","prism_features","titan_features",
    "eagle_features","cobra_features","madeleine_features","patch_features",
    "tokens","cls"
]

def IterateDatasets(H5File):
    out = []
    def _Visitor(Name, Obj):
        if isinstance(Obj, h5py.Dataset):
            out.append((Name, Obj))
    H5File.visititems(_Visitor)
    return out

def LooksLikeFeatures(Name, Dataset):
    if Dataset.ndim != 2: return False
    NumRows, Dim = Dataset.shape
    if NumRows < 16 or Dim < 16: return False
    if Dim in (2, 3): return False
    if not np.issubdtype(Dataset.dtype, np.floating): return False
    Score = 0
    Lower = Name.lower()
    if any(k in Lower for k in ["feat","emb","rep","token","cls"]): Score += 2
    if 32 <= Dim <= 8192: Score += 1
    return Score > 0

def FindFeaturePath(H5File, ForceName=None):
    if ForceName:
        Target = ForceName.lower()
        for Name, Ds in IterateDatasets(H5File):
            if Name.split("/")[-1].lower() == Target and Ds.ndim == 2:
                return Name
    Candidates = []
    KeysLower = [k.lower() for k in CandidateFeatureKeys]
    for Name, Ds in IterateDatasets(H5File):
        if Ds.ndim == 2 and Name.split("/")[-1].lower() in KeysLower:
            Candidates.append((Name, Ds.shape[0]))
    if Candidates:
        Candidates.sort(key=lambda x: x[1], reverse=True)
        return Candidates[0][0]
    Best, BestN = None, -1
    for Name, Ds in IterateDatasets(H5File):
        if LooksLikeFeatures(Name, Ds):
            N = Ds.shape[0]
            if N > BestN:
                Best, BestN = Name, N
    return Best

def LoadFeaturesFromH5(PathStr, MaxRows=None, ForceName=None):
    with h5py.File(PathStr, "r") as F:
        FeaturePath = FindFeaturePath(F, ForceName)
        if FeaturePath is None:
            raise KeyError(f"No suitable 2D float feature dataset found in {PathStr}")
        Ds = F[FeaturePath]
        N = Ds.shape[0]
        if MaxRows is not None and N > MaxRows:
            Rng = np.random.default_rng(abs(hash(os.path.basename(PathStr))) % (2**32))
            Idx = np.sort(Rng.choice(N, MaxRows, replace=False))
            X = Ds[Idx, :]
        else:
            X = Ds[...]
    if not np.isfinite(X).all():
        raise ValueError(f"NaN/Inf found in {PathStr}:{FeaturePath}")
    if X.ndim != 2:
        raise ValueError(f"{PathStr}:{FeaturePath} not 2D, shape {X.shape}")
    return X


# Function to ensure the file is in the correct file format
def CollectH5FromPath(PathStr):
    if os.path.isdir(PathStr):
        Files = sorted(glob.glob(os.path.join(PathStr, "*.h5")))
        if not Files:
            Files = sorted(glob.glob(os.path.join(PathStr, "**/*.h5"), recursive=True))
    elif os.path.isfile(PathStr) and PathStr.lower().endswith(".h5"):
        Files = [PathStr]
    else:
        raise FileNotFoundError(f"{PathStr} is not a .h5 file or a directory containing .h5 files")
    if not Files:
        raise FileNotFoundError(f"No .h5 files found under {PathStr}")
    return Files

def SlideIdFromFilename(FilePath):
    return os.path.basename(FilePath).split("-")[0]

def LoadTileFeatures(RootDir, MaxTilesPerSlide=None):
    print(f"Loading tile-level features from: {RootDir}")
    Files = CollectH5FromPath(RootDir)
    SlideData = {}
    Bad = 0
    for I, Fp in enumerate(Files):
        try:
            SlideId = SlideIdFromFilename(Fp)
            X = LoadFeaturesFromH5(Fp, MaxRows=MaxTilesPerSlide, ForceName=ForceFeatureName)
            X = np.asarray(X, dtype=np.float32)
            SlideData[SlideId] = {
                "features": X,
                "filename": os.path.basename(Fp),
                "n_tiles": X.shape[0],
                "feature_dim": X.shape[1],
            }
        except Exception as E:
            Bad += 1
            print(f"[WARN] {Fp}: {E}")
            continue
        if (I + 1) % 20 == 0:
            print(f"  loaded {I + 1}/{len(Files)} files...")
    if not SlideData:
        raise RuntimeError(f"No valid features loaded from {RootDir} (bad files: {Bad}/{len(Files)})")
    Dims = sorted({v["feature_dim"] for v in SlideData.values()})
    print(f"Loaded {len(SlideData)} slides (bad files: {Bad}); feature dims found: {Dims}")
    return SlideData

def MatchSlides(OriginalData, RotatedData):
    Common = sorted(set(OriginalData.keys()) & set(RotatedData.keys()))
    if not Common:
        raise ValueError("No common slides between the two directories")
    # Check dimension match across the pair to ensure consistency 
    for Sid in Common:
        D0 = OriginalData[Sid]["feature_dim"]
        D1 = RotatedData[Sid]["feature_dim"]
        if D0 != D1:
            raise ValueError(f"Feature dims differ for slide {Sid}: {D0} vs {D1}")
    return Common


# for the Cosine box plots 

def RowL2Normalise(X):
    X = X.astype(np.float32)
    Norms = np.maximum(1e-9, LaNorm(X, axis=1, keepdims=True))
    return X / Norms

def Cosine(u, v):
    
         return float(np.dot(u, v))

def ComputeCosineDistributions(OriginalData, RotatedData, CommonSlides):
    """
    Returns:
      tileToMeanOriginal  — list of cosine values for tiles to their own slide mean (Original)
      tileToMeanRotated   — list of cosine values against their slide mean Rotated
      meanVecCosine       — Compares (Original mean vs Rotated mean)
    """
    TileToMeanOriginal, TileToMeanRotated = [], []
    MeanVecCosine = []

    for Sid in CommonSlides:
        X0 = RowL2Normalise(OriginalData[Sid]["features"])
        X1 = RowL2Normalise(RotatedData[Sid]["features"])

        m0 = RowL2Normalise(X0.mean(axis=0, keepdims=True))[0]
        m1 = RowL2Normalise(X1.mean(axis=0, keepdims=True))[0]

        # slide mean cosines
        TileToMeanOriginal.extend((X0 @ m0).tolist())
        TileToMeanRotated.extend((X1 @ m1).tolist())

        # Slide mean compared to slide mean cosine
        MeanVecCosine.append(Cosine(m0, m1))

    return np.array(TileToMeanOriginal, dtype=np.float32), \
           np.array(TileToMeanRotated, dtype=np.float32), \
           np.array(MeanVecCosine, dtype=np.float32)

def PlotCosineBoxplotTileToMean(TileToMeanOriginal, TileToMeanRotated, OutPath):
    # Colours: Original in blue , Rotated  in (red)
    colours = ["#1f77b4","#d62728"]
    labels = ["Original (Tile→Mean)", "Rotated (Tile→Mean)"]

    plt.figure(figsize=(9, 6))
    bp = plt.boxplot([TileToMeanOriginal, TileToMeanRotated],
                     labels=labels, patch_artist=True, showfliers=False)
    for patch, col in zip(bp['boxes'], colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    plt.ylabel("Cosine similarity tile to slide mean")
    plt.title("Cosine: Tile to Slide-Mean (Original vs Rotated)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OutPath, dpi=220)
    plt.close()

def PlotCosineBoxplotMeanVec(MeanVecCosine, OutPath):
    # Single distribution in grey
    plt.figure(figsize=(7, 6))
    bp = plt.boxplot([MeanVecCosine], labels=["Mean(Original) vs Mean(Rotated)"],
                     patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor("#7f7f7f")
    bp['boxes'][0].set_alpha(0.6)
    plt.ylabel("Cosine similarity (slide means)")
    plt.title("Cosine: Slide-Mean(Original) vs Slide-Mean(Rotated)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OutPath, dpi=220)
    plt.close()

def MakeVeryLargeBoxplotAcrossExtractors(PairsCosine, PairNames, OutPath):
    """
    PairsCosine: list of 1D arrays (per pair) of mean-vector cosine across slides
    PairNames:   list of names per pair (e.g., extractor names)
    """
    n = len(PairsCosine)
    if n == 0:
        return
    # Big figure for the larger graph
    width = max(18, 4.5 * n)
    plt.figure(figsize=(width, 8))
    bp = plt.boxplot(PairsCosine, labels=PairNames, patch_artist=True, showfliers=False)
    # Alternate colours
    palette = ["#6baed6", "#9ecae1", "#c6dbef", "#bcbddc", "#9ecae1", "#6baed6"]
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(palette[i % len(palette)])
        box.set_alpha(0.7)
    plt.ylabel("Cosine similarity: Slide-Mean(Original) vs Slide-Mean(Rotated)")
    plt.title("Rotation Robustness Across Extractors (Mean-Vector Cosine per Slide)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OutPath, dpi=220)
    plt.close()


# metrics to tune 

def ComputeTrustworthinessSafe(HighX, LowZ, NumNeighbours=15, Metric="cosine",
                               MaxSamples=10000, Seed=42):
    N = len(HighX)
    if N > MaxSamples:
        Rng = np.random.default_rng(Seed)
        Idx = Rng.choice(N, MaxSamples, replace=False)
        Xs = HighX[Idx]; Zs = LowZ[Idx]
    else:
        Xs, Zs = HighX, LowZ
    return float(trustworthiness(Xs, Zs, n_neighbors=NumNeighbours, metric=Metric))

def ComputeKnnOverlap(HighX, LowZ, K=15, Sample=3000, Seed=42, HighMetric="cosine"):
    Rng = np.random.default_rng(Seed)
    if len(HighX) > Sample:
        Idx = Rng.choice(len(HighX), Sample, replace=False)
        HighX = HighX[Idx]; LowZ = LowZ[Idx]
    KEff = max(1, min(K, len(HighX) - 1))
    NnHigh = NearestNeighbors(n_neighbors=KEff + 1, metric=HighMetric).fit(HighX)
    _, IdxHigh = NnHigh.kneighbors(HighX)
    NnLow = NearestNeighbors(n_neighbors=KEff + 1, metric="euclidean").fit(LowZ)
    _, IdxLow = NnLow.kneighbors(LowZ)
    Overlap = []
    for I in range(len(HighX)):
        A = set(IdxHigh[I, 1:1+KEff]); B = set(IdxLow[I, 1:1+KEff])
        Overlap.append(len(A & B) / float(KEff))
    return float(np.mean(Overlap))

def ComputeAngleSilhouette(LowZ, Labels):
    try:
        Unique = np.unique(Labels)
        if len(Unique) < 2: return None
        if any((Labels == V).sum() < 2 for V in Unique): return None
        return float(silhouette_score(LowZ, Labels))
    except Exception:
        return None

def ComputeTuningScore(HighX, LowZ, Labels, LambdaSil=0.2, KNeighbours=15, HighMetric="cosine",
                       TrustMax=10000, KnnSample=3000, Seed=42):
    Tw  = ComputeTrustworthinessSafe(HighX, LowZ, NumNeighbours=KNeighbours, Metric=HighMetric,
                                     MaxSamples=TrustMax, Seed=Seed)
    Knn = ComputeKnnOverlap(HighX, LowZ, K=KNeighbours, Sample=KnnSample, Seed=Seed, HighMetric=HighMetric)
    Sil = ComputeAngleSilhouette(LowZ, Labels)
    Score = 0.7 * Tw + 0.3 * Knn
    if Sil is not None:
        Score -= LambdaSil * max(0.0, Sil)  # penalise artificial separation by condition
    return {"trustworthiness": Tw, "knn_overlap": Knn, "silhouette": Sil, "score": Score}

# tuning metrics
def MakeTSNE(Perplexity, Seed, LearningRate=200.0):
    sig = inspect.signature(TSNE.__init__)
    kwargs = dict(n_components=2, perplexity=Perplexity, random_state=Seed, init="pca")
    if "n_iter" in sig.parameters:
        kwargs["n_iter"] = 1000
    elif "max_iter" in sig.parameters:
        kwargs["max_iter"] = 1000
    if "learning_rate" in sig.parameters:
        kwargs["learning_rate"] = LearningRate
    return TSNE(**kwargs)

def TuneUMAP(HighX, Labels, ParamGrid, Metric="cosine", Seed=42,
             TuneMax=20000, KNeighbours=15, LambdaSil=0.2):
    if not HaveUMAP:
        return None, []
    N = len(HighX)
    Rng = np.random.default_rng(Seed)
    if N > TuneMax:
        Idx = Rng.choice(N, TuneMax, replace=False)
        Xs = HighX[Idx]; Ys = Labels[Idx]
    else:
        Xs, Ys = HighX, Labels
    Results = []
    Best = {"score": -np.inf, "params": None, "metrics": None}
    for Params in ParamGrid:
        Um = umap.UMAP(
            n_components=2,
            n_neighbors=Params["n_neighbors"],
            min_dist=Params["min_dist"],
            metric=Metric,
            random_state=Seed,
        )
        Z = Um.fit_transform(Xs)
        Met = ComputeTuningScore(
            Xs, Z, Ys, LambdaSil=LambdaSil, KNeighbours=KNeighbours, HighMetric=Metric,
            TrustMax=TrustMaxSamples, KnnSample=KnnPreserveSamples, Seed=Seed
        )
        Results.append({"params": Params, "metrics": Met})
        if Met["score"] > Best["score"]:
            Best = {"score": Met["score"], "params": Params, "metrics": Met}
    return Best, Results

def TuneTSNE(HighX, Labels, ParamGrid, Seed=42, TuneMax=20000, KNeighbours=15, LambdaSil=0.2):
    if not HaveTSNE:
        return None, []
    N = len(HighX)
    Rng = np.random.default_rng(Seed)
    if N > TuneMax:
        Idx = Rng.choice(N, TuneMax, replace=False)
        Xs = HighX[Idx]; Ys = Labels[Idx]
    else:
        Xs, Ys = HighX, Labels
    PerpLimit = (len(Xs) - 1) // 3
    Results = []
    Best = {"score": -np.inf, "params": None, "metrics": None}
    for Params in ParamGrid:
        Perp = min(Params["perplexity"], max(5, PerpLimit))
        Ts = MakeTSNE(Perplexity=Perp, LearningRate=Params.get("learning_rate", 200.0), Seed=Seed)
        Z = Ts.fit_transform(Xs)
        Met = ComputeTuningScore(
            Xs, Z, Ys, LambdaSil=LambdaSil, KNeighbours=KNeighbours,
            HighMetric=("cosine" if UseCosineMetric else "euclidean"),
            TrustMax=TrustMaxSamples, KnnSample=KnnPreserveSamples, Seed=Seed
        )
        Rec = {"params": {"perplexity": Perp, "learning_rate": Params.get("learning_rate", 200.0)}, "metrics": Met}
        Results.append(Rec)
        if Met["score"] > Best["score"]:
            Best = {"score": Met["score"], "params": Rec["params"], "metrics": Met}
    return Best, Results



def PlotEmbedding(LowZ, Labels, Title, OutPath):
    # 0 = Original in blue, 1 = Rotated in red
    Colours = {0: "#1f77b4", 1: "#d62728"}  
    Names   = {0: "Original", 1: "Rotated"}

    plt.figure(figsize=(9, 7))
    for Lab in sorted(np.unique(Labels)):
        Mask = (Labels == Lab)
        plt.scatter(
            LowZ[Mask, 0], LowZ[Mask, 1],
            s=8, alpha=0.6,
            c=Colours[int(Lab)],
            label=Names[int(Lab)],
            edgecolors="none",
            rasterized=True
        )
    plt.legend()
    plt.title(Title)
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OutPath, dpi=220)
    plt.close()


def StackTilesForEmbeddings(OriginalData, RotatedData, CommonSlides):
    AllX, AllY = [], []
    for Sid in CommonSlides:
        X0 = OriginalData[Sid]["features"]
        X1 = RotatedData[Sid]["features"]
        AllX.append(X0); AllY.extend([0] * len(X0))
        AllX.append(X1); AllY.extend([1] * len(X1))
    X = np.vstack(AllX).astype(np.float32)
    Y = np.asarray(AllY, dtype=np.int32)
    # Row L2 normalise for cosine geometry for UMAP
    Norms = np.maximum(1e-9, LaNorm(X, axis=1, keepdims=True))
    X = X / Norms
    return X, Y


def Main():
    print("="*76)
    print("Rotation Robustness — Cosine Boxplots + Tuned UMAP/t-SNE")
    print("="*76)

    # 1) Load both directories (primary pair)
    print("\n1) Loading features for the primary pair...")
    OriginalData = LoadTileFeatures(NoRotationDir, MaxTilesPerSlide=MaxTilesPerSlide)
    RotatedData  = LoadTileFeatures(RotatedDir,    MaxTilesPerSlide=MaxTilesPerSlide)

    # 2) Match slides and check 
    print("\n2) Matching slides...")
    CommonSlides = MatchSlides(OriginalData, RotatedData)
    print(f"   Common slides: {len(CommonSlides)}")

    # 3) Cosine distributions and boxplots 
    print("\n3) Computing cosine distributions (tile→mean & mean↔mean)...")
    TileToMeanOrig, TileToMeanRot, MeanVecCos = ComputeCosineDistributions(OriginalData, RotatedData, CommonSlides)

    PlotCosineBoxplotTileToMean(
        TileToMeanOrig, TileToMeanRot,
        os.path.join(OutputDir, "cosine_boxplot_tile_to_mean.png")
    )
    PlotCosineBoxplotMeanVec(
        MeanVecCos,
        os.path.join(OutputDir, "cosine_boxplot_meanvec_orig_vs_rot.png")
    )
    print("   Saved cosine boxplots for the primary pair.")

    # Large plot comparing all the feature extractors 
    if len(MultiPairOriginalDirs) == len(MultiPairRotatedDirs) and len(MultiPairOriginalDirs) > 0:
        print("\n4) Computing very large summary boxplot across extractor pairs...")
        PairNames, PairDistributions = [], []
        for i, (OrigDir, RotDir) in enumerate(zip(MultiPairOriginalDirs, MultiPairRotatedDirs)):
            try:
                OD = LoadTileFeatures(OrigDir, MaxTilesPerSlide=MaxTilesPerSlide)
                RD = LoadTileFeatures(RotDir,  MaxTilesPerSlide=MaxTilesPerSlide)
                Common = MatchSlides(OD, RD)
                _, _, MV = ComputeCosineDistributions(OD, RD, Common)
                PairDistributions.append(MV)
                if MultiPairNames and i < len(MultiPairNames):
                    PairNames.append(MultiPairNames[i])
                else:
                    PairNames.append(os.path.basename(OrigDir).split("-")[0] or f"Pair{i+1}")
            except Exception as E:
                print(f" Skipping pair {i+1}: {E}")
                continue
        if PairDistributions:
            MakeVeryLargeBoxplotAcrossExtractors(
                PairDistributions, PairNames,
                os.path.join(OutputDir, "cosine_boxplot_across_extractors.png")
            )
            print("   Saved very large summary boxplot across extractor pairs.")
        else:
            print("   No valid pairs to plot.")
    else:
        if len(MultiPairOriginalDirs) != len(MultiPairRotatedDirs):
            print(" Multi-pair lists have different lengths; skipping large summary.")
        else:
            print("4) No multi-pair inputs provided; skipping large summary.")

    # tuning for Umap / t-SNE
    print("\n5) Tuning UMAP/t-SNE and creating embeddings (Original vs Rotated)...")
    X, Y = StackTilesForEmbeddings(OriginalData, RotatedData, CommonSlides)

    
    if HaveUMAP:
        BestUMAP, UMAPResults = TuneUMAP(
            X, Y, UMAPParamGrid,
            Metric=("cosine" if UseCosineMetric else "euclidean"),
            Seed=RandomSeed, TuneMax=TuneMaxSamples,
            KNeighbours=KForMetrics, LambdaSil=LambdaSilhouette
        )
        if BestUMAP["params"]:
            print(f"   Best UMAP: {BestUMAP['params']}  Score={BestUMAP['metrics']['score']:.4f} "
                  f"(tw={BestUMAP['metrics']['trustworthiness']:.3f}, "
                  f"knn={BestUMAP['metrics']['knn_overlap']:.3f}, "
                  f"sil={BestUMAP['metrics']['silhouette']})")
            Um = umap.UMAP(
                n_components=2,
                n_neighbors=BestUMAP["params"]["n_neighbors"],
                min_dist=BestUMAP["params"]["min_dist"],
                metric=("cosine" if UseCosineMetric else "euclidean"),
                random_state=RandomSeed
            )
            ZU = Um.fit_transform(X)
            PlotEmbedding(ZU, Y, "UMAP (tuned)", os.path.join(OutputDir, "umap_tuned.png"))
        else:
            print("   UMAP tuning failed; no plot produced.")
            UMAPResults = []
    else:
        print("   UMAP not available.")
        BestUMAP, UMAPResults = {"params": None, "metrics": None, "score": None}, []

    #  tuning t-SNE
    if HaveTSNE:
        BestTSNE, TSNEResults = TuneTSNE(
            X, Y, TSNEParamGrid,
            Seed=RandomSeed, TuneMax=TuneMaxSamples,
            KNeighbours=KForMetrics, LambdaSil=LambdaSilhouette
        )
        if BestTSNE["params"]:
            print(f"   Best t-SNE: {BestTSNE['params']}  Score={BestTSNE['metrics']['score']:.4f} "
                  f"(tw={BestTSNE['metrics']['trustworthiness']:.3f}, "
                  f"knn={BestTSNE['metrics']['knn_overlap']:.3f}, "
                  f"sil={BestTSNE['metrics']['silhouette']})")
            # t-sne 
            N = len(X)
            Rng = np.random.default_rng(RandomSeed)
            if N > TsneFinalMax:
                Idx = Rng.choice(N, TsneFinalMax, replace=False)
                Xs = X[Idx]; Ys = Y[Idx]
            else:
                Xs, Ys = X, Y
            Perp = min(BestTSNE["params"]["perplexity"], max(5, ((len(Xs) - 1)//3)))
            Ts = MakeTSNE(Perplexity=Perp, LearningRate=BestTSNE["params"]["learning_rate"], Seed=RandomSeed)
            ZT = Ts.fit_transform(Xs)
            PlotEmbedding(ZT, Ys, f"t-SNE (tuned, n={len(Xs)})", os.path.join(OutputDir, "tsne_tuned.png"))
        else:
            print("t-SNE tuning failed; no plot produced.")
            TSNEResults = []
    else:
        print("t-SNE not available.")
        BestTSNE, TSNEResults = {"params": None, "metrics": None, "score": None}, []

    # Saving 
    Summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "output_dir": os.path.abspath(OutputDir),
        "primary_pair": {
            "original_dir": NoRotationDir,
            "rotated_dir": RotatedDir,
            "n_common_slides": len(CommonSlides),
            "tile_to_mean_original_n": int(len(TileToMeanOrig)),
            "tile_to_mean_rotated_n": int(len(TileToMeanRot)),
            "meanvec_cosine_n": int(len(MeanVecCos)),
        },
        "umap": {
            "available": HaveUMAP,
            "best": BestUMAP,
            "grid": UMAPResults,
            "final_image": os.path.join(OutputDir, "umap_tuned.png") if HaveUMAP and BestUMAP.get("params") else None
        },
        "tsne": {
            "available": HaveTSNE,
            "best": BestTSNE,
            "grid": TSNEResults,
            "final_image": os.path.join(OutputDir, "tsne_tuned.png") if HaveTSNE and BestTSNE.get("params") else None
        }
    }
    with open(os.path.join(OutputDir, "tuning_summary.json"), "w") as F:
        json.dump(Summary, F, indent=2)

    print("\nOutputs written to:", os.path.abspath(OutputDir))
    print("cosineBoxplotTiletomean.png")
    print(" cosineBoxplotmeanvecOrigvsRot.png")
    if len(MultiPairOriginalDirs) == len(MultiPairRotatedDirs) and len(MultiPairOriginalDirs) > 0:
        print("cosine_boxplot_across_extractors.png")
    if HaveUMAP and BestUMAP.get("params"):
        print("umapTuned.png")
    if HaveTSNE and BestTSNE.get("params"):
        print("tsneTuned.png")
    print("tuningSummary.json")
    print("Done.")

if __name__ == "__main__":
    Main()
