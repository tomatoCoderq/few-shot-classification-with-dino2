from pathlib import Path

import torch

# Paths (all relative to the project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
SPLITS_PATH = PROJECT_ROOT / "splits" / "splits.json"
FEATURES_DIR = PROJECT_ROOT / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
ARCHIVE_DIR = PROJECT_ROOT / "archive"

FINETUNE_LOG_PATH = LOGS_DIR / "finetune_output.txt"

FINETUNE_RESULTS_PATH = RESULTS_DIR / "finetune.csv"
VAL_RESULTS_PATH = RESULTS_DIR / "checkpoint_val.csv"
BEST_RESULT_PATH = RESULTS_DIR / "best_model.json"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best.pt"
LINEAR_PROBE_RESULTS_PATH = RESULTS_DIR / "linear_probe.csv"
BACKBONES_RESULTS_PATH = RESULTS_DIR / "backbones.csv"

# File name templates
FEATURE_FILE = "{backbone}_{split}_{kind}.npy"  # kind: "x" (embeddings) or "y" (labels)
CHECKPOINT_FILE = "{variant}_k{k}_seed{seed}.pt"
FROZEN_RESULTS_FILE = "frozen_{experiment}.csv"

# Dataset (Oxford-IIIT Pet)
TRAINVAL_SPLIT = "trainval"
TEST_SPLIT = "test"
NUM_CLASSES = 37
NUM_TRAINVAL = 3680
NUM_TEST = 3669
VAL_PER_CLASS = 20
VAL_SEED = 2026
SHOTS = (1, 5, 10, 20)
SEEDS = (0, 1, 2, 3, 4)

# Preprocessing (224 is a multiple of the ViT patch size 14, so ResNet and ViT get the same input)
RESIZE_SIZE = 256
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Backbone
BACKBONE = "resnet50"
PRETRAINED_WEIGHTS = "IMAGENET1K_V2"  # torchvision ResNet50_Weights member name
EMBEDDING_DIM = 2048

# Frozen backbones for cached embeddings: name -> embedding size.
# DINOv2 names are torch.hub entry points; their output is the normalized CLS token.
FROZEN_EMBEDDING_DIMS = {
    BACKBONE: EMBEDDING_DIM,
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
}
DINOV2_REPO = "facebookresearch/dinov2"

# Experiments
CLASSIFIER = "finetune"
# variant name -> experiment id in the results table
VARIANTS = {"layer4": "E5", "full": "E6"}
# parameter-name prefixes that stay trainable; "" matches every parameter
VARIANT_TRAINABLE = {"layer4": ("layer4", "fc"), "full": ("",)}
HEAD_PREFIX = "fc"

# Linear probe (E1): C is chosen on validation
LINEAR_PROBE_C_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
LINEAR_PROBE_MAX_ITER = 5000

# Lightweight classifiers on frozen embeddings: experiment id -> (backbone, classifiers)
FROZEN_EXPERIMENTS = {
    "E2": ("dinov2_vits14", ("logreg",)),
    "E3": ("dinov2_vits14", ("prototype", "knn")),
    "E4": (BACKBONE, ("prototype", "knn")),
    # prototype: best E3 classifier by mean validation accuracy
    "E7": ("dinov2_vitb14", ("prototype",)),
    "E7b": ("dinov2_vitb14", ("logreg",)),
}
# k-NN: number of neighbours chosen on validation, only values k <= K are tried
KNN_K_GRID = (1, 3, 5, 10, 20)

# Fine-tuning
EPOCHS = 30
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128
HEAD_LR = 1e-3
BACKBONE_LR = 1e-4
WEIGHT_DECAY = 1e-4
# Training seed for run `seed` is FINETUNE_SEED + seed (head init, shuffling, augmentation)
FINETUNE_SEED = 0

# Feature extraction
FEATURE_BATCH_SIZE = 64

# Runtime
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
NUM_WORKERS = 4
# Required for deterministic cuBLAS on CUDA
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
