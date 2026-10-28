# Few-Shot Classification with DINOv2 Embeddings

Computer Vision 2026 course project. We classify the 37 cat and dog breeds of
[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) with only
K = 1, 5, 10, 20 labeled images per breed, using frozen DINOv2 embeddings with
lightweight classifiers, and compare against a supervised ImageNet ResNet-50
transfer-learning baseline. The full plan is in `plan/plan.tex`.

## Setup

Python 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or with pip:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

```bash
uv run python -m src.data
```

This downloads Oxford-IIIT Pet (about 800 MB) into `data/` with torchvision, then
validates the committed splits against the local dataset. It prints:

```
trainval 3680, test 3669, classes 37
val 740, train pool 2940
min train pool images per class: 73
splits valid: splits/splits.json
```

Splits (indices into the official `trainval` split):

- **test**: the official test split, 3669 images, used only for final metrics;
- **val**: 20 images per breed from `trainval` (740 images), for hyperparameters;
- **train pool**: the remaining 2940 `trainval` images;
- **support sets**: K images per breed from the train pool for K = 1, 5, 10, 20 and
  seeds 0–4. For a fixed seed the sets are nested: K = 5 contains K = 1, and so on.

`splits/splits.json` is committed, so everyone uses the same images. `src.data`
only loads and validates this file; it does not regenerate the splits. To
recreate the file from scratch, call `save_splits(make_splits(...))` explicitly.

Preprocessing for every image: resize the shorter side to 256, center crop
224×224, ImageNet mean/std normalization.

Using the data in code:

```python
from src.data import load_pets, load_splits, support_subset

trainval = load_pets("trainval")
test = load_pets("test")
splits = load_splits()
support = support_subset(trainval, splits, k=5, seed=0)  # 37 * 5 images
```

## Baseline (ResNet-50)

All paths, hyperparameters and experiment definitions live in `src/config.py`.

### 1. Cache frozen ResNet-50 embeddings

Used by the linear classifiers (E1/E4). Writes `features/resnet50_{split}_{x,y}.npy`
and the backbone's parameters, GFLOPs and extraction time to `results/backbones.csv`:

```bash
uv run python -m src.features resnet50
```

### 2. Linear probe (E1)

Logistic regression on the cached, L2-normalized embeddings for every K and seed;
C is chosen on validation. Takes about 15 seconds on CPU and writes
`results/linear_probe.csv`:

```bash
uv run python -m src.linear_probe
```

### 3. Fine-tune (E5 and E6)

E5 trains the new head plus `layer4`; E6 trains the whole network. Runs are
deterministic (seeded from `FINETUNE_SEED`) and a restart resumes: runs already
present in `results/finetune.csv` are skipped.

```bash
uv run python -m src.finetune
```

This runs 2 variants × 4 values of K × 5 seeds = 40 runs. Each run writes its
best-validation checkpoint to `checkpoints/{variant}_k{K}_seed{seed}.pt`, one row
to `results/finetune.csv`, and per-epoch progress to `logs/finetune_output.txt`.

To run a single configuration:

```bash
uv run python -c "import src.finetune as f; from src.config import DEVICE; f.run_finetune('layer4', 5, 0, DEVICE)"
```

### 4. Select the best checkpoint

Recomputes validation accuracy for every checkpoint, copies the best to
`checkpoints/best.pt`, evaluates it on test once, and writes
`results/best_model.json`. Selection uses validation only, never test.

```bash
uv run python -m src.select_best
```

## DINOv2 and lightweight classifiers

### 1. Cache frozen DINOv2 embeddings

Weights come from `torch.hub` (`facebookresearch/dinov2`, downloaded to
`~/.cache/torch/hub` on the first run); the embedding is the CLS token. Same
preprocessing as ResNet-50. About 50 s (ViT-S/14) and 95 s (ViT-B/14) on an
RTX 4060 Laptop GPU:

```bash
uv run python -m src.features dinov2_vits14
uv run python -m src.features dinov2_vitb14
```

### 2. Classifiers on frozen embeddings (E2, E3, E4, E7, E7b)

Logistic regression (C on validation, same as E1), prototype (nearest class
mean, cosine) and cosine k-NN (k ≤ K on validation) for every K and seed.
Experiments are defined in `FROZEN_EXPERIMENTS` in `src/config.py`; each takes
seconds on CPU and writes `results/frozen_{experiment}.csv`:

```bash
uv run python -m src.classifiers E2   # ViT-S/14 + logreg
uv run python -m src.classifiers E3   # ViT-S/14 + prototype, k-NN
uv run python -m src.classifiers E4   # ResNet-50 + prototype, k-NN
uv run python -m src.classifiers E7   # ViT-B/14 + prototype
uv run python -m src.classifiers E7b  # ViT-B/14 + logreg
```

### Outputs

- `features/`: cached embeddings (ignored by git);
- `checkpoints/`: fine-tuned weights (ignored by git);
- `results/`: `linear_probe.csv`, `finetune.csv`, `checkpoint_val.csv`, `best_model.json`,
  `backbones.csv`, `frozen_e*.csv` (committed);
- `logs/`: run logs (ignored by git).
