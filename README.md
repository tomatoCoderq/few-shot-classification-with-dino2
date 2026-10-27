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

Used by the linear classifiers (E1/E4). Writes `features/resnet50_{split}_{x,y}.npy`:

```bash
uv run python -m src.features
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

### Outputs

- `features/`: cached embeddings (ignored by git);
- `checkpoints/`: fine-tuned weights (ignored by git);
- `results/`: `linear_probe.csv`, `finetune.csv`, `checkpoint_val.csv`, `best_model.json`,
  plus evaluator outputs `summary_metrics.csv`, `efficiency_k20.csv`, plots, and `analysis/` (committed);
- `logs/`: run logs (ignored by git).

## Evaluation

`src.evaluate` does not train models. It reads the committed per-run CSVs,
recomputes mean +/- sample standard deviation over the five support-set seeds,
and rebuilds the efficiency table and accuracy/macro-F1 plots:

```bash
uv run python -m src.evaluate
```

This writes `results/summary_metrics.csv`, `results/efficiency_k20.csv`,
`results/accuracy_vs_k.png`, and `results/macro_f1_vs_k.png`. Only CSV files
with the required per-run metric schema are included; incomplete or unrelated
CSV files are not silently treated as experiment results.

If a temporary `report_results.json` bundle is present, copy its per-run rows
into `linear_probe.csv` / `finetune.csv` and then recompute the summaries:

```bash
uv run python -m src.evaluate --report-results report_results.json
```

For the required K=20 per-breed analysis, confusion matrix, and three visual
failure cases with nearest support images, evaluate an existing fine-tuning
checkpoint (the checkpoint and dataset must be available):

```bash
uv run python -m src.evaluate --checkpoint E5 20 4
```

It saves the raw predictions in `results/predictions/` and the per-breed table,
confusion matrix, frequent-confusion failure-case table, and image panel in
`results/analysis/`. The selected examples are the most confident representative
errors of the three most frequent true-to-predicted breed pairs; inspect them
before assigning a visual cause in the final report.
