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
builds the splits and checks them. It prints:

```
trainval 3680, test 3669, classes 37
val 740, train pool 2940
min train pool images per class: 73
splits saved to splits/splits.json
```

Splits (indices into the official `trainval` split):

- **test**: the official test split, 3669 images, used only for final metrics;
- **val**: 20 images per breed from `trainval` (740 images), for hyperparameters;
- **train pool**: the remaining 2940 `trainval` images;
- **support sets**: K images per breed from the train pool for K = 1, 5, 10, 20 and
  seeds 0–4. For a fixed seed the sets are nested: K = 5 contains K = 1, and so on.

`splits/splits.json` is committed, so everyone uses the same images. Rerunning
`src.data` regenerates the splits and fails if they differ from the saved file.

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