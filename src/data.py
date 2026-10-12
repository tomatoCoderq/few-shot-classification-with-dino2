import json

import numpy as np
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from src.config import (
    DATA_ROOT,
    IMAGE_SIZE,
    MEAN,
    NUM_CLASSES,
    NUM_TEST,
    NUM_TRAINVAL,
    PROJECT_ROOT,
    RESIZE_SIZE,
    SEEDS,
    SHOTS,
    SPLITS_PATH,
    STD,
    TEST_SPLIT,
    TRAINVAL_SPLIT,
    VAL_PER_CLASS,
    VAL_SEED,
)

PREPROCESS = transforms.Compose(
    [
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)


def load_pets(split):
    """Return the official 'trainval' or 'test' split with breed labels."""
    dataset = OxfordIIITPet(
        DATA_ROOT,
        split=split,
        target_types="category",
        transform=PREPROCESS,
        download=True,
    )
    expected_size = {TRAINVAL_SPLIT: NUM_TRAINVAL, TEST_SPLIT: NUM_TEST}[split]
    assert len(dataset) == expected_size, f"{split}: {len(dataset)} images"
    assert len(dataset.classes) == NUM_CLASSES
    return dataset


def get_labels(dataset):
    """Breed labels without decoding images (torchvision keeps them in _labels)."""
    return np.array(dataset._labels)


def make_splits(trainval_labels):
    """Split trainval into val and train pool, then sample nested support sets.

    For each seed, the support set for K=5 contains the one for K=1, and so on,
    so larger K only adds images.
    """
    rng = np.random.default_rng(VAL_SEED)
    val, train_pool = [], []
    for label in range(NUM_CLASSES):
        class_indices = rng.permutation(np.flatnonzero(trainval_labels == label))
        val.extend(class_indices[:VAL_PER_CLASS])
        train_pool.extend(class_indices[VAL_PER_CLASS:])
    train_pool = np.array(train_pool)

    support = {k: {} for k in SHOTS}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        per_class = [
            rng.permutation(train_pool[trainval_labels[train_pool] == label])
            for label in range(NUM_CLASSES)
        ]
        for k in SHOTS:
            support[k][seed] = sorted(
                int(i) for indices in per_class for i in indices[:k]
            )

    return {
        "val": sorted(int(i) for i in val),
        "train_pool": sorted(int(i) for i in train_pool),
        "support": support,
    }


def check_splits(splits, trainval_labels):
    """Fail loudly if splits overlap, have wrong sizes or are not nested."""
    val = set(splits["val"])
    train_pool = set(splits["train_pool"])
    assert not val & train_pool, "val and train pool overlap"
    assert len(val | train_pool) == NUM_TRAINVAL, "split does not cover trainval"
    assert np.all(np.bincount(trainval_labels[splits["val"]]) == VAL_PER_CLASS)

    for seed in SEEDS:
        previous = set()
        for k in SHOTS:
            indices = splits["support"][k][seed]
            assert set(indices) <= train_pool, f"K={k} seed={seed} leaves train pool"
            counts = np.bincount(trainval_labels[indices], minlength=NUM_CLASSES)
            assert np.all(counts == k), f"K={k} seed={seed} has counts {counts}"
            assert previous <= set(indices), f"K={k} seed={seed} is not nested"
            previous = set(indices)


def save_splits(splits):
    """Write splits to disk. Only call this when deliberately replacing splits.json."""
    SPLITS_PATH.parent.mkdir(exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(splits, indent=1))


def load_splits():
    """Load splits.json; JSON turns K and seed keys into strings, convert back."""
    raw = json.loads(SPLITS_PATH.read_text())
    raw["support"] = {
        int(k): {int(seed): indices for seed, indices in by_seed.items()}
        for k, by_seed in raw["support"].items()
    }
    return raw


def support_subset(trainval, splits, k, seed):
    """The K-shot training images for one seed, as a dataset."""
    return Subset(trainval, splits["support"][k][seed])


if __name__ == "__main__":
    # Validate the committed splits against the local dataset. Splits are not regenerated.
    trainval = load_pets(TRAINVAL_SPLIT)
    test = load_pets(TEST_SPLIT)
    labels = get_labels(trainval)
    splits = load_splits()
    check_splits(splits, labels)

    image, _ = trainval[splits["support"][SHOTS[0]][SEEDS[0]][0]]
    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE), image.shape
    print(f"trainval {len(trainval)}, test {len(test)}, classes {NUM_CLASSES}")
    print(f"val {len(splits['val'])}, train pool {len(splits['train_pool'])}")
    print(
        f"min train pool images per class: "
        f"{np.bincount(labels[splits['train_pool']]).min()}"
    )
    print(f"splits valid: {SPLITS_PATH.relative_to(PROJECT_ROOT)}")
