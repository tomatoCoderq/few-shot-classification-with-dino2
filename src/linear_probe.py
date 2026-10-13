"""E1: logistic regression on frozen ResNet-50 embeddings.

Needs the cached embeddings from `python -m src.features`.

Run: python -m src.linear_probe
"""

import csv
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.config import (
    BACKBONE,
    EMBEDDING_DIM,
    FEATURE_FILE,
    FEATURES_DIR,
    LINEAR_PROBE_C_GRID,
    LINEAR_PROBE_MAX_ITER,
    LINEAR_PROBE_RESULTS_PATH,
    NUM_CLASSES,
    SEEDS,
    SHOTS,
    TEST_SPLIT,
    TRAINVAL_SPLIT,
)
from src.data import get_labels, load_pets, load_splits
from src.features import count_params, load_backbone
from src.finetune import macro_f1

# A model that did not converge is a broken result, not a warning.
warnings.simplefilter("error", ConvergenceWarning)
# At K=1 there are as many classes as samples, which sklearn flags as a possible regression.
warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")

RESULT_FIELDS = [
    "experiment",
    "backbone",
    "classifier",
    "K",
    "seed",
    "C",
    "best_val_accuracy",
    "accuracy",
    "macro_f1",
    "training_time_s",
    "device",
    "trainable_params",
    "total_params",
]


def load_features(split):
    """L2-normalized embeddings and labels saved by src.features."""
    x = np.load(FEATURES_DIR / FEATURE_FILE.format(backbone=BACKBONE, split=split, kind="x"))
    y = np.load(FEATURES_DIR / FEATURE_FILE.format(backbone=BACKBONE, split=split, kind="y"))
    assert x.shape == (len(y), EMBEDDING_DIM), x.shape
    return x / np.linalg.norm(x, axis=1, keepdims=True), y


def fit_best_c(x_train, y_train, x_val, y_val):
    """Fit one model per C and keep the one with the best validation accuracy."""
    best_model, best_c, best_acc = None, None, -1.0
    for c in LINEAR_PROBE_C_GRID:
        model = LogisticRegression(C=c, max_iter=LINEAR_PROBE_MAX_ITER)
        model.fit(x_train, y_train)
        acc = float(np.mean(model.predict(x_val) == y_val))
        if acc > best_acc:
            best_model, best_c, best_acc = model, c, acc
    return best_model, best_c, best_acc


def main():
    x_trainval, y_trainval = load_features(TRAINVAL_SPLIT)
    x_test, y_test = load_features(TEST_SPLIT)
    # Split indices refer to the dataset order, so the cached features must follow it.
    assert np.array_equal(y_trainval, get_labels(load_pets(TRAINVAL_SPLIT)))
    assert np.array_equal(y_test, get_labels(load_pets(TEST_SPLIT)))

    splits = load_splits()
    x_val, y_val = x_trainval[splits["val"]], y_trainval[splits["val"]]
    head_params = EMBEDDING_DIM * NUM_CLASSES + NUM_CLASSES
    total_params = count_params(load_backbone("cpu"))[0] + head_params

    rows = []
    for k in SHOTS:
        for seed in SEEDS:
            support = splits["support"][k][seed]
            start = time.time()
            model, c, val_acc = fit_best_c(
                x_trainval[support], y_trainval[support], x_val, y_val
            )
            training_time = time.time() - start
            preds = model.predict(x_test)
            rows.append({
                "experiment": "E1",
                "backbone": BACKBONE,
                "classifier": "logreg",
                "K": k,
                "seed": seed,
                "C": c,
                "best_val_accuracy": val_acc,
                "accuracy": float(np.mean(preds == y_test)),
                "macro_f1": macro_f1(y_test, preds),
                "training_time_s": training_time,
                "device": "cpu",
                "trainable_params": head_params,
                "total_params": total_params,
            })
            print(f"[E1 K={k} seed={seed}] C={c} val {val_acc:.3f} "
                  f"test {rows[-1]['accuracy']:.3f}", flush=True)

    LINEAR_PROBE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LINEAR_PROBE_RESULTS_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {LINEAR_PROBE_RESULTS_PATH}")


if __name__ == "__main__":
    main()
