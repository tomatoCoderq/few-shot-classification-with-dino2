"""Lightweight classifiers on frozen embeddings (E2, E3, E4, E7).

Needs the cached embeddings from `python -m src.features <backbone>`.
Experiments are defined in FROZEN_EXPERIMENTS in src/config.py.

Run: python -m src.classifiers E2
"""

import argparse
import csv
import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from src.config import (
    BACKBONES_RESULTS_PATH,
    FEATURE_FILE,
    FEATURES_DIR,
    FROZEN_EMBEDDING_DIMS,
    FROZEN_EXPERIMENTS,
    FROZEN_RESULTS_FILE,
    KNN_K_GRID,
    NUM_CLASSES,
    RESULTS_DIR,
    SEEDS,
    SHOTS,
    TEST_SPLIT,
    TRAINVAL_SPLIT,
)
from src.data import get_labels, load_pets, load_splits
from src.finetune import macro_f1
# Importing linear_probe also turns sklearn ConvergenceWarning into an error.
from src.linear_probe import fit_best_c

RESULT_FIELDS = [
    "experiment",
    "backbone",
    "classifier",
    "K",
    "seed",
    "hparam",
    "best_val_accuracy",
    "accuracy",
    "macro_f1",
    "training_time_s",
    "device",
    "trainable_params",
    "total_params",
]


def load_features(backbone, split):
    """L2-normalized embeddings and labels saved by src.features."""
    x = np.load(FEATURES_DIR / FEATURE_FILE.format(backbone=backbone, split=split, kind="x"))
    y = np.load(FEATURES_DIR / FEATURE_FILE.format(backbone=backbone, split=split, kind="y"))
    assert x.shape == (len(y), FROZEN_EMBEDDING_DIMS[backbone]), x.shape
    return x / np.linalg.norm(x, axis=1, keepdims=True), y


def backbone_params(backbone):
    """Total backbone parameters, as recorded by src.features."""
    with BACKBONES_RESULTS_PATH.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["backbone"] == backbone]
    assert len(rows) == 1, f"run `python -m src.features {backbone}` first"
    return int(rows[0]["total_params"])


def accuracy(predict, x, y):
    return float(np.mean(predict(x) == y))


def fit_logreg(x_train, y_train, x_val, y_val):
    """Multinomial logistic regression, C chosen on validation (same as E1)."""
    model, c, val_acc = fit_best_c(x_train, y_train, x_val, y_val)
    return model.predict, f"C={c}", val_acc


def fit_prototype(x_train, y_train, x_val, y_val):
    """Nearest class mean: highest cosine similarity to the normalized class mean.

    No hyperparameters; validation accuracy is only reported.
    """
    prototypes = np.stack([x_train[y_train == c].mean(axis=0) for c in range(NUM_CLASSES)])
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)

    def predict(x):
        return np.argmax(x @ prototypes.T, axis=1)

    return predict, "-", accuracy(predict, x_val, y_val)


def fit_knn(x_train, y_train, x_val, y_val):
    """Cosine k-NN, votes weighted by inverse distance; k <= K chosen on validation."""
    k_shot = len(y_train) // NUM_CLASSES
    best_model, best_k, best_acc = None, None, -1.0
    for k in (k for k in KNN_K_GRID if k <= k_shot):
        model = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
        model.fit(x_train, y_train)
        acc = accuracy(model.predict, x_val, y_val)
        if acc > best_acc:
            best_model, best_k, best_acc = model, k, acc
    return best_model.predict, f"k={best_k}", best_acc


CLASSIFIERS = {"logreg": fit_logreg, "prototype": fit_prototype, "knn": fit_knn}


def fitted_params(classifier, dim):
    """Parameters fitted on the support set; k-NN only stores the support embeddings."""
    return {
        "logreg": dim * NUM_CLASSES + NUM_CLASSES,
        "prototype": dim * NUM_CLASSES,
        "knn": 0,
    }[classifier]


def main(experiment):
    backbone, classifiers = FROZEN_EXPERIMENTS[experiment]
    x_trainval, y_trainval = load_features(backbone, TRAINVAL_SPLIT)
    x_test, y_test = load_features(backbone, TEST_SPLIT)
    # Split indices refer to the dataset order, so the cached features must follow it.
    assert np.array_equal(y_trainval, get_labels(load_pets(TRAINVAL_SPLIT)))
    assert np.array_equal(y_test, get_labels(load_pets(TEST_SPLIT)))

    splits = load_splits()
    x_val, y_val = x_trainval[splits["val"]], y_trainval[splits["val"]]
    dim = FROZEN_EMBEDDING_DIMS[backbone]
    total_backbone = backbone_params(backbone)

    rows = []
    for classifier in classifiers:
        for k in SHOTS:
            for seed in SEEDS:
                support = splits["support"][k][seed]
                start = time.time()
                predict, hparam, val_acc = CLASSIFIERS[classifier](
                    x_trainval[support], y_trainval[support], x_val, y_val
                )
                training_time = time.time() - start
                preds = predict(x_test)
                rows.append({
                    "experiment": experiment,
                    "backbone": backbone,
                    "classifier": classifier,
                    "K": k,
                    "seed": seed,
                    "hparam": hparam,
                    "best_val_accuracy": val_acc,
                    "accuracy": float(np.mean(preds == y_test)),
                    "macro_f1": macro_f1(y_test, preds),
                    "training_time_s": training_time,
                    "device": "cpu",
                    "trainable_params": fitted_params(classifier, dim),
                    "total_params": total_backbone + fitted_params(classifier, dim),
                })
                print(f"[{experiment} {classifier} K={k} seed={seed}] {hparam} "
                      f"val {val_acc:.3f} test {rows[-1]['accuracy']:.3f}", flush=True)

    path = RESULTS_DIR / FROZEN_RESULTS_FILE.format(experiment=experiment.lower())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=list(FROZEN_EXPERIMENTS))
    main(parser.parse_args().experiment)
