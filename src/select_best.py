"""Pick the fine-tuned checkpoint with the best validation accuracy and test it once.

Validation accuracy is recomputed from the saved weights, so selection never
looks at the test split. Only the selected checkpoint is evaluated on test.

Run: python -m src.select_best
"""

import csv
import json
import shutil

import numpy as np
from torch.utils.data import Subset

from src.config import (
    BEST_CHECKPOINT_PATH,
    BEST_RESULT_PATH,
    CHECKPOINT_DIR,
    DEVICE,
    RESULTS_DIR,
    SEEDS,
    SHOTS,
    VAL_RESULTS_PATH,
    VARIANTS,
)
from src.finetune import (
    EVAL_TRANSFORM,
    build_model,
    checkpoint_path,
    load_splits,
    load_test,
    load_trainval,
    load_weights,
    macro_f1,
    predict,
)


def val_accuracy(model, val_set):
    preds, labels = predict(model, val_set, DEVICE)
    return float(np.mean(preds == labels))


def main():
    splits = load_splits()
    val_set = Subset(load_trainval(EVAL_TRANSFORM), splits["val"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in VARIANTS:
        for k in SHOTS:
            for seed in SEEDS:
                path = checkpoint_path(variant, k, seed)
                model = load_weights(build_model(variant), path, DEVICE)
                acc = val_accuracy(model, val_set)
                rows.append(
                    {
                        "variant": variant,
                        "K": k,
                        "seed": seed,
                        "val_accuracy": acc,
                        "checkpoint": path.name,
                    }
                )
                print(f"{path.name}: val_accuracy {acc:.4f}", flush=True)

    with VAL_RESULTS_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda r: r["val_accuracy"])
    best_path = checkpoint_path(best["variant"], best["K"], best["seed"])
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best_path, BEST_CHECKPOINT_PATH)

    model = load_weights(build_model(best["variant"]), best_path, DEVICE)
    preds, labels = predict(model, load_test(), DEVICE)
    result = {
        "experiment": VARIANTS[best["variant"]],
        "variant": best["variant"],
        "K": best["K"],
        "seed": best["seed"],
        "val_accuracy": best["val_accuracy"],
        "test_accuracy": float(np.mean(preds == labels)),
        "macro_f1": macro_f1(labels, preds),
        "checkpoint": BEST_CHECKPOINT_PATH.name,
    }
    BEST_RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
