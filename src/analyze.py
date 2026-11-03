"""Export verifiable K=20 predictions without replacing experiment metrics.

Run ``python -m src.analyze --frozen E1 E2 E7b`` after caching embeddings.
Seed 0 is fixed in advance for the frozen-model comparison. The saved best
fine-tuned checkpoint can be inspected with ``--best-checkpoint``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import BEST_CHECKPOINT_PATH, BEST_RESULT_PATH, DEVICE, NUM_CLASSES
from src.data import check_splits, get_labels, load_pets, load_splits
from src.evaluate import (
    ANALYSIS_DIR,
    RESULTS_DIR,
    predictions_and_features,
    write_prediction_analysis,
)
from src.finetune import macro_f1

FROZEN = {
    "E1": ("resnet50", "linear_probe.csv"),
    "E2": ("dinov2_vits14", "frozen_e2.csv"),
    "E7b": ("dinov2_vitb14", "frozen_e7b.csv"),
}


def write_predictions(experiment, seed, labels, predictions, confidence,
                      test_features, support_features, trainval, test, support,
                      reference, provenance):
    """Save predictions, source metrics and the predicted-class support neighbour."""
    assert len(labels) == len(test) == 3669
    assert np.array_equal(labels, get_labels(test))
    accuracy = float(np.mean(labels == predictions))
    f1 = macro_f1(labels, predictions)
    # MPS/CUDA embeddings can produce small numerical differences. Preserve and
    # report the difference; a larger discrepancy needs investigation.
    if abs(accuracy - float(reference["accuracy"])) > 0.005:
        raise ValueError(f"{experiment}: prediction accuracy disagrees with CSV: "
                         f"{accuracy} vs {reference['accuracy']}")
    support_labels = get_labels(trainval)[support]
    rows = []
    image_root = Path(__file__).resolve().parent.parent
    for index, (true, pred, score) in enumerate(zip(labels, predictions, confidence)):
        candidates = np.flatnonzero(support_labels == pred)
        similarities = support_features[candidates] @ test_features[index]
        neighbour = candidates[similarities.argmax()]
        rows.append({
            "experiment": experiment, "K": 20, "seed": seed,
            "test_index": index, "true_label": true, "pred_label": pred,
            "true_breed": test.classes[true], "pred_breed": test.classes[pred],
            "confidence": score,
            "image_path": str(test._images[index].relative_to(image_root)),
            "nearest_support_path": str(trainval._images[support[neighbour]].relative_to(image_root)),
            "nearest_support_breed": test.classes[pred],
            "nearest_support_cosine": float(similarities.max()),
        })
    output = RESULTS_DIR / "predictions" / f"{experiment}_k20_seed{seed}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    write_prediction_analysis(output)
    audit = {
        "experiment": experiment, "K": 20, "seed": seed,
        "prediction_accuracy": accuracy, "prediction_macro_f1": f1,
        "recorded_accuracy": float(reference["accuracy"]),
        "recorded_macro_f1": float(reference["macro_f1"]),
        "accuracy_difference_pp": 100 * (accuracy - float(reference["accuracy"])),
        "predictions": str(output.relative_to(image_root)),
        "device": str(DEVICE), **provenance,
    }
    audit_path = ANALYSIS_DIR / f"{experiment}_k20_seed{seed}_provenance.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit), flush=True)


def analyze_frozen(experiment, seed=0):
    """Refit the original validation-selected C on the fixed support set."""
    from src.classifiers import load_features
    from src.config import LINEAR_PROBE_MAX_ITER

    backbone, filename = FROZEN[experiment]
    trainval, test = load_pets("trainval"), load_pets("test")
    splits = load_splits()
    check_splits(splits, get_labels(trainval))
    x_train, y_train = load_features(backbone, "trainval")
    x_test, y_test = load_features(backbone, "test")
    assert np.array_equal(y_train, get_labels(trainval))
    assert np.array_equal(y_test, get_labels(test))
    rows = pd.read_csv(RESULTS_DIR / filename)
    selected = rows[(rows.K == 20) & (rows.seed == seed)]
    assert len(selected) == 1
    reference = selected.iloc[0]
    c = float(reference["C"]) if experiment == "E1" else float(reference.hparam.split("=")[1])
    support = splits["support"][20][seed]
    model = LogisticRegression(C=c, max_iter=LINEAR_PROBE_MAX_ITER)
    model.fit(x_train[support], y_train[support])
    probabilities = model.predict_proba(x_test)
    predictions = model.classes_[probabilities.argmax(axis=1)]
    write_predictions(
        experiment, seed, y_test, predictions, probabilities.max(axis=1),
        x_test, x_train[support], trainval, test, support, reference,
        {"backbone": backbone, "classifier": "logreg", "C": c,
         "selection": "seed 0 fixed in advance; C from original validation-selected CSV"},
    )


def analyze_best_checkpoint():
    """Use the saved checkpoint directly; do not download pretrained weights."""
    import torch
    from torch.utils.data import Subset
    from torchvision.models import resnet50

    metadata = json.loads(BEST_RESULT_PATH.read_text())
    assert metadata["K"] == 20
    seed, experiment = int(metadata["seed"]), metadata["experiment"]
    model = resnet50(weights=None, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(BEST_CHECKPOINT_PATH, map_location="cpu", weights_only=True))
    model = model.to(DEVICE).eval()
    trainval, test = load_pets("trainval"), load_pets("test")
    support = load_splits()["support"][20][seed]
    labels, predictions, confidence, test_features = predictions_and_features(model, test)
    _, _, _, support_features = predictions_and_features(model, Subset(trainval, support))
    recorded = pd.read_csv(RESULTS_DIR / "finetune.csv")
    reference = recorded[(recorded.experiment == experiment) &
                         (recorded.K == 20) & (recorded.seed == seed)].iloc[0]
    write_predictions(
        experiment, seed, labels, predictions, confidence, test_features,
        support_features, trainval, test, support, reference,
        {"checkpoint": "checkpoints/best.pt", "selection": "best validation checkpoint"},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", nargs="+", choices=list(FROZEN))
    parser.add_argument("--best-checkpoint", action="store_true")
    args = parser.parse_args()
    if not args.frozen and not args.best_checkpoint:
        parser.error("choose --frozen and/or --best-checkpoint")
    for experiment in args.frozen or []:
        analyze_frozen(experiment)
    if args.best_checkpoint:
        analyze_best_checkpoint()
