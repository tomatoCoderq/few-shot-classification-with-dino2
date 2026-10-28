"""Aggregate experiment metrics and create reproducible K=20 error analyses.

Run ``python -m src.evaluate`` after per-run CSVs are in ``results/``. It
recomputes mean +/- sample standard deviation over the five seeds, writes the
summary tables, and rebuilds the accuracy and macro-F1 plots. No training is
performed.

To analyse a fine-tuned checkpoint, use
``python -m src.evaluate --checkpoint E5 20 4``.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LINEAR_PROBE_PATH = RESULTS_DIR / "linear_probe.csv"
FINETUNE_PATH = RESULTS_DIR / "finetune.csv"
BEST_MODEL_PATH = RESULTS_DIR / "best_model.json"
CHECKPOINT_VAL_PATH = RESULTS_DIR / "checkpoint_val.csv"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

METRIC_COLUMNS = {
    "experiment", "backbone", "classifier", "K", "seed", "accuracy",
    "macro_f1", "training_time_s", "device", "trainable_params", "total_params",
}
LINEAR_PROBE_FIELDS = [
    "experiment", "backbone", "classifier", "K", "seed", "C",
    "best_val_accuracy", "accuracy", "macro_f1", "training_time_s", "device",
    "trainable_params", "total_params",
]
FINETUNE_FIELDS = [
    "experiment", "backbone", "classifier", "K", "seed", "train_seed",
    "best_epoch", "best_val_accuracy", "accuracy", "macro_f1",
    "training_time_s", "device", "trainable_params", "total_params",
]
CHECKPOINT_VAL_FIELDS = ["variant", "K", "seed", "val_accuracy", "checkpoint"]
PREDICTION_COLUMNS = {
    "experiment", "K", "seed", "test_index", "true_label", "pred_label",
    "true_breed", "pred_breed", "confidence", "image_path",
    "nearest_support_path", "nearest_support_breed", "nearest_support_cosine",
}
EXPERIMENT_VARIANTS = {"E5": "layer4", "E6": "full"}
SKIP_METRIC_FILES = {
    "summary_metrics.csv", "efficiency_k20.csv", "checkpoint_val.csv",
}


def metric_frames(results_dir=RESULTS_DIR):
    """Load only result files that contain a complete per-run metric schema."""
    frames = []
    for path in sorted(results_dir.glob("*.csv")):
        if path.name in SKIP_METRIC_FILES:
            continue
        frame = pd.read_csv(path)
        if METRIC_COLUMNS <= set(frame.columns):
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No metric CSVs found in {results_dir}")
    metrics = pd.concat(frames, ignore_index=True)
    metrics["K"] = metrics["K"].astype(int)
    metrics["seed"] = metrics["seed"].astype(int)
    return metrics.drop_duplicates(["experiment", "K", "seed"], keep="last")


def summarize_metrics(metrics):
    """Return mean and sample standard deviation over the five support sets."""
    group_columns = ["experiment", "backbone", "classifier", "K"]
    summary = metrics.groupby(group_columns, as_index=False).agg(
        runs=("seed", "nunique"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        training_time_s_mean=("training_time_s", "mean"),
        training_time_s_std=("training_time_s", "std"),
        trainable_params=("trainable_params", "max"),
        total_params=("total_params", "max"),
        devices=("device", lambda values: ", ".join(sorted(set(values)))),
    )
    assert (summary["runs"] == 5).all(), "Every summary cell must have five seeds"
    return summary.sort_values(["experiment", "K"]).reset_index(drop=True)


def plot_curves(summary, metric, output_path):
    """Plot one mean +/- std curve per experiment."""
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"
    figure, axis = plt.subplots(figsize=(8, 5))
    for experiment, rows in summary.groupby("experiment", sort=True):
        rows = rows.sort_values("K")
        axis.errorbar(
            rows["K"], rows[mean_column] * 100, yerr=rows[std_column] * 100,
            marker="o", capsize=3, label=experiment,
        )
    axis.set(xlabel="K (labeled images per breed)", ylabel=f"Test {metric.replace('_', '-')} (%)")
    axis.set_xticks(sorted(summary["K"].unique()))
    axis.grid(alpha=0.3)
    axis.legend(title="Experiment")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_summary_outputs(summary):
    """Write summary tables and plots from an already aggregated dataframe."""
    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
    efficiency_columns = [
        "experiment", "backbone", "classifier", "runs", "training_time_s_mean",
        "training_time_s_std", "trainable_params", "total_params", "devices",
    ]
    efficiency = summary[summary["K"] == 20][efficiency_columns]
    efficiency.to_csv(RESULTS_DIR / "efficiency_k20.csv", index=False)
    plot_curves(summary, "accuracy", RESULTS_DIR / "accuracy_vs_k.png")
    plot_curves(summary, "macro_f1", RESULTS_DIR / "macro_f1_vs_k.png")


def write_summary_artifacts():
    """Write result tables and accuracy/macro-F1 plots from all available runs."""
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics = metric_frames()
    summary = summarize_metrics(metrics)
    write_summary_outputs(summary)
    return metrics, summary


def _write_csv(path, frame, fields):
    missing = [name for name in fields if name not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    ordered = frame[fields].sort_values(
        [column for column in ("experiment", "variant", "K", "seed") if column in fields]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(path, index=False)


def write_experiment_csvs(metrics):
    """Persist per-run rows in the experiment CSV layout used by training scripts."""
    linear = metrics[metrics["experiment"] == "E1"].copy()
    finetune = metrics[metrics["experiment"].isin(EXPERIMENT_VARIANTS)].copy()
    if linear.empty and finetune.empty:
        raise ValueError("No E1/E5/E6 per-run metrics to write")
    if not linear.empty:
        _write_csv(LINEAR_PROBE_PATH, linear, LINEAR_PROBE_FIELDS)
    if not finetune.empty:
        _write_csv(FINETUNE_PATH, finetune, FINETUNE_FIELDS)
        val_rows = []
        for _, row in finetune.iterrows():
            variant = EXPERIMENT_VARIANTS[row["experiment"]]
            val_rows.append({
                "variant": variant,
                "K": int(row["K"]),
                "seed": int(row["seed"]),
                "val_accuracy": row["best_val_accuracy"],
                "checkpoint": f"{variant}_k{int(row['K'])}_seed{int(row['seed'])}.pt",
            })
        _write_csv(CHECKPOINT_VAL_PATH, pd.DataFrame(val_rows), CHECKPOINT_VAL_FIELDS)


def write_confusion_plot(matrix, names, output_path):
    figure, axis = plt.subplots(figsize=(14, 12))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Test images")
    axis.set(
        xticks=range(len(names)), yticks=range(len(names)), xticklabels=names,
        yticklabels=names, xlabel="Predicted breed", ylabel="True breed",
    )
    plt.setp(axis.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(axis.get_yticklabels(), fontsize=6)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def write_bundle_analysis(analysis):
    """Write K=20 per-breed and confusion artifacts from a portable analysis dump."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = analysis["run"]
    per_breed = pd.DataFrame(analysis["per_breed"])
    per_breed.to_csv(ANALYSIS_DIR / f"{run_name}_per_breed.csv", index=False)
    matrix = np.asarray(analysis["confusion_matrix"], dtype=int)
    np.savetxt(ANALYSIS_DIR / f"{run_name}_confusion.csv", matrix, fmt="%d", delimiter=",")
    names = [item["true_breed"] for item in analysis["class_labels"]]
    write_confusion_plot(matrix, names, ANALYSIS_DIR / f"{run_name}_confusion.png")
    failures = pd.DataFrame(analysis["failure_cases"])
    if "image_path" in failures.columns:
        failures["image_path"] = failures["image_path"].map(lambda path: Path(path).name)
    if "nearest_support_path" in failures.columns:
        failures["nearest_support_path"] = failures["nearest_support_path"].map(
            lambda path: Path(path).name
        )
    failures.to_csv(ANALYSIS_DIR / f"{run_name}_failure_cases.csv", index=False)


def import_report_results(report_path):
    """Copy per-run rows out of a temporary bundle, then recompute summaries."""
    bundle = json.loads(report_path.read_text())
    required = {"per_run_metrics"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"{report_path} misses keys: {sorted(missing)}")
    metrics = pd.DataFrame(bundle["per_run_metrics"])
    if metrics.empty:
        raise ValueError(f"{report_path} contains no experiment metrics")
    if not METRIC_COLUMNS <= set(metrics.columns):
        raise ValueError(f"{report_path} has an invalid per-run metric schema")
    write_experiment_csvs(metrics)
    if bundle.get("best_model"):
        BEST_MODEL_PATH.write_text(json.dumps(bundle["best_model"], indent=2) + "\n")
    for analysis in bundle.get("analysis") or []:
        write_bundle_analysis(analysis)
    return write_summary_artifacts()


def predictions_and_features(model, dataset):
    """Return classifier predictions, confidence, and normalized avg-pool features."""
    import torch
    from src.config import DEVICE, EVAL_BATCH_SIZE, NUM_WORKERS
    from torch.utils.data import DataLoader

    activations = []

    def save_avg_pool(_, __, output):
        activations.append(torch.flatten(output, 1).cpu())

    handle = model.avgpool.register_forward_hook(save_avg_pool)
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    labels, predictions, confidence = [], [], []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            probabilities = model(images.to(DEVICE)).softmax(dim=1).cpu()
            predictions.append(probabilities.argmax(dim=1))
            confidence.append(probabilities.max(dim=1).values)
            labels.append(targets)
    handle.remove()
    features = torch.cat(activations).numpy()
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    return (
        torch.cat(labels).numpy(), torch.cat(predictions).numpy(),
        torch.cat(confidence).numpy(), features,
    )


def export_checkpoint_predictions(experiment, k, seed):
    """Evaluate one saved fine-tuning checkpoint and retain support neighbours."""
    from torch.utils.data import Subset

    from src.config import DEVICE
    from src.data import load_pets, load_splits
    from src.finetune import build_model, checkpoint_path, load_weights

    variants = {"E5": "layer4", "E6": "full"}
    if experiment not in variants:
        raise ValueError(f"Checkpoint analysis supports {sorted(variants)}, got {experiment}")
    path = checkpoint_path(variants[experiment], k, seed)
    if not path.exists():
        raise FileNotFoundError(path)

    model = load_weights(build_model(variants[experiment]), path, DEVICE)
    test_set = load_pets("test")
    trainval_set = load_pets("trainval")
    splits = load_splits()
    support_indices = splits["support"][k][seed]
    support_set = Subset(trainval_set, support_indices)
    true_labels, pred_labels, confidence, test_features = predictions_and_features(model, test_set)
    support_labels, _, _, support_features = predictions_and_features(model, support_set)
    assert np.array_equal(true_labels, np.asarray(test_set._labels))
    classes = test_set.classes

    rows = []
    for index, (true_label, pred_label, score, feature) in enumerate(
        zip(true_labels, pred_labels, confidence, test_features)
    ):
        candidates = np.flatnonzero(support_labels == pred_label)
        assert len(candidates) == k, (index, pred_label, len(candidates))
        similarities = support_features[candidates] @ feature
        neighbour = candidates[similarities.argmax()]
        source_index = support_indices[neighbour]
        rows.append({
            "experiment": experiment, "K": k, "seed": seed, "test_index": index,
            "true_label": true_label, "pred_label": pred_label,
            "true_breed": classes[true_label], "pred_breed": classes[pred_label],
            "confidence": score, "image_path": str(test_set._images[index]),
            "nearest_support_path": str(trainval_set._images[source_index]),
            "nearest_support_breed": classes[pred_label],
            "nearest_support_cosine": similarities.max(),
        })
    predictions_dir = RESULTS_DIR / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    output = predictions_dir / f"{experiment}_k{k}_seed{seed}.csv"
    pd.DataFrame(rows, columns=sorted(PREDICTION_COLUMNS)).to_csv(output, index=False)
    return output


def write_prediction_analysis(prediction_path):
    """Create K=20 per-breed, confusion, and three-case visual analysis."""
    from sklearn.metrics import confusion_matrix

    predictions = pd.read_csv(prediction_path)
    missing = PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"{prediction_path} misses columns: {sorted(missing)}")
    if predictions["K"].nunique() != 1 or predictions["K"].iloc[0] != 20:
        raise ValueError("Per-breed analysis is defined only for a single K=20 run")
    if len(predictions) == 0:
        raise ValueError("Prediction file is empty")

    experiment = predictions["experiment"].iloc[0]
    seed = int(predictions["seed"].iloc[0])
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    grouped = predictions.assign(correct=predictions.true_label == predictions.pred_label)
    per_breed = grouped.groupby(["true_label", "true_breed"], as_index=False).agg(
        test_images=("correct", "size"), accuracy=("correct", "mean"),
        errors=("correct", lambda values: int((~values).sum())),
    ).sort_values("accuracy")
    per_breed.to_csv(ANALYSIS_DIR / f"{experiment}_k20_seed{seed}_per_breed.csv", index=False)

    labels = sorted(set(predictions.true_label) | set(predictions.pred_label))
    names = [predictions.loc[predictions.true_label == label, "true_breed"].iloc[0] for label in labels]
    matrix = confusion_matrix(predictions.true_label, predictions.pred_label, labels=labels)
    np.savetxt(ANALYSIS_DIR / f"{experiment}_k20_seed{seed}_confusion.csv", matrix, fmt="%d", delimiter=",")
    write_confusion_plot(matrix, names, ANALYSIS_DIR / f"{experiment}_k20_seed{seed}_confusion.png")

    errors = predictions[predictions.true_label != predictions.pred_label].copy()
    errors["pair_count"] = errors.groupby(["true_breed", "pred_breed"])["test_index"].transform("size")
    failures = errors.sort_values(["pair_count", "confidence"], ascending=False).drop_duplicates(
        ["true_breed", "pred_breed"]
    ).head(3)
    if len(failures) < 3:
        raise ValueError("Expected at least three distinct misclassification pairs")
    failures.to_csv(ANALYSIS_DIR / f"{experiment}_k20_seed{seed}_failure_cases.csv", index=False)

    figure, axes = plt.subplots(3, 2, figsize=(10, 12))
    for row_number, (_, row) in enumerate(failures.iterrows()):
        axes[row_number, 0].imshow(plt.imread(row.image_path))
        axes[row_number, 0].set_title(f"Test: {row.true_breed} -> {row.pred_breed}\nconfidence {row.confidence:.2f}")
        axes[row_number, 1].imshow(plt.imread(row.nearest_support_path))
        axes[row_number, 1].set_title(f"Nearest support: {row.nearest_support_breed}\ncosine {row.nearest_support_cosine:.2f}")
        axes[row_number, 0].axis("off")
        axes[row_number, 1].axis("off")
    figure.tight_layout()
    figure.savefig(ANALYSIS_DIR / f"{experiment}_k20_seed{seed}_failure_cases.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs=3, metavar=("EXPERIMENT", "K", "SEED"))
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--report-results", type=Path,
        help="Temporary per-run bundle. Metrics are copied into experiment CSVs, then summaries are recomputed.",
    )
    arguments = parser.parse_args()
    if arguments.report_results:
        if arguments.checkpoint or arguments.predictions:
            raise ValueError("--report-results cannot be combined with prediction analysis")
        import_report_results(arguments.report_results)
        print(f"imported per-run metrics from {arguments.report_results} and rebuilt summaries")
        return
    write_summary_artifacts()
    if arguments.checkpoint:
        experiment, k, seed = arguments.checkpoint
        prediction_path = export_checkpoint_predictions(experiment, int(k), int(seed))
        write_prediction_analysis(prediction_path)
    elif arguments.predictions:
        write_prediction_analysis(arguments.predictions)


if __name__ == "__main__":
    main()
