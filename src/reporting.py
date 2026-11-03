"""Generate report tables and the detailed results supplement from audited CSVs.

Run after ``python -m src.evaluate`` and ``python -m src.analyze``.
"""

import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluate import ANALYSIS_DIR, RESULTS_DIR, metric_frames, summarize_metrics

ROOT = RESULTS_DIR.parent
TABLES = RESULTS_DIR / "report_tables"
METHODS = {
    ("E1", "logreg"): "ResNet-50 + LR",
    ("E2", "logreg"): "DINOv2-S + LR",
    ("E3", "knn"): "DINOv2-S + k-NN",
    ("E3", "prototype"): "DINOv2-S + prototype",
    ("E4", "knn"): "ResNet-50 + k-NN",
    ("E4", "prototype"): "ResNet-50 + prototype",
    ("E5", "finetune"): "ResNet-50: head + layer4",
    ("E6", "finetune"): "ResNet-50: full fine-tuning",
    ("E7", "prototype"): "DINOv2-B + prototype",
    ("E7b", "logreg"): "DINOv2-B + LR",
}
FAILURE_REASONS = [
    "American Pit Bull Terrier to Staffordshire Bull Terrier (23 errors): "
    "the test and support share a brindle coat and broad head. The close facial "
    "view omits body proportions; indoor clutter may add irrelevant similarity.",
    "Staffordshire Bull Terrier to American Pit Bull Terrier (18 errors): "
    "both images show a light-coated, muscular dog on grass. Running pose and "
    "a ball covering the test dog's mouth obscure potentially useful shape cues.",
    "Ragdoll to Birman (14 errors): blue eyes, a dark facial mask and pale long "
    "fur look similar in the pair. The close view hides paw and body markings. "
    "These are plausible visual causes, not experimentally established causes.",
]
SHORT_FAILURE_REASONS = [
    "Pit Bull to Staffordshire (23 errors): shared brindle coat and broad head; "
    "the close facial view omits body proportions, while indoor clutter may distract.",
    "Staffordshire to Pit Bull (18): light coat and muscular build resemble the "
    "support dog; running pose and a ball obscure the test dog's mouth and shape cues.",
    "Ragdoll to Birman (14): both show blue eyes, a dark mask and pale long fur; "
    "the close view hides paw/body markings. These are visual hypotheses, not proven causes.",
]


def mean_std(mean, std, digits=1, latex=False):
    separator = r" $\pm$ " if latex else " ± "
    return f"{mean:.{digits}f}{separator}{std:.{digits}f}"


def quality_table(summary, metric, latex=False):
    rows = []
    for (experiment, classifier), group in summary.groupby(["experiment", "classifier"]):
        label = f"{experiment}: {METHODS[experiment, classifier]}"
        values = [mean_std(row[f"{metric}_mean"] * 100,
                           row[f"{metric}_std"] * 100, latex=latex)
                  for _, row in group.sort_values("K").iterrows()]
        if latex and experiment == "E7b":
            values = [rf"\textbf{{{value}}}" for value in values]
        rows.append([label, *values])
    return rows


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def latex_table(headers, rows, spec):
    return "\n".join([rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
                      " & ".join(headers) + r" \\", r"\midrule",
                      *[" & ".join(map(str, row)) + r" \\" for row in rows],
                      r"\bottomrule", r"\end{tabular}", ""])


def compare_breeds():
    comparison = None
    for experiment in ("E1", "E2", "E7b"):
        path = ANALYSIS_DIR / f"{experiment}_k20_seed0_per_breed.csv"
        frame = pd.read_csv(path)[["true_label", "true_breed", "accuracy"]]
        frame = frame.rename(columns={"accuracy": f"{experiment}_accuracy"})
        comparison = frame if comparison is None else comparison.merge(
            frame, on=["true_label", "true_breed"], validate="one_to_one"
        )
    comparison = comparison.sort_values("E7b_accuracy")
    comparison.to_csv(ANALYSIS_DIR / "frozen_per_breed_comparison.csv", index=False)
    return comparison


def failure_figure():
    failures = pd.read_csv(ANALYSIS_DIR / "E7b_k20_seed0_failure_cases.csv")
    assert len(failures) == 3
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 4.5))
    for column, row in enumerate(failures.itertuples()):
        axes[0, column].imshow(plt.imread(ROOT / row.image_path))
        title = f"{column + 1}. {row.true_breed} -> {row.pred_breed}"
        axes[0, column].set_title(textwrap.fill(title, 33), fontsize=9)
        axes[1, column].imshow(plt.imread(ROOT / row.nearest_support_path))
        axes[1, column].set_title(
            f"Predicted-class support\ncosine={row.nearest_support_cosine:.3f}", fontsize=9
        )
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    figure.tight_layout(pad=0.6)
    figure.savefig(ANALYSIS_DIR / "final_failure_examples.png", dpi=220)
    plt.close(figure)
    return failures


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    metrics = metric_frames()
    summary = summarize_metrics(metrics)
    supplement = ["# Final experimental evaluation", "",
                  "200 original runs; 10 methods; K=1,5,10,20; seeds 0--4. "
                  "All quality values are test percentages, mean ± sample standard deviation (ddof=1).",
                  "", "## Accuracy", ""]
    for metric, title in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")):
        rows = quality_table(summary, metric, latex=True)
        (TABLES / f"{metric}.tex").write_text(latex_table(
            ["Method", "$K=1$", "$K=5$", "$K=10$", "$K=20$"], rows, "@{}lrrrr@{}"
        ))
        if metric == "macro_f1":
            supplement.extend(["", "## Macro-F1", ""])
        supplement.append(markdown_table(
            ["Method", "K=1", "K=5", "K=10", "K=20"], quality_table(summary, metric)
        ))

    efficiency = pd.read_csv(RESULTS_DIR / "efficiency_k20.csv")
    runtime_rows = []
    for row in efficiency.itertuples():
        digits = 4 if row.training_time_s_mean < 0.01 else 3 if row.training_time_s_mean < 1 else 2
        runtime_rows.append([
            f"{row.experiment}: {row.classifier}",
            mean_std(row.training_time_s_mean, row.training_time_s_std, digits, True),
            f"{row.total_params / 1e6:.2f}", f"{row.trainable_params / 1000:.2f}",
            f"{row.gflops_per_image:.2f}", row.devices.upper(),
        ])
    (TABLES / "runtime.tex").write_text(latex_table(
        ["Method", "Fit time (s)", "Total (M)", "Fitted (k)", "GFLOPs", "Device"],
        runtime_rows, "@{}llrrrr@{}",
    ))
    supplement.extend(["", "## Runtime and computational cost (K=20)", "",
                       markdown_table(
                           ["Method", "Fit seconds (mean ± std)", "Total M", "Fitted k", "GFLOPs/image", "Device"],
                           [[value.replace(r" $\pm$ ", " ± ") for value in row] for row in runtime_rows]),
                       "", "Fit time includes validation selection; frozen methods exclude one-time feature "
                       "extraction and test prediction. Fine-tuning includes 30 epochs and validation. "
                       "Prototype entries count stored class means, not gradient-trained weights; "
                       "k-NN stores 740 support vectors (K=20), excluded from parameter counts. "
                       "GFLOPs cover the backbone forward pass at 224×224, counting multiply/add separately. "
                       "Different devices were used, so timings are not a controlled speed benchmark.",
                       "", "### Original feature extraction (7,349 images)", ""])
    backbones = pd.read_csv(RESULTS_DIR / "backbones.csv")
    supplement.append(markdown_table(
        ["Backbone", "Time (s)", "Device", "GFLOPs/image"],
        [[r.backbone, f"{r.extraction_time_s:.2f}", r.device, f"{r.gflops_per_image:.2f}"]
         for r in backbones.itertuples()],
    ))
    supplement.extend(["", "## Paired accuracy differences", "",
                       "Same support seed in each pair; percentage points, mean ± sample std.", ""])
    pairs = [("E2", "E1"), ("E7b", "E2"), ("E7b", "E1")]
    paired_rows = []
    for target, baseline in pairs:
        values = []
        for k in (1, 5, 10, 20):
            a = metrics[(metrics.experiment == target) & (metrics.K == k)].set_index("seed")
            b = metrics[(metrics.experiment == baseline) & (metrics.K == k)].set_index("seed")
            delta = (a.accuracy - b.accuracy) * 100
            values.append(mean_std(delta.mean(), delta.std()))
        paired_rows.append([f"{target} - {baseline}", *values])
    supplement.append(markdown_table(["Comparison", "K=1", "K=5", "K=10", "K=20"], paired_rows))

    breeds = compare_breeds()
    failures = failure_figure()
    assert list(zip(failures.true_breed, failures.pred_breed)) == [
        ("American Pit Bull Terrier", "Staffordshire Bull Terrier"),
        ("Staffordshire Bull Terrier", "American Pit Bull Terrier"),
        ("Ragdoll", "Birman"),
    ], "Review the visual explanations if the selected failure pairs change"
    discussion = "\n\n".join(
        rf"\textbf{{{index + 1}.}} {reason}"
        for index, reason in enumerate(SHORT_FAILURE_REASONS)
    )
    discussion += (
        "\n\nAll three errors have predicted probability above 99\\%. "
        "Staffordshire (68.5\\%) and Pit Bull (69.0\\%) remain hardest in this run; "
        "Staffordshire falls below E1's 79.8\\%, so gains are not uniform across breeds.\n"
    )
    (TABLES / "failure_discussion.tex").write_text(discussion)
    supplement.extend(["", "## Failure analysis", "",
                       "Frozen comparisons use seed 0 fixed in advance, K=20, and the C recorded "
                       "in the original CSV. Predictions and regenerated metrics are stored separately "
                       "from the 200 original runs. The three examples are the most frequent distinct "
                       "error pairs in E7b, with the highest-probability error in each pair. "
                       "Support neighbours are restricted to the predicted breed; cosine similarity "
                       "is not a calibrated probability.", "",
                       "![Three failure examples](../results/analysis/final_failure_examples.png)", "",
                       markdown_table(["True breed", "Predicted", "Pair count", "Probability", "Cosine"],
                                      [[r.true_breed, r.pred_breed, r.pair_count,
                                        f"{r.confidence:.3f}", f"{r.nearest_support_cosine:.3f}"]
                                       for r in failures.itertuples()]), "",
                       *[f"{i + 1}. {reason}" for i, reason in enumerate(FAILURE_REASONS)], "",
                       "### Ten hardest breeds for E7b (single seed, not a five-seed average)", "",
                       markdown_table(["Breed", "E1 %", "E2 %", "E7b %"],
                                      [[r.true_breed, f"{100*r.E1_accuracy:.1f}",
                                        f"{100*r.E2_accuracy:.1f}", f"{100*r.E7b_accuracy:.1f}"]
                                       for r in breeds.head(10).itertuples()]), "",
                       "### Prediction verification", ""])
    for experiment, seed in (("E1", 0), ("E2", 0), ("E7b", 0), ("E5", 4)):
        path = ANALYSIS_DIR / f"{experiment}_k20_seed{seed}_provenance.json"
        audit = json.loads(path.read_text())
        supplement.append(f"- {experiment}, seed {seed}: regenerated accuracy "
                          f"{100*audit['prediction_accuracy']:.3f}%, original "
                          f"{100*audit['recorded_accuracy']:.3f}%; delta "
                          f"{audit['accuracy_difference_pp']:+.3f} pp.")
    supplement.extend(["", "## Reproducibility", "",
                       "See README.md for setup, original experiment commands, prediction export and "
                       "report generation. Report source: reports/evaluation_report.tex. "
                       "Pretraining overlap was not audited; architecture and pretraining data differ "
                       "between ResNet and DINOv2. Validation uses 20 additional labeled images per "
                       "breed, so the label budget exceeds K. Five-seed variability describes support "
                       "sampling, not uncertainty over datasets.", ""])
    (ROOT / "reports" / "final_results.md").write_text("\n".join(supplement))
    print("Generated report tables, detailed supplement and failure figure.")


if __name__ == "__main__":
    main()
