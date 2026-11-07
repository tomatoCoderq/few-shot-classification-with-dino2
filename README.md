# Few-shot pet breed classification

Oxford-IIIT Pet: 37 breeds, K = 1, 5, 10, 20 and five seeds.
Frozen DINOv2/ResNet features with LR, prototypes or k-NN; ResNet partial/full
fine-tuning. 200 recorded runs. Best mean K=20 accuracy: DINOv2-B + LR, 94.20 ± 0.27%.

## Setup and demo

Python 3.12 and uv:

```bash
uv sync --locked --extra demo
uv run --extra demo python -m src.demo
```

Open http://127.0.0.1:7860. Choose a method and upload an image.
All demo methods use K=20, seed=0. “Best model” is a shortcut to E7b.
Missing files are listed in the UI; the demo does not download weights.
E5/E6 use `checkpoints/layer4_k20_seed0.pt` and `full_k20_seed0.pt`.
Frozen methods need local backbone weights, trainval feature caches and source CSVs.
Prototype scores are relative similarities, not calibrated probabilities.

## Reproduce

```bash
uv run python -m src.data
uv run python -m src.features resnet50 --cache-only
uv run python -m src.features dinov2_vits14 --cache-only
uv run python -m src.features dinov2_vitb14 --cache-only
uv run python -m src.linear_probe
uv run python -m src.classifiers E2
```

Repeat the last command for E3, E4, E7 and E7b. These classifier commands
replace their result CSVs; preserve recorded results before rerunning.
Feature extraction downloads pretrained weights on first use. Committed splits
are validated, not regenerated. Validation selects hyperparameters; test is held out.

`uv run python -m src.finetune` runs all E5/E6 configurations, skipping rows
already in the results CSV. To independently retrain demo weights without changing
original results, use `uv run python scripts/train_new_runs.py --device mps`
(or `cuda`/`cpu`). See `new_runs/README.md`. Training is currently stopped.

## Export all experiment weights on Kaggle

Import [the Kaggle notebook](notebooks/kaggle_all_weights.ipynb), enable GPU and
Internet, and choose **Save & Run All**. The notebook bundles the project code
and fixed splits. It exports 160 frozen classifier states, 40 ResNet checkpoints,
shared backbone weights, and metric comparisons under `/kaggle/working/new_runs`.
To continue from a previous saved Kaggle output, attach it and set `RESUME_FROM`.

## Reports

```bash
uv run python -m src.evaluate
uv run python -m src.reporting
mkdir -p output/pdf
(cd reports && tectonic --outdir ../output/pdf evaluation_report.tex)
```

Tectonic is a separate LaTeX dependency. Reporting requires local dataset images
for the failure panel. Recorded prediction analyses are included; regenerate with
`uv run python -m src.analyze --frozen E1 E2 E7b --best-checkpoint`
after restoring caches and `checkpoints/best.pt`.

- `src/`: data, training, evaluation and demo.
- `splits/`: fixed support and validation indices.
- `results/`: source metrics, summaries, figures and prediction provenance.
- `reports/evaluation_report.tex`: final two-page report source.
- `output/pdf/evaluation_report.pdf`: compiled report.
- `reports/presentation.md`: slide-by-slide plan for five minutes.
- `plan/`, `reports/first_working_model.tex`: earlier submissions.
- `new_runs/`: independent retraining artifacts, excluded from final results.

## Team contributions

- **Khasan Sadykov** set up the project and fixed dataset splits, implemented the ResNet-50 linear probe (E1), and consolidated the final report and cross-model error analysis.
- **Marat Mestnikov** implemented ResNet-50 feature extraction, partial and full fine-tuning (E5/E6), and checkpoint selection by validation accuracy.
- **Evgeni Moskal** added frozen DINOv2-S/B feature extraction and ran the logistic regression, prototype, and k-NN experiments (E2, E3, E4, E7, E7b).
- **Martin Klimov** built the evaluation pipeline, generated metric tables and plots, and analyzed per-breed results and failures for the fine-tuned ResNet-50.
