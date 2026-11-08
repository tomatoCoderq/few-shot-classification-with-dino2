"""Local pet-breed demo: python -m src.demo. Never downloads model weights."""
import argparse
import csv
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from torchvision.models import resnet50

from src.config import (BEST_CHECKPOINT_PATH, DEVICE, FEATURES_DIR,
                        LINEAR_PROBE_MAX_ITER, NUM_CLASSES, PROJECT_ROOT,
                        RESULTS_DIR, SPLITS_PATH)
from src.data import load_splits
from src.feature_cache import load_features
from src.features import PREPROCESS

# Fixed representative runs, not seed selection by test accuracy.
EXPERIMENTS = {
    'E1 · ResNet-50 + LR': ('E1', 'resnet50', 'logreg'),
    'E2 · DINOv2-S + LR': ('E2', 'dinov2_vits14', 'logreg'),
    'E3 · DINOv2-S + prototype': ('E3', 'dinov2_vits14', 'prototype'),
    'E3 · DINOv2-S + k-NN': ('E3', 'dinov2_vits14', 'knn'),
    'E4 · ResNet-50 + prototype': ('E4', 'resnet50', 'prototype'),
    'E4 · ResNet-50 + k-NN': ('E4', 'resnet50', 'knn'),
    'E5 · ResNet-50 · layer4': ('E5', 'resnet50', 'finetune'),
    'E6 · ResNet-50 · full': ('E6', 'resnet50', 'finetune'),
    'E7 · DINOv2-B + prototype': ('E7', 'dinov2_vitb14', 'prototype'),
    'E7b · DINOv2-B + LR': ('E7b', 'dinov2_vitb14', 'logreg'),
}
RESNET = 'E5 · ResNet-50 · layer4'
DINO = 'E7b · DINOv2-B + LR'
BEST = 'Best model · E7b · DINOv2-B + LR'
MODELS = [BEST, *EXPERIMENTS]
LABELS = Path(__file__).with_name('breed_names.json')


def spec(name):
    return EXPERIMENTS[DINO if name == BEST else name]


def result_path(experiment):
    return RESULTS_DIR / ('linear_probe.csv' if experiment == 'E1'
                          else f'frozen_{experiment.lower()}.csv')


def checkpoint(experiment):
    variant = {'E5': 'layer4', 'E6': 'full'}[experiment]
    return BEST_CHECKPOINT_PATH.parent / f'{variant}_k20_seed0.pt'


def backbone_files(backbone):
    hub = Path(torch.hub.get_dir())
    if backbone == 'resnet50':
        return [hub / 'checkpoints' / 'resnet50-11ad3fa6.pth']
    return [hub / 'facebookresearch_dinov2_main' / 'hubconf.py',
            hub / 'checkpoints' / f'{backbone}_pretrain.pth']


def required_files(name):
    experiment, backbone, method = spec(name)
    if method == 'finetune':
        return [checkpoint(experiment), LABELS]
    return [*backbone_files(backbone), LABELS,
            FEATURES_DIR / f'{backbone}_trainval_x.npy',
            FEATURES_DIR / f'{backbone}_trainval_y.npy',
            SPLITS_PATH, result_path(experiment)]


def missing_files(name):
    return [p for p in required_files(name) if not p.is_file()]


def status():
    lines = ['### Model files']
    for name in MODELS:
        missing = missing_files(name)
        lines.append(f'**{name}**: ' + ('Missing required files:' if missing else
                     'Files found. The app loads and checks them on the first prediction.'))
        lines.extend(f'- `{p}`' for p in missing)
    lines.append('All models use K=20, seed=0. The app loads weights from local files. '
                 'Build frozen feature caches with `python -m src.features <backbone> --cache-only`. '
                 'E5 and E6 use `checkpoints/layer4_k20_seed0.pt` and '
                 '`checkpoints/full_k20_seed0.pt`, respectively.')
    return '\n\n'.join(lines)


@lru_cache(maxsize=3)
def load_backbone(backbone):
    files = backbone_files(backbone)
    if backbone == 'resnet50':
        model = resnet50(weights=None)
        model.load_state_dict(torch.load(files[0], map_location='cpu', weights_only=True))
        model.fc = torch.nn.Identity()
    else:
        model = torch.hub.load(str(files[0].parent), backbone, source='local', pretrained=False)
        model.load_state_dict(torch.load(files[1], map_location='cpu', weights_only=True))
    return model.to(DEVICE).eval()


class Prototype:
    def __init__(self, x, y):
        self.centers = np.stack([x[y == c].mean(axis=0) for c in range(NUM_CLASSES)])
        self.centers /= np.linalg.norm(self.centers, axis=1, keepdims=True)

    def predict_proba(self, x):
        # Softmax only for display: preserves cosine ranking, is not calibrated.
        scores = x @ self.centers.T
        scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return scores / scores.sum(axis=1, keepdims=True)


@lru_cache(maxsize=10)
def load_model(name):
    if name == BEST:
        return load_model(DINO)
    missing = missing_files(name)
    if missing:
        raise FileNotFoundError('Missing required files: ' + ', '.join(map(str, missing)))
    experiment, backbone, method = spec(name)
    if method == 'finetune':
        model = resnet50(weights=None, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(checkpoint(experiment), map_location='cpu', weights_only=True))
        return model.to(DEVICE).eval(), None
    model = load_backbone(backbone)
    with result_path(experiment).open() as f:
        rows = [r for r in csv.DictReader(f) if r['K'] == '20' and r['seed'] == '0'
                and r['classifier'] == method and r['experiment'] == experiment]
    if len(rows) != 1:
        raise ValueError(f'Expected one result for {experiment}/{method} at K=20, seed=0.')
    x, y = load_features(backbone, 'trainval')
    support = load_splits()['support'][20][0]
    x, y = x[support], y[support]
    if not np.array_equal(np.unique(y), np.arange(NUM_CLASSES)):
        raise ValueError('The cached labels do not match the 37 breeds.')
    if method == 'prototype':
        classifier = Prototype(x, y)
    elif method == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=int(rows[0]['hparam'].split('=')[1]),
                                           metric='cosine', weights='distance').fit(x, y)
    else:
        c = float(rows[0]['C']) if experiment == 'E1' else float(rows[0]['hparam'].split('=')[1])
        classifier = LogisticRegression(C=c, max_iter=LINEAR_PROBE_MAX_ITER).fit(x, y)
    return model, classifier


def predict(image, name):
    if image is None:
        return {}, 'Upload an image to get a prediction.'
    try:
        import json
        names = json.loads(LABELS.read_text())
        model, classifier = load_model(name)
        tensor = PREPROCESS(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            output = model(tensor)
        if classifier is None:
            scores = output.softmax(dim=1).cpu().numpy()[0]
        else:
            embedding = output.cpu().numpy()
            embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
            scores = classifier.predict_proba(embedding)[0]
        return dict(zip(names, map(float, scores))), f'{name} · K=20, seed=0 · Device: {DEVICE}' + (
            ' · Prototype scores are cosine similarities rescaled with softmax for display.'
            if isinstance(classifier, Prototype) else '')
    except Exception as exc:
        logging.exception('Demo inference failed')
        return {}, f'Prediction failed: {exc}'


def examples():
    path = RESULTS_DIR / 'predictions' / 'E7b_k20_seed0.csv'
    if not path.exists():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    selected = []
    for failure in (False, True):
        seen = set()
        for row in rows:
            image = PROJECT_ROOT / row['image_path']
            if ((row['true_label'] != row['pred_label']) == failure
                    and row['true_breed'] not in seen and image.is_file()):
                selected.append([str(image)])
                seen.add(row['true_breed'])
                if len(seen) == 3:
                    break
    return selected


def build_app():
    import gradio as gr
    with gr.Blocks(title='Pet breed classification', analytics_enabled=False) as app:
        gr.Markdown('# What breed is this?\nThe model predicts one of 37 Oxford-IIIT Pet breeds '
                    'for every image, including photos of breeds and objects outside the dataset.')
        gr.Markdown('All methods use K=20, seed=0. E7b had the highest mean test accuracy '
                    'at K=20 (94.20% across five seeds). Hyperparameters were selected on validation data. '
                    'Prototype scores are cosine similarities rescaled with softmax for display.')
        with gr.Accordion('Weights and feature caches', open=False):
            availability = gr.Markdown(status())
        gr.Button('Check model files').click(status, outputs=availability)
        selector = gr.Dropdown(MODELS, value=BEST, label='Model')
        with gr.Row():
            image = gr.Image(type='pil', label='Pet photo')
            output = gr.Label(num_top_classes=5, label='Top 5 breeds')
        message = gr.Textbox(label='Status', interactive=False)
        gr.Button('Predict breed', variant='primary').click(
            predict, [image, selector], [output, message], concurrency_limit=1)
        samples = examples()
        if samples:
            gr.Markdown('### Examples\nThe first three images show correct E7b predictions. '
                        'The next three show errors from the recorded run. Other models may predict different breeds.')
            gr.Examples(samples, inputs=image)
        else:
            gr.Markdown('Example images are unavailable because the dataset images are missing locally.')
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    build_app().queue().launch(server_name='127.0.0.1', server_port=args.port, share=False)
