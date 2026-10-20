"""Cache frozen-backbone embeddings for trainval and test as .npy files.

Run: python -m src.features {resnet50,dinov2_vits14,dinov2_vitb14}
"""

import argparse
import csv
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import ResNet50_Weights, resnet50

from src.config import (
    BACKBONE,
    BACKBONES_RESULTS_PATH,
    DATA_ROOT,
    DEVICE,
    DINOV2_REPO,
    FEATURE_BATCH_SIZE,
    FEATURE_FILE,
    FEATURES_DIR,
    FROZEN_EMBEDDING_DIMS,
    IMAGE_SIZE,
    MEAN,
    NUM_WORKERS,
    PRETRAINED_WEIGHTS,
    RESIZE_SIZE,
    STD,
    TEST_SPLIT,
    TRAINVAL_SPLIT,
)

PREPROCESS = transforms.Compose(
    [
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)


def load_backbone(device):
    """ImageNet-pretrained ResNet-50 that outputs the 2048-d pooled embedding."""
    model = resnet50(weights=getattr(ResNet50_Weights, PRETRAINED_WEIGHTS))
    model.fc = nn.Identity()
    return model.eval().to(device)


def load_dinov2(name, device):
    """Frozen DINOv2 from torch.hub; its forward returns the normalized CLS token."""
    # skip_validation avoids a GitHub API call that fails when the API is rate limited
    model = torch.hub.load(DINOV2_REPO, name, skip_validation=True, trust_repo=True)
    return model.eval().to(device)


def load_frozen_backbone(backbone, device):
    """ResNet-50 or one of the DINOv2 models listed in FROZEN_EMBEDDING_DIMS."""
    assert backbone in FROZEN_EMBEDDING_DIMS, backbone
    if backbone == BACKBONE:
        return load_backbone(device)
    return load_dinov2(backbone, device)


@torch.no_grad()
def extract(model, dataset, device):
    """Return (features, labels) as numpy arrays for a whole dataset."""
    loader = DataLoader(
        dataset, batch_size=FEATURE_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    feats, labels = [], []
    for images, targets in loader:
        feats.append(model(images.to(device)).cpu())
        labels.append(targets)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def count_params(model):
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def gflops_per_image(model):
    """Forward-pass GFLOPs for one image (multiply and add counted separately)."""
    from torch.utils.flop_counter import FlopCounterMode

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device))
    model.train(was_training)
    return counter.get_total_flops() / 1e9


def save_backbone_stats(row):
    """Write one row per backbone to results/backbones.csv, replacing an older row."""
    rows = []
    if BACKBONES_RESULTS_PATH.exists():
        with BACKBONES_RESULTS_PATH.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["backbone"] != row["backbone"]]
    rows.append(row)
    BACKBONES_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BACKBONES_RESULTS_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)


def main(backbone):
    """Extract embeddings for trainval and test, save them and the backbone's cost."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    model = load_frozen_backbone(backbone, DEVICE)
    extraction_time = 0.0
    for split in (TRAINVAL_SPLIT, TEST_SPLIT):
        dataset = OxfordIIITPet(
            DATA_ROOT,
            split=split,
            target_types="category",
            transform=PREPROCESS,
            download=True,
        )
        start = time.time()
        x, y = extract(model, dataset, DEVICE)
        extraction_time += time.time() - start
        assert x.shape == (len(dataset), FROZEN_EMBEDDING_DIMS[backbone]), x.shape
        np.save(FEATURES_DIR / FEATURE_FILE.format(backbone=backbone, split=split, kind="x"), x)
        np.save(FEATURES_DIR / FEATURE_FILE.format(backbone=backbone, split=split, kind="y"), y)
        print(f"{split}: features {x.shape}, labels {y.shape}, saved to {FEATURES_DIR}")

    total_params, _ = count_params(model)
    save_backbone_stats({
        "backbone": backbone,
        "embedding_dim": FROZEN_EMBEDDING_DIMS[backbone],
        "total_params": total_params,
        "gflops_per_image": gflops_per_image(model),
        "extraction_time_s": extraction_time,
        "device": DEVICE.type,
    })
    print(f"{backbone}: {total_params / 1e6:.1f}M params, "
          f"extraction {extraction_time:.0f} s on {DEVICE.type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backbone", choices=list(FROZEN_EMBEDDING_DIMS))
    main(parser.parse_args().backbone)
