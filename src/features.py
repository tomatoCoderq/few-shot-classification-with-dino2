import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import ResNet50_Weights, resnet50

from src.config import (
    BACKBONE,
    DATA_ROOT,
    DEVICE,
    EMBEDDING_DIM,
    FEATURE_BATCH_SIZE,
    FEATURE_FILE,
    FEATURES_DIR,
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


if __name__ == "__main__":
    # Extract backbone embeddings for trainval and test and save them as .npy.
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    model = load_backbone(DEVICE)
    for split in (TRAINVAL_SPLIT, TEST_SPLIT):
        dataset = OxfordIIITPet(
            DATA_ROOT,
            split=split,
            target_types="category",
            transform=PREPROCESS,
            download=True,
        )
        x, y = extract(model, dataset, DEVICE)
        assert x.shape[1] == EMBEDDING_DIM, x.shape
        np.save(FEATURES_DIR / FEATURE_FILE.format(backbone=BACKBONE, split=split, kind="x"), x)
        np.save(FEATURES_DIR / FEATURE_FILE.format(backbone=BACKBONE, split=split, kind="y"), y)
        print(f"{split}: features {x.shape}, labels {y.shape}, saved to {FEATURES_DIR}")
