import copy
import csv
import json
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import ResNet50_Weights, resnet50

from src.config import (
    BACKBONE,
    BACKBONE_LR,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CHECKPOINT_FILE,
    CLASSIFIER,
    CUBLAS_WORKSPACE_CONFIG,
    DATA_ROOT,
    DEVICE,
    EPOCHS,
    EVAL_BATCH_SIZE,
    FINETUNE_LOG_PATH,
    FINETUNE_RESULTS_PATH,
    FINETUNE_SEED,
    HEAD_LR,
    HEAD_PREFIX,
    IMAGE_SIZE,
    LOGS_DIR,
    MEAN,
    NUM_CLASSES,
    NUM_WORKERS,
    PRETRAINED_WEIGHTS,
    RESIZE_SIZE,
    SEEDS,
    SHOTS,
    SPLITS_PATH,
    STD,
    TEST_SPLIT,
    TRAINVAL_SPLIT,
    VARIANT_TRAINABLE,
    VARIANTS,
    WEIGHT_DECAY,
)

# Must be set before the first CUDA matmul so cuBLAS is deterministic.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)

RESULT_FIELDS = [
    "experiment",
    "backbone",
    "classifier",
    "K",
    "seed",
    "train_seed",
    "best_epoch",
    "best_val_accuracy",
    "accuracy",
    "macro_f1",
    "training_time_s",
    "device",
    "trainable_params",
    "total_params",
]


def set_seed(seed):
    """Seed every RNG the training uses and force deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: a few CUDA ops have no deterministic version; they print a warning.
    torch.use_deterministic_algorithms(True, warn_only=True)


def log(message):
    """Print to the console and append to the finetune log file."""
    print(message, flush=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with FINETUNE_LOG_PATH.open("a") as f:
        f.write(message + "\n")


def checkpoint_path(variant, k, seed):
    return CHECKPOINT_DIR / CHECKPOINT_FILE.format(variant=variant, k=k, seed=seed)


def load_splits():
    """Read the saved splits; JSON turns K and seed keys into strings, convert back."""
    raw = json.loads(SPLITS_PATH.read_text())
    raw["support"] = {
        int(k): {int(seed): indices for seed, indices in by_seed.items()}
        for k, by_seed in raw["support"].items()
    }
    return raw


def load_trainval(transform):
    return OxfordIIITPet(
        DATA_ROOT,
        split=TRAINVAL_SPLIT,
        target_types="category",
        transform=transform,
        download=True,
    )


def load_test():
    return OxfordIIITPet(
        DATA_ROOT,
        split=TEST_SPLIT,
        target_types="category",
        transform=EVAL_TRANSFORM,
        download=True,
    )


def build_model(variant):
    """ResNet-50 with a new 37-class head; variant decides which parameters train."""
    model = resnet50(weights=ResNet50_Weights[PRETRAINED_WEIGHTS])
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    prefixes = VARIANT_TRAINABLE[variant]
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(prefixes)
    return model


def load_weights(model, path, device):
    """Load a saved state dict (from train_run) into a model built by build_model."""
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def count_params(model):
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_optimizer(model):
    """AdamW with a lower learning rate for the pretrained backbone than the head."""
    head = list(model.fc.parameters())
    backbone = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith(HEAD_PREFIX)
    ]
    return torch.optim.AdamW(
        [{"params": head, "lr": HEAD_LR}, {"params": backbone, "lr": BACKBONE_LR}],
        weight_decay=WEIGHT_DECAY,
    )


def freeze_batchnorm(model):
    """Keep BatchNorm in eval mode; call after model.train()."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


@torch.no_grad()
def predict(model, dataset, device):
    """Return (predictions, labels) as numpy arrays."""
    model.eval()
    loader = DataLoader(
        dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(dim=1).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def macro_f1(y_true, y_pred):
    scores = []
    for c in range(NUM_CLASSES):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


def train_run(variant, k, seed, device):
    """Fine-tune one model on the K-shot support set and test it once.

    The epoch is chosen as the one with the best validation accuracy.
    Returns a dict with the metrics and counts.
    """
    train_seed = FINETUNE_SEED + seed
    set_seed(train_seed)

    splits = load_splits()
    train_set = Subset(load_trainval(TRAIN_TRANSFORM), splits["support"][k][seed])
    val_set = Subset(load_trainval(EVAL_TRANSFORM), splits["val"])
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=torch.Generator().manual_seed(train_seed),
    )

    model = build_model(variant).to(device)
    total_params, trainable_params = count_params(model)
    optimizer = make_optimizer(model)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc, best_epoch = -1.0, 0
    best_state = copy.deepcopy(model.state_dict())
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        freeze_batchnorm(model)
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)

        val_preds, val_labels = predict(model, val_set, device)
        val_acc = float(np.mean(val_preds == val_labels))
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        log(
            f"[{variant} K={k} seed={seed}] epoch {epoch:2d}/{EPOCHS} "
            f"train_loss {loss_sum / total:.4f} train_acc {correct / total:.3f} "
            f"val_acc {val_acc:.3f}{' (best)' if improved else ''}"
        )
    training_time = time.time() - start

    model.load_state_dict(best_state)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_path(variant, k, seed)
    torch.save(best_state, checkpoint)
    log(
        f"[{variant} K={k} seed={seed}] best epoch {best_epoch}, "
        f"best val_acc {best_val_acc:.3f}, saved {checkpoint.name}"
    )

    preds, labels = predict(model, load_test(), device)
    return {
        "train_seed": train_seed,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "accuracy": float(np.mean(preds == labels)),
        "macro_f1": macro_f1(labels, preds),
        "training_time_s": training_time,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }


def completed_runs():
    """(experiment, K, seed) triples already written to the results CSV."""
    if not FINETUNE_RESULTS_PATH.exists():
        return set()
    with FINETUNE_RESULTS_PATH.open(newline="") as f:
        return {
            (row["experiment"], int(row["K"]), int(row["seed"]))
            for row in csv.DictReader(f)
        }


def append_result(row):
    FINETUNE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not FINETUNE_RESULTS_PATH.exists()
    with FINETUNE_RESULTS_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_finetune(variant, k, seed, device):
    """Run one fine-tuning experiment and append its row to the results CSV.

    Skips the run if its (experiment, K, seed) is already in the CSV.
    """
    experiment = VARIANTS[variant]
    if (experiment, k, seed) in completed_runs():
        log(f"[{variant} K={k} seed={seed}] already in results, skipping")
        return
    metrics = train_run(variant, k, seed, device)
    append_result(
        {
            "experiment": experiment,
            "backbone": BACKBONE,
            "classifier": CLASSIFIER,
            "K": k,
            "seed": seed,
            "device": str(device),
            **metrics,
        }
    )


if __name__ == "__main__":
    # All E5 and E6 runs: 2 variants x 4 values of K x 5 seeds = 40 runs.
    for variant in VARIANTS:
        for k in SHOTS:
            for seed in SEEDS:
                run_finetune(variant, k, seed, DEVICE)
