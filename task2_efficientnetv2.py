"""Unified Task 2 pipeline: train/infer CNN and post-process quality grades.

This single script includes:
- Multitask transfer-learning CNN training (ResNet-50 backbone)
- Fresh/Rotten classification and Colour/Size/Ripeness regression
- Grade assignment (A/B/C) and inventory action simulation

The project remains fully computer-vision/deep-learning based.
"""

from __future__ import annotations

import copy
import os
import random
import ssl
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.error import URLError

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


@dataclass
class History:
    """Stores training and validation metrics per epoch."""

    train_total_loss: List[float]
    val_total_loss: List[float]
    train_cls_loss: List[float]
    val_cls_loss: List[float]
    train_quality_loss: List[float]
    val_quality_loss: List[float]
    train_acc: List[float]
    val_acc: List[float]


@dataclass(frozen=True)
class QualityScores:
    """Container for quality percentages in the range [0, 100]."""

    colour: float
    size: float
    ripeness: float


@dataclass
class RunConfig:
    """Editable training configuration for local runs/Colab notebooks."""

    dataset_dir: Path = Path("FruitAndVegetableDataset")
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    quality_loss_weight: float = 0.1
    quality_warmup_epochs: int = 3
    image_size: int = 224
    patience: int = 5
    early_stop_min_delta: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    no_pretrained: bool = False
    save_model_path: Path = Path("fresh_rotten_resnet50.pth")
    save_plot_path: Path = Path("learning_curves.png")
    predict_image: Path | None = None


CONFIG = RunConfig(
    dataset_dir=Path("FruitAndVegetableDataset"),
    epochs=20,
    batch_size=32,
    learning_rate=3e-4,
    weight_decay=1e-4,
    label_smoothing=0.05,
    quality_loss_weight=0.1,
    quality_warmup_epochs=3,
    image_size=224,
    patience=5,
    early_stop_min_delta=1e-4,
    num_workers=2,
    seed=42,
    no_pretrained=False,
    save_model_path=Path("fresh_rotten_resnet50.pth"),
    save_plot_path=Path("learning_curves.png"),
    predict_image=None,
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

class AddGaussianNoise:
    """Apply additive Gaussian noise to an image tensor."""

    def __init__(self, mean: float = 0.0, std: float = 0.03) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    """Clamp a numeric value to a closed interval."""
    return max(lower, min(value, upper))


def normalize_label(label: str) -> str:
    """Normalize and validate a predicted class label."""
    normalized = label.strip().lower()
    if normalized in {"fresh", "rotten"}:
        return normalized

    if "fresh" in normalized or "healthy" in normalized:
        return "fresh"
    if "rotten" in normalized or "disease" in normalized or "spoiled" in normalized:
        return "rotten"

    raise ValueError(
        "label must map to Fresh or Rotten "
        "(examples: Fresh, Rotten, Healthy, Apple__Rotten)."
    )


# ---------------------------------------------------------------------------
# Quality scoring and grading
# ---------------------------------------------------------------------------

def validate_quality_scores(
    quality_scores: Dict[str, float] | QualityScores,
) -> QualityScores:
    """Validate and normalize quality-score inputs from a deep model head."""
    if isinstance(quality_scores, QualityScores):
        return QualityScores(
            colour=round(clamp(float(quality_scores.colour)), 2),
            size=round(clamp(float(quality_scores.size)), 2),
            ripeness=round(clamp(float(quality_scores.ripeness)), 2),
        )

    required_keys = {"colour", "size", "ripeness"}
    missing = required_keys - set(quality_scores.keys())
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"quality_scores is missing required keys: {missing_keys}")

    return QualityScores(
        colour=round(clamp(float(quality_scores["colour"])), 2),
        size=round(clamp(float(quality_scores["size"])), 2),
        ripeness=round(clamp(float(quality_scores["ripeness"])), 2),
    )


def assign_overall_grade(scores: QualityScores) -> str:
    """Assign grade using strict thresholds: A>=85/90/80, C<65/70/60, else B."""
    if scores.colour < 65.0 or scores.size < 70.0 or scores.ripeness < 60.0:
        return "C"
    if scores.colour >= 75.0 and scores.size >= 80.0 and scores.ripeness >= 75.0:
        return "A"
    return "B"


def update_inventory_and_discount(grade: str) -> Dict[str, object]:
    """Simulate inventory update and discount recommendation by grade."""
    normalized_grade = grade.strip().upper()
    if normalized_grade not in {"A", "B", "C"}:
        raise ValueError("grade must be one of: 'A', 'B', 'C'.")

    if normalized_grade == "A":
        return {
            "status": "premium_stock",
            "discount_recommended": False,
            "discount_percent": 0,
            "action_note": "Keep at regular price and standard shelf placement.",
        }
    if normalized_grade == "B":
        return {
            "status": "lower_grade_stock",
            "discount_recommended": True,
            "discount_percent": 10,
            "action_note": "Apply mild discount and prioritize near-term sale.",
        }
    return {
        "status": "lower_grade_stock",
        "discount_recommended": True,
        "discount_percent": 25,
        "action_note": "Apply strong discount and move to clearance channel.",
    }


def process_prediction(
    label: str,
    confidence: float,
    quality_scores: Dict[str, float] | QualityScores,
) -> Dict[str, object]:
    """End-to-end post-processing for a single deep-model prediction."""
    normalized_label = normalize_label(label)
    scores = validate_quality_scores(quality_scores)
    grade = assign_overall_grade(scores)
    inventory_action = update_inventory_and_discount(grade)

    return {
        "input_label": label,
        "normalized_label": normalized_label,
        "defect_detected": normalized_label == "rotten",
        "input_confidence": round(clamp(confidence, 0.0, 1.0), 4),
        "quality_scores": {
            "colour": scores.colour,
            "size": scores.size,
            "ripeness": scores.ripeness,
        },
        "overall_grade": grade,
        "inventory_action": inventory_action,
    }


# ---------------------------------------------------------------------------
# Quality proxy targets (computed from image pixels for multitask training)
# ---------------------------------------------------------------------------

def _safe_mean(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.mean(values))


def _safe_std(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.std(values))


def compute_quality_proxy_targets(image: Image.Image) -> np.ndarray:
    """Create proxy quality targets from image pixels for multitask learning."""
    rgb = image.convert("RGB").resize((256, 256))
    hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255.0
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    mask = ((sat > 0.12) | (val < 0.92)) & (val > 0.08)
    if float(np.mean(mask)) < 0.03:
        mask = np.ones_like(mask, dtype=bool)

    sat_pixels, val_pixels = sat[mask], val[mask]
    mean_sat = _safe_mean(sat_pixels)
    mean_val = _safe_mean(val_pixels)
    val_std = _safe_std(val_pixels)

    brown_mask = mask & (hue >= 0.05) & (hue <= 0.14) & (sat >= 0.2) & (val <= 0.65)
    brown_ratio = float(np.mean(brown_mask[mask])) if np.any(mask) else 0.0

    area_ratio = float(np.mean(mask))
    ys, xs = np.where(mask)
    if ys.size > 0 and xs.size > 0:
        bbox_ratio = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) / float(mask.size)
        fill_ratio = area_ratio / max(bbox_ratio, 1e-6)
    else:
        fill_ratio = 0.0

    saturation_score = 100.0 * mean_sat
    brightness_score = 100.0 * clamp(1.0 - abs(mean_val - 0.65) / 0.65, 0.0, 1.0)
    colour = 0.6 * saturation_score + 0.4 * brightness_score - 35.0 * brown_ratio

    size = 0.8 * (100.0 * clamp((area_ratio - 0.08) / 0.52, 0.0, 1.0)) + \
           0.2 * (100.0 * clamp(fill_ratio, 0.0, 1.0))

    uniformity_score = 100.0 * clamp(1.0 - min(val_std / 0.25, 1.0), 0.0, 1.0)
    ripeness = (
        0.45 * clamp(colour, 0.0, 100.0)
        + 0.30 * uniformity_score
        + 0.25 * (100.0 - 100.0 * brown_ratio)
    )

    return np.array(
        [clamp(colour, 0.0, 100.0), clamp(size, 0.0, 100.0), clamp(ripeness, 0.0, 100.0)],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Dataset and data loading
# ---------------------------------------------------------------------------

def _safe_pil_loader(path: str) -> Image.Image:
    """Load an image robustly, handling palette transparency before RGB conversion."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
            category=UserWarning,
        )
        with Image.open(path) as image:
            if image.mode == "P" and "transparency" in image.info:
                image = image.convert("RGBA")
            return image.convert("RGB")


class QualityProxyImageFolder(datasets.ImageFolder):
    """ImageFolder variant that returns pre-cached proxy quality targets per sample."""

    def __init__(
        self,
        root: str,
        transform=None,
        target_transform=None,
        loader=_safe_pil_loader,
        validated_samples: List[Tuple[str, int]] | None = None,
        cached_quality_targets: List[torch.Tensor] | None = None,
    ) -> None:
        super().__init__(root=root, transform=transform,
                         target_transform=target_transform, loader=loader)

        if validated_samples is not None:
            self.samples = list(validated_samples)
        else:
            self.samples = self._filter_valid_samples(self.samples)

        if not self.samples:
            raise ValueError(f"No readable image files found in dataset directory: {root}")

        self.imgs = self.samples
        self.targets = [target for _, target in self.samples]

        # Pre-compute quality targets once to avoid repeated HSV analysis.
        if cached_quality_targets is not None:
            self._quality_cache = list(cached_quality_targets)
        else:
            print(f"Pre-computing quality proxy targets for {len(self.samples)} images...")
            self._quality_cache = []
            for i, (path, _) in enumerate(self.samples):
                try:
                    img = self.loader(path)
                    qt = torch.tensor(compute_quality_proxy_targets(img), dtype=torch.float32)
                except Exception:
                    qt = torch.zeros(3, dtype=torch.float32)
                self._quality_cache.append(qt)
                if (i + 1) % 2000 == 0:
                    print(f"  ...processed {i + 1}/{len(self.samples)} images")
            print("Quality target pre-computation complete.")

    @staticmethod
    def _is_readable_image(path: str) -> bool:
        try:
            _safe_pil_loader(path)
            return True
        except (UnidentifiedImageError, OSError, ValueError):
            return False

    @classmethod
    def _filter_valid_samples(cls, samples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        valid = [(p, t) for p, t in samples if cls._is_readable_image(p)]
        skipped = len(samples) - len(valid)
        if skipped > 0:
            print(f"Skipping {skipped} unreadable image file(s) in dataset.")
        return valid

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Unreadable image encountered after validation: {path}") from exc

        quality_target = self._quality_cache[index]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, quality_target


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _contains_supported_image(directory: Path, recursive: bool = True) -> bool:
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    return any(f.is_file() and f.suffix.lower() in image_suffixes for f in iterator)


def _count_direct_image_subdirs(directory: Path) -> int:
    return sum(1 for p in directory.iterdir()
               if p.is_dir() and _contains_supported_image(p, recursive=False))


def resolve_dataset_root(dataset_dir: Path) -> Path | None:
    """Resolve dataset root for torchvision ImageFolder, supporting wrappers."""
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return None

    candidates: list[tuple[int, int, Path]] = []
    first_level_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    search_roots = [dataset_dir] + first_level_dirs
    for child in first_level_dirs:
        search_roots.extend(p for p in child.iterdir() if p.is_dir())

    for root in search_roots:
        score = _count_direct_image_subdirs(root)
        if score >= 2:
            depth = len(root.relative_to(dataset_dir).parts) if root != dataset_dir else 0
            candidates.append((score, -depth, root))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][2]
    return None


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create train and validation preprocessing pipelines."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.03),  # applied on [0,1] range before Normalize
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.2)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def split_indices(
    targets: List[int], train_ratio: float, seed: int,
) -> Tuple[List[int], List[int]]:
    """Create deterministic stratified train/validation indices by class."""
    if not targets:
        raise ValueError("Cannot split an empty dataset.")

    class_to_indices: Dict[int, List[int]] = {}
    for idx, t in enumerate(targets):
        class_to_indices.setdefault(t, []).append(idx)

    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for class_indices in class_to_indices.values():
        rng.shuffle(class_indices)
        if len(class_indices) == 1:
            train_indices.extend(class_indices)
            continue
        train_size = min(max(int(len(class_indices) * train_ratio), 1), len(class_indices) - 1)
        train_indices.extend(class_indices[:train_size])
        val_indices.extend(class_indices[train_size:])

    if not val_indices:
        rng.shuffle(train_indices)
        fallback = max(1, int(len(train_indices) * (1.0 - train_ratio)))
        val_indices = train_indices[-fallback:]
        train_indices = train_indices[:-fallback]

    if not train_indices or not val_indices:
        raise ValueError("Failed to create non-empty train/validation splits.")

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def compute_class_weights(targets: List[int], num_classes: int) -> torch.Tensor:
    """Compute normalized inverse-frequency class weights from train targets."""
    if not targets:
        raise ValueError("Cannot compute class weights without training targets.")
    target_tensor = torch.tensor(targets, dtype=torch.long)
    class_counts = torch.clamp(torch.bincount(target_tensor, minlength=num_classes).float(), min=1.0)
    weights = class_counts.sum() / (num_classes * class_counts)
    weights = weights / weights.mean()
    return torch.clamp(weights, max=10.0)  # cap at 10x to prevent extreme loss spikes


def create_dataloaders(
    dataset_dir: Path, image_size: int, batch_size: int, num_workers: int, seed: int,
) -> Tuple[DataLoader, DataLoader, List[str], torch.Tensor]:
    """Build train/val loaders and class weights using a stratified 80/20 split."""
    resolved = resolve_dataset_root(dataset_dir)
    if resolved is None:
        raise FileNotFoundError(
            f"Dataset directory not found or invalid ImageFolder structure: {dataset_dir}"
        )

    train_tf, val_tf = build_transforms(image_size=image_size)
    base_dataset = QualityProxyImageFolder(root=str(resolved))
    if len(base_dataset) == 0:
        raise ValueError(f"No images found in dataset directory: {resolved}")

    class_names = base_dataset.classes
    train_idx, val_idx = split_indices(base_dataset.targets, train_ratio=0.8, seed=seed)
    class_weights = compute_class_weights(
        [base_dataset.targets[i] for i in train_idx], num_classes=len(class_names),
    )

    train_ds = QualityProxyImageFolder(root=str(resolved), transform=train_tf,
                                       validated_samples=base_dataset.samples,
                                       cached_quality_targets=base_dataset._quality_cache)
    val_ds = QualityProxyImageFolder(root=str(resolved), transform=val_tf,
                                     validated_samples=base_dataset.samples,
                                     cached_quality_targets=base_dataset._quality_cache)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, class_names, class_weights


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class MultiTaskProduceNet(nn.Module):
    """EfficientNetV2-S backbone with classification and quality regression heads."""

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feature_dim, num_classes),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 3),
            nn.Sigmoid(),
        )

    def train(self, mode: bool = True) -> "MultiTaskProduceNet":
        """Override train() to keep frozen BN layers in eval mode.

        EfficientNetV2 has batch norm throughout. When layers are frozen, their
        BN running stats should not be updated during training — otherwise eval
        mode uses stale ImageNet stats, causing massive train/val discrepancy.
        """
        super().train(mode)
        if mode:
            for module in self.backbone.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    # Only keep BN in eval for frozen layers.
                    if not any(p.requires_grad for p in module.parameters()):
                        module.eval()
        return self

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        logits = self.classifier_head(features)
        quality_scores = self.quality_head(features) * 100.0
        return logits, quality_scores


def build_model(
    num_classes: int, device: torch.device, use_pretrained: bool = True,
) -> nn.Module:
    """Create transfer-learning multitask model with an EfficientNetV2-S backbone."""
    pretrained_loaded = False

    if use_pretrained:
        try:
            backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            pretrained_loaded = True
        except (URLError, RuntimeError, OSError) as exc:
            retry_error = exc
            if "CERTIFICATE_VERIFY_FAILED" in str(exc).upper():
                try:
                    import certifi
                    cafile = certifi.where()
                    os.environ.setdefault("SSL_CERT_FILE", cafile)
                    os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)
                    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=cafile)
                    ctx = ssl.create_default_context(cafile=cafile)
                    urllib.request.install_opener(
                        urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
                    )
                    backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
                    pretrained_loaded = True
                    print("Loaded pretrained EfficientNetV2-S weights after applying certifi SSL certificates.")
                except (ImportError, URLError, RuntimeError, OSError) as cert_exc:
                    retry_error = cert_exc

            if not pretrained_loaded:
                print(f"Warning: Could not download pretrained weights ({retry_error}). "
                      "Falling back to randomly initialized weights.")
                backbone = efficientnet_v2_s(weights=None)
    else:
        backbone = efficientnet_v2_s(weights=None)

    if pretrained_loaded:
        # Freeze all layers; unfreeze the last 3 feature blocks for fine-tuning.
        for param in backbone.parameters():
            param.requires_grad = False
        for block in backbone.features[-3:]:
            for param in block.parameters():
                param.requires_grad = True

    # EfficientNetV2 uses backbone.classifier instead of backbone.fc.
    feature_dim = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    model = MultiTaskProduceNet(backbone=backbone, feature_dim=feature_dim, num_classes=num_classes)
    return model.to(device)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def _extract_checkpoint_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    """Normalize common checkpoint formats into current model key names."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary.")

    if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
        raw = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        raw = checkpoint["state_dict"]
    else:
        raw = checkpoint

    if not isinstance(raw, dict) or not raw:
        raise ValueError("Checkpoint does not contain a usable state dict.")

    normalised: Dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not torch.is_tensor(value):
            continue
        clean = key[len("module."):] if key.startswith("module.") else key
        if clean.startswith(("backbone.", "classifier_head.", "quality_head.")):
            normalised[clean] = value
        elif clean.startswith("fc."):
            suffix = clean[len("fc."):]
            for prefix in ("0.", "1."):
                if suffix.startswith(prefix):
                    suffix = suffix[len(prefix):]
                    break
            if suffix in {"weight", "bias"}:
                normalised[f"classifier_head.1.{suffix}"] = value
        else:
            normalised[f"backbone.{clean}"] = value
    return normalised


def _list_dataset_images(dataset_root: Path) -> List[Path]:
    """Return all supported image files under dataset_root."""
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return [f for f in dataset_root.rglob("*") if f.is_file() and f.suffix.lower() in image_suffixes]


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module, loader: DataLoader, cls_criterion: nn.Module,
    quality_criterion: nn.Module, quality_loss_weight: float, device: torch.device,
) -> Tuple[float, float, float, float]:
    """Run multitask evaluation on a data loader."""
    model.eval()
    total_loss_sum = cls_loss_sum = quality_loss_sum = 0.0
    correct = total = 0

    with torch.no_grad():
        for images, labels, quality_targets in loader:
            images, labels = images.to(device), labels.to(device)
            quality_targets = quality_targets.to(device)

            logits, quality_preds = model(images)
            cls_loss = cls_criterion(logits, labels)
            q_loss = quality_criterion(quality_preds, quality_targets)
            total_loss = cls_loss + quality_loss_weight * q_loss

            bs = labels.size(0)
            total_loss_sum += total_loss.item() * bs
            cls_loss_sum += cls_loss.item() * bs
            quality_loss_sum += q_loss.item() * bs
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total += bs

    n = max(total, 1)
    return total_loss_sum / n, 100.0 * correct / n, cls_loss_sum / n, quality_loss_sum / n


def train_model(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
    cls_criterion: nn.Module, quality_criterion: nn.Module,
    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    quality_loss_weight: float,
    quality_warmup_epochs: int, device: torch.device,
    num_epochs: int, patience: int, early_stop_min_delta: float,
) -> Tuple[nn.Module, History]:
    """Train multitask model with early stopping on validation cls loss."""
    history = History([], [], [], [], [], [], [], [])
    best_wts = copy.deepcopy(model.state_dict())
    best_val_cls_loss = float("inf")
    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        # Linear warmup: ramp quality weight from 0 to full over warmup period
        if quality_warmup_epochs > 0 and epoch < quality_warmup_epochs:
            eff_q_weight = quality_loss_weight * (epoch / quality_warmup_epochs)
        else:
            eff_q_weight = quality_loss_weight
        t_total = t_cls = t_q = 0.0
        correct = total = 0

        for images, labels, quality_targets in train_loader:
            images, labels = images.to(device), labels.to(device)
            quality_targets = quality_targets.to(device)

            optimizer.zero_grad()
            logits, q_preds = model(images)
            cls_loss = cls_criterion(logits, labels)
            q_loss = quality_criterion(q_preds, quality_targets)
            loss = cls_loss + eff_q_weight * q_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = labels.size(0)
            t_total += loss.item() * bs
            t_cls += cls_loss.item() * bs
            t_q += q_loss.item() * bs
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total += bs

        n = max(total, 1)
        tr_total, tr_cls, tr_q, tr_acc = t_total / n, t_cls / n, t_q / n, 100.0 * correct / n

        v_total, v_acc, v_cls, v_q = evaluate(
            model, val_loader, cls_criterion, quality_criterion, quality_loss_weight, device,
        )

        history.train_total_loss.append(tr_total)
        history.val_total_loss.append(v_total)
        history.train_cls_loss.append(tr_cls)
        history.val_cls_loss.append(v_cls)
        history.train_quality_loss.append(tr_q)
        history.val_quality_loss.append(v_q)
        history.train_acc.append(tr_acc)
        history.val_acc.append(v_acc)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Total: {tr_total:.4f} | Train Cls: {tr_cls:.4f} | "
            f"Train Q: {tr_q:.4f} | Train Acc: {tr_acc:.2f}% | "
            f"Val Total: {v_total:.4f} | Val Cls: {v_cls:.4f} | "
            f"Val Q: {v_q:.4f} | Val Acc: {v_acc:.2f}% | "
            f"QWeight: {eff_q_weight:.3f}"
        )

        improved = v_cls < (best_val_cls_loss - early_stop_min_delta)
        tie_break = abs(v_cls - best_val_cls_loss) <= early_stop_min_delta and v_acc > best_val_acc

        if improved or tie_break:
            best_val_cls_loss, best_val_acc = v_cls, v_acc
            best_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if scheduler is not None:
            scheduler.step()

        if no_improve >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_wts)
    return model, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_learning_curves(history: History, save_path: Path | None = None) -> None:
    """Plot multitask training curves."""
    epochs = range(1, len(history.train_total_loss) + 1)
    plt.figure(figsize=(14, 10))

    for idx, (train, val, title, ylabel) in enumerate([
        (history.train_total_loss, history.val_total_loss, "Total Loss", "Loss"),
        (history.train_cls_loss, history.val_cls_loss, "Classification Loss", "Loss"),
        (history.train_quality_loss, history.val_quality_loss, "Quality Regression Loss", "MSE"),
        (history.train_acc, history.val_acc, "Classification Accuracy", "Accuracy (%)"),
    ], start=1):
        plt.subplot(2, 2, idx)
        plt.plot(epochs, train, label=f"Train {title.split()[0]}")
        plt.plot(epochs, val, label=f"Val {title.split()[0]}")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Single-image prediction
# ---------------------------------------------------------------------------

def predict_single_image(
    model: nn.Module, image_path: Path, class_names: List[str],
    image_size: int, device: torch.device,
) -> Tuple[str, float, Dict[str, float]]:
    """Predict class label, confidence, and quality percentages for one image."""
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits, q_preds = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    label = class_names[pred_idx.item()]
    q_np = torch.clamp(q_preds.squeeze(0), 0.0, 100.0).cpu().numpy()
    return label, confidence.item() * 100.0, {
        "colour": round(float(q_np[0]), 2),
        "size": round(float(q_np[1]), 2),
        "ripeness": round(float(q_np[2]), 2),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run end-to-end multitask training, evaluation, plotting, and inference."""
    cfg = CONFIG
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_root = resolve_dataset_root(cfg.dataset_dir)
    if dataset_root is None:
        raise FileNotFoundError(
            f"Dataset not found at '{cfg.dataset_dir}'. "
            "Download and extract the dataset to this directory."
        )
    print(f"Using dataset directory: {dataset_root}")

    train_loader, val_loader, class_names, class_weights = create_dataloaders(
        dataset_dir=dataset_root, image_size=cfg.image_size,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers, seed=cfg.seed,
    )
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(
        num_classes=len(class_names), device=device,
        use_pretrained=not cfg.no_pretrained,
    )

    cls_criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=cfg.label_smoothing,
    )
    quality_criterion = nn.SmoothL1Loss()

    # Differential learning rates: backbone fine-tuned layers get lower LR.
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "backbone" in n]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "backbone" not in n]
    optimizer = Adam([
        {"params": backbone_params, "lr": cfg.learning_rate * 0.01},
        {"params": head_params, "lr": cfg.learning_rate},
    ], weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6,
    )

    trained_now = False
    history = History([], [], [], [], [], [], [], [])

    if cfg.save_model_path.exists():
        checkpoint = torch.load(cfg.save_model_path, map_location=device)
        state_dict = _extract_checkpoint_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded existing model from: {cfg.save_model_path}")
    else:
        print("No existing model checkpoint found. Training from scratch.")
        model, history = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            cls_criterion=cls_criterion, quality_criterion=quality_criterion,
            optimizer=optimizer, scheduler=scheduler, quality_loss_weight=cfg.quality_loss_weight,
            quality_warmup_epochs=cfg.quality_warmup_epochs, device=device,
            num_epochs=cfg.epochs, patience=cfg.patience,
            early_stop_min_delta=cfg.early_stop_min_delta,
        )
        trained_now = True

    if trained_now:
        tr_total, tr_acc, tr_cls, tr_q = evaluate(
            model, train_loader, cls_criterion, quality_criterion,
            cfg.quality_loss_weight, device,
        )
        v_total, v_acc, v_cls, v_q = evaluate(
            model, val_loader, cls_criterion, quality_criterion,
            cfg.quality_loss_weight, device,
        )
        print("\nFinal Evaluation Metrics")
        print(f"Train Total Loss: {tr_total:.4f} | Cls: {tr_cls:.4f} | "
              f"Quality: {tr_q:.4f} | Accuracy: {tr_acc:.2f}%")
        print(f"Val   Total Loss: {v_total:.4f} | Cls: {v_cls:.4f} | "
              f"Quality: {v_q:.4f} | Accuracy: {v_acc:.2f}%")

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": cfg.image_size,
            "architecture": "efficientnetv2s_multitask_transfer_learning",
        }, cfg.save_model_path)
        print(f"Saved model weights to: {cfg.save_model_path}")

        plot_learning_curves(history=history, save_path=cfg.save_plot_path)
        print(f"Saved learning curves to: {cfg.save_plot_path}")

    # Run prediction on a sample image.
    prediction_image = cfg.predict_image
    if prediction_image is None:
        dataset_images = _list_dataset_images(dataset_root)
        if not dataset_images:
            raise FileNotFoundError(f"No image files found in: {dataset_root}")
        prediction_image = random.SystemRandom().choice(dataset_images)
        print(f"Selected random dataset image for prediction: {prediction_image}")

    if not prediction_image.exists() or not prediction_image.is_file():
        raise FileNotFoundError(f"Prediction image not found: {prediction_image}")

    label, confidence_pct, quality_scores = predict_single_image(
        model, prediction_image, class_names, cfg.image_size, device,
    )
    result = process_prediction(
        label=label, confidence=confidence_pct / 100.0, quality_scores=quality_scores,
    )

    print(f"\nPrediction for '{prediction_image.name}': {label} "
          f"(confidence: {confidence_pct:.2f}%)")
    print(f"Quality Scores: {result['quality_scores']}")
    print(f"Overall Grade: {result['overall_grade']}")
    print(f"Inventory Action: {result['inventory_action']}")


if __name__ == "__main__":
    main()
