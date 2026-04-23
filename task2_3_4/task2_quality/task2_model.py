"""Shared model, utilities, robust data loading, and grading for Task 2.

This module contains shared components used by both training and prediction:
- EfficientNetV2-S gatekeeper model for Fresh vs Rotten classification
- Robust ImageFolder loading with unreadable-image filtering
- Data splitting and dataloader creation
- Post-processing for quality scores and inventory actions
"""

from __future__ import annotations

import os
import random
import ssl
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.error import URLError

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


@dataclass
class History:
    """Stores training and validation metrics per epoch."""

    train_total_loss: List[float]
    val_total_loss: List[float]
    train_cls_loss: List[float]
    val_cls_loss: List[float]
    train_acc: List[float]
    val_acc: List[float]


@dataclass
class RunConfig:
    """Editable training configuration for local runs and notebooks."""

    dataset_dir: Path = Path("../task2_data")
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    image_size: int = 224
    patience: int = 5
    early_stop_min_delta: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    no_pretrained: bool = False
    save_model_path: Path = Path("fresh_rotten_efficientnetv2.pth")
    save_plot_path: Path = Path("learning_curves.png")
    predict_image: Path | None = None


CONFIG = RunConfig(
    dataset_dir=Path("../task2_data"),
    epochs=20,
    batch_size=32,
    learning_rate=3e-4,
    weight_decay=1e-4,
    label_smoothing=0.05,
    image_size=224,
    patience=5,
    early_stop_min_delta=1e-4,
    num_workers=2,
    seed=42,
    no_pretrained=False,
    save_model_path=Path("fresh_rotten_efficientnetv2.pth"),
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

def _coerce_quality_component(component: Any, metric_name: str) -> Dict[str, Any]:
    """Normalize one quality component into score/justification form."""
    if isinstance(component, dict):
        if "score" not in component:
            raise ValueError(f"'{metric_name}' must contain a 'score' field.")
        score = round(clamp(float(component["score"])), 2)
        justification_raw = component.get("justification")
        if justification_raw is None:
            justification = None
        elif isinstance(justification_raw, str):
            cleaned = justification_raw.strip()
            justification = cleaned if cleaned else None
        else:
            justification = str(justification_raw)
        return {"score": score, "justification": justification}

    if isinstance(component, (int, float, np.integer, np.floating)):
        score = round(clamp(float(component)), 2)
        return {"score": score, "justification": None}

    raise ValueError(
        f"'{metric_name}' must be a number or a dictionary with keys "
        "'score' and optional 'justification'."
    )


def validate_quality_scores(quality_scores: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Validate and normalize incoming quality scores.

    Accepts either:
    - {'colour': 88.0, 'shape': 91.5, 'ripeness': 86.0}
    - {'colour': {'score': 88.0, 'justification': '...'}, ...}

    For backward compatibility, 'size' is accepted and mapped to 'shape'.
    """
    if not isinstance(quality_scores, dict):
        raise ValueError("quality_scores must be a dictionary.")

    payload = dict(quality_scores)
    if "shape" not in payload and "size" in payload:
        payload["shape"] = payload["size"]

    required_keys = {"colour", "shape", "ripeness"}
    missing = required_keys - set(payload.keys())
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"quality_scores is missing required keys: {missing_keys}")

    return {
        "colour": _coerce_quality_component(payload["colour"], "colour"),
        "shape": _coerce_quality_component(payload["shape"], "shape"),
        "ripeness": _coerce_quality_component(payload["ripeness"], "ripeness"),
    }

#  Original grading logic using hard thresholds, which is simpler but less forgiving to borderline cases. Retained for reference.
# def assign_overall_grade(scores: Dict[str, Dict[str, Any]], is_rotten: bool = False) -> str:
#     """Assign grade using calibrated zero-shot thresholds for organic produce."""
#     if is_rotten:
#         return "C"

#     colour_score = float(scores["colour"]["score"])
#     shape_score = float(scores["shape"]["score"])
#     ripeness_score = float(scores["ripeness"]["score"])

#     # Calibrated thresholds to account for Softmax Dilution in 3-tier prompting
#     if colour_score < 50.0 or shape_score < 55.0 or ripeness_score < 50.0:
#         return "C"
#     if colour_score >= 70.0 and shape_score >= 70.0 and ripeness_score >= 65.0:
#         return "A"
#     return "B"

# Weighted average grading, allows a very bad score in one metric to be compensated by strong scores in others, which is more realistic for produce grading in practice.
def assign_overall_grade(scores: Dict[str, Dict[str, Any]], is_rotten: bool = False) -> str:
    """Assign grade using a weighted average for higher robustness."""
    if is_rotten:
        return "C"

    c = float(scores["colour"]["score"])
    s = float(scores["shape"]["score"])
    r = float(scores["ripeness"]["score"])

    # 1. Hard Floor: If any metric is truly terrible, it's a C
    if c < 35.0 or s < 40.0 or r < 35.0:
        return "C"

    # 2. Weighted Calculation (Shape is often most reliable for quality)
    # Weights: Colour 30%, Shape 40%, Ripeness 30%
    weighted_score = (c * 0.30) + (s * 0.40) + (r * 0.30)

    if weighted_score >= 68.0:
        return "A"
    if weighted_score >= 50.0:
        return "B"
    return "C"


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
    quality_scores: Dict[str, Any],
) -> Dict[str, object]:
    """End-to-end post-processing for a single prediction.

    The returned quality_scores object preserves score and justification for
    explainable UI rendering in downstream consumers.
    """
    normalized_label = normalize_label(label)
    scores = validate_quality_scores(quality_scores)
    is_rotten = normalized_label == "rotten"

    grade = assign_overall_grade(scores, is_rotten=is_rotten)
    inventory_action = update_inventory_and_discount(grade)

    return {
        "input_label": label,
        "normalized_label": normalized_label,
        "defect_detected": is_rotten,
        "input_confidence": round(clamp(confidence, 0.0, 1.0), 4),
        "quality_scores": {
            "colour": scores["colour"],
            "shape": scores["shape"],
            "ripeness": scores["ripeness"],
        },
        "overall_grade": grade,
        "inventory_action": inventory_action,
    }


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


class RobustImageFolder(datasets.ImageFolder):
    """ImageFolder variant that filters unreadable files up front."""

    def __init__(
        self,
        root: str,
        transform=None,
        target_transform=None,
        loader=_safe_pil_loader,
        validated_samples: List[Tuple[str, int]] | None = None,
    ) -> None:
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
        )

        if validated_samples is not None:
            self.samples = list(validated_samples)
        else:
            self.samples = self._filter_valid_samples(self.samples)

        if not self.samples:
            raise ValueError(f"No readable image files found in dataset directory: {root}")

        self.imgs = self.samples
        self.targets = [target for _, target in self.samples]

    @staticmethod
    def _is_readable_image(path: str) -> bool:
        try:
            _safe_pil_loader(path)
            return True
        except (UnidentifiedImageError, OSError, ValueError):
            return False

    @classmethod
    def _filter_valid_samples(cls, samples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        valid = [(path, target) for path, target in samples if cls._is_readable_image(path)]
        skipped = len(samples) - len(valid)
        if skipped > 0:
            print(f"Skipping {skipped} unreadable image file(s) in dataset.")
        return valid

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Unreadable image encountered after validation: {path}") from exc

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


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
    return sum(
        1
        for p in directory.iterdir()
        if p.is_dir() and _contains_supported_image(p, recursive=False)
    )


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
        AddGaussianNoise(mean=0.0, std=0.03),
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
    for idx, target in enumerate(targets):
        class_to_indices.setdefault(target, []).append(idx)

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
    return torch.clamp(weights, max=10.0)


def create_dataloaders(
    dataset_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, List[str], torch.Tensor]:
    """Build train/val loaders and class weights using a stratified 80/20 split."""
    resolved = resolve_dataset_root(dataset_dir)
    if resolved is None:
        raise FileNotFoundError(
            f"Dataset directory not found or invalid ImageFolder structure: {dataset_dir}"
        )

    train_tf, val_tf = build_transforms(image_size=image_size)
    base_dataset = RobustImageFolder(root=str(resolved))
    if len(base_dataset) == 0:
        raise ValueError(f"No images found in dataset directory: {resolved}")

    class_names = base_dataset.classes
    train_idx, val_idx = split_indices(base_dataset.targets, train_ratio=0.8, seed=seed)
    class_weights = compute_class_weights(
        [base_dataset.targets[i] for i in train_idx],
        num_classes=len(class_names),
    )

    train_ds = RobustImageFolder(
        root=str(resolved),
        transform=train_tf,
        validated_samples=base_dataset.samples,
    )
    val_ds = RobustImageFolder(
        root=str(resolved),
        transform=val_tf,
        validated_samples=base_dataset.samples,
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader, class_names, class_weights


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class ProduceHealthClassifier(nn.Module):
    """EfficientNetV2-S gatekeeper classifier for Fresh vs Rotten prediction."""

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feature_dim, num_classes),
        )

    def train(self, mode: bool = True) -> "ProduceHealthClassifier":
        """Keep frozen BN layers in eval mode while training unfrozen layers."""
        super().train(mode)
        if mode:
            for module in self.backbone.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    if not any(p.requires_grad for p in module.parameters()):
                        module.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        logits = self.classifier_head(features)
        return logits


def build_model(
    num_classes: int,
    device: torch.device,
    use_pretrained: bool = True,
) -> nn.Module:
    """Create transfer-learning gatekeeper model with EfficientNetV2-S backbone."""
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
                print(
                    f"Warning: Could not download pretrained weights ({retry_error}). "
                    "Falling back to randomly initialized weights."
                )
                backbone = efficientnet_v2_s(weights=None)
    else:
        backbone = efficientnet_v2_s(weights=None)

    if pretrained_loaded:
        for param in backbone.parameters():
            param.requires_grad = False
        for block in backbone.features[-3:]:
            for param in block.parameters():
                param.requires_grad = True

    feature_dim = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()

    model = ProduceHealthClassifier(
        backbone=backbone,
        feature_dim=feature_dim,
        num_classes=num_classes,
    )
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

        if clean.startswith(("backbone.", "classifier_head.")):
            normalised[clean] = value
            continue

        if clean.startswith("quality_head."):
            continue

        if clean.startswith("fc."):
            suffix = clean[len("fc."):]
            for prefix in ("0.", "1."):
                if suffix.startswith(prefix):
                    suffix = suffix[len(prefix):]
                    break
            if suffix in {"weight", "bias"}:
                normalised[f"classifier_head.1.{suffix}"] = value
            continue

        if clean.startswith("classifier.1."):
            suffix = clean[len("classifier.1."):]
            if suffix in {"weight", "bias"}:
                normalised[f"classifier_head.1.{suffix}"] = value
            continue

        normalised[f"backbone.{clean}"] = value

    return normalised


def _list_dataset_images(dataset_root: Path) -> List[Path]:
    """Return all supported image files under dataset_root."""
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return [f for f in dataset_root.rglob("*") if f.is_file() and f.suffix.lower() in image_suffixes]
