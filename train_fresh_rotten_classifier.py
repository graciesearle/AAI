"""Unified Task 2 pipeline: train/infer CNN and post-process quality grades.

This single script includes:
- Multitask transfer-learning CNN training (ResNet-50 backbone)
- Fresh/Rotten classification and Colour/Size/Ripeness regression
- Grade assignment (A/B/C) and inventory action simulation

The project remains fully computer-vision/deep-learning based.
"""

from __future__ import annotations

import copy
import importlib
import os
import random
import shutil
import ssl
import subprocess
import sys
import urllib.request
import warnings
import zipfile
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
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights


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
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    quality_loss_weight: float = 0.1
    quality_warmup_epochs: int = 3
    image_size: int = 224
    patience: int = 20
    early_stop_min_delta: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    # Transfer learning must be enabled by default for Task 2 compliance.
    no_pretrained: bool = False
    save_model_path: Path = Path("fresh_rotten_resnet50.pth")
    save_plot_path: Path = Path("learning_curves.png")
    use_mlp_quality_mapper: bool = True
    quality_mapper_path: Path = Path("quality_mapper_mlp.joblib")
    mapper_hidden_layer_sizes: Tuple[int, ...] = (128, 64)
    mapper_max_iter: int = 250
    mapper_train_max_samples: int = 12000
    predict_image: Path | None = None
    # File-based Kaggle setup (replace placeholder values before first run).
    auto_download_from_kaggle: bool = True
    kaggle_dataset: str = "muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten"
    kaggle_username: str = "YOUR_KAGGLE_USERNAME"
    kaggle_key: str = "YOUR_KAGGLE_KEY"
    kaggle_download_dir: Path = Path(".")


CONFIG = RunConfig(
    dataset_dir=Path("FruitAndVegetableDataset"),
    epochs=20,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=1e-4,
    label_smoothing=0.05,
    quality_loss_weight=0.1,
    quality_warmup_epochs=3,
    image_size=224,
    patience=20,
    early_stop_min_delta=1e-4,
    num_workers=2,
    seed=42,
    no_pretrained=False,
    save_model_path=Path("fresh_rotten_resnet50.pth"),
    save_plot_path=Path("learning_curves.png"),
    use_mlp_quality_mapper=True,
    quality_mapper_path=Path("quality_mapper_mlp.joblib"),
    mapper_hidden_layer_sizes=(128, 64),
    mapper_max_iter=250,
    mapper_train_max_samples=12000,
    predict_image=None,
    auto_download_from_kaggle=True,
    kaggle_dataset="muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten",
    kaggle_username="YOUR_KAGGLE_USERNAME",
    kaggle_key="YOUR_KAGGLE_KEY",
    kaggle_download_dir=Path("."),
)


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

    if scores.colour >= 85.0 and scores.size >= 90.0 and scores.ripeness >= 80.0:
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


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def _compute_quality_and_metrics_from_image(
    image: Image.Image,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Compute quality proxy values and intermediate metrics from an image."""
    rgb = image.convert("RGB").resize((256, 256))
    hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255.0
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]

    mask = ((sat > 0.12) | (val < 0.92)) & (val > 0.08)
    if float(np.mean(mask)) < 0.03:
        mask = np.ones_like(mask, dtype=bool)

    sat_pixels = sat[mask]
    val_pixels = val[mask]
    mean_sat = _safe_mean(sat_pixels)
    mean_val = _safe_mean(val_pixels)
    val_std = _safe_std(val_pixels)

    brown_mask = (
        mask
        & (hue >= 0.05)
        & (hue <= 0.14)
        & (sat >= 0.2)
        & (val <= 0.65)
    )
    brown_ratio = float(np.mean(brown_mask[mask])) if np.any(mask) else 0.0

    area_ratio = float(np.mean(mask))
    ys, xs = np.where(mask)
    if ys.size > 0 and xs.size > 0:
        bbox_height = float(ys.max() - ys.min() + 1)
        bbox_width = float(xs.max() - xs.min() + 1)
        bbox_ratio = (bbox_height * bbox_width) / float(mask.size)
        fill_ratio = area_ratio / max(bbox_ratio, 1e-6)
    else:
        fill_ratio = 0.0

    saturation_score = 100.0 * mean_sat
    brightness_score = 100.0 * clamp(1.0 - abs(mean_val - 0.65) / 0.65, 0.0, 1.0)
    decay_penalty = 35.0 * brown_ratio
    colour = 0.6 * saturation_score + 0.4 * brightness_score - decay_penalty

    size_area_score = 100.0 * clamp((area_ratio - 0.08) / 0.52, 0.0, 1.0)
    size_shape_score = 100.0 * clamp(fill_ratio, 0.0, 1.0)
    size = 0.8 * size_area_score + 0.2 * size_shape_score

    uniformity_score = 100.0 * clamp(1.0 - min(val_std / 0.25, 1.0), 0.0, 1.0)
    ripeness = (
        0.45 * clamp(colour, 0.0, 100.0)
        + 0.30 * uniformity_score
        + 0.25 * (100.0 - 100.0 * brown_ratio)
    )

    quality_array = np.array(
        [
            clamp(colour, 0.0, 100.0),
            clamp(size, 0.0, 100.0),
            clamp(ripeness, 0.0, 100.0),
        ],
        dtype=np.float32,
    )

    metrics = {
        "foreground_ratio": area_ratio,
        "mean_saturation": mean_sat,
        "mean_value": mean_val,
        "value_std": val_std,
        "brown_ratio": brown_ratio,
        "fill_ratio": fill_ratio,
    }
    return quality_array, metrics


def compute_quality_proxy_targets(image: Image.Image) -> np.ndarray:
    """Create proxy quality targets from image pixels for multitask learning."""
    quality_array, _ = _compute_quality_and_metrics_from_image(image)
    return quality_array


def generate_quality_attributes_from_image(
    image_path: str | Path,
) -> tuple[QualityScores, Dict[str, float]]:
    """Estimate quality attributes from image content as a CV fallback path."""
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path).convert("RGB")
    quality_array, metrics = _compute_quality_and_metrics_from_image(image)

    scores = QualityScores(
        colour=round(clamp(float(quality_array[0])), 2),
        size=round(clamp(float(quality_array[1])), 2),
        ripeness=round(clamp(float(quality_array[2])), 2),
    )
    rounded_metrics = {key: round(float(value), 4) for key, value in metrics.items()}
    return scores, rounded_metrics


def process_prediction(
    label: str,
    confidence: float,
    quality_scores: Dict[str, float] | QualityScores | None = None,
    image_path: str | Path | None = None,
    quality_source_override: str | None = None,
) -> Dict[str, object]:
    """End-to-end post-processing for a single deep-model prediction.

    Preferred path:
    - Pass quality_scores from the CNN quality head.

    Fallback path:
    - Pass image_path to estimate quality attributes from image pixels.
    """
    normalized_label = normalize_label(label)

    image_metrics: Dict[str, float] | None = None
    if quality_scores is not None:
        scores = validate_quality_scores(quality_scores)
        quality_source = quality_source_override or "cnn_quality_head"
    elif image_path is not None:
        scores, image_metrics = generate_quality_attributes_from_image(image_path)
        quality_source = "image_postprocessing_fallback"
    else:
        raise ValueError(
            "Provide quality_scores from the CNN quality head, or provide image_path "
            "for CV fallback quality estimation."
        )

    grade = assign_overall_grade(scores)
    inventory_action = update_inventory_and_discount(grade)

    result: Dict[str, object] = {
        "input_label": label,
        "normalized_label": normalized_label,
        "defect_detected": normalized_label == "rotten",
        "input_confidence": round(clamp(confidence, 0.0, 1.0), 4),
        "quality_source": quality_source,
        "quality_scores": {
            "colour": scores.colour,
            "size": scores.size,
            "ripeness": scores.ripeness,
        },
        "overall_grade": grade,
        "inventory_action": inventory_action,
    }

    if image_path is not None:
        result["image_path"] = str(image_path)
    if image_metrics is not None:
        result["image_metrics"] = image_metrics

    return result


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
    """ImageFolder variant that returns proxy quality targets per sample."""

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
            raise ValueError(
                f"No readable image files found in dataset directory: {root}"
            )

        self.imgs = self.samples
        self.targets = [target for _, target in self.samples]

    @staticmethod
    def _is_readable_image(path: str) -> bool:
        """Return True if Pillow can open and verify an image file."""
        try:
            _safe_pil_loader(path)
            return True
        except (UnidentifiedImageError, OSError, ValueError):
            return False

    @classmethod
    def _filter_valid_samples(
        cls,
        samples: List[Tuple[str, int]],
    ) -> List[Tuple[str, int]]:
        """Remove unreadable/corrupt image files from dataset samples."""
        valid_samples = [
            (path, target)
            for path, target in samples
            if cls._is_readable_image(path)
        ]
        skipped = len(samples) - len(valid_samples)
        if skipped > 0:
            print(f"Skipping {skipped} unreadable image file(s) in dataset.")
        return valid_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(
                f"Unreadable image encountered after validation: {path}"
            ) from exc

        quality_target = torch.tensor(
            compute_quality_proxy_targets(sample),
            dtype=torch.float32,
        )

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
    """Return True if a directory contains at least one image file."""
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    for file_path in iterator:
        if file_path.is_file() and file_path.suffix.lower() in image_suffixes:
            return True
    return False


def _count_direct_image_subdirs(directory: Path) -> int:
    """Count immediate subdirectories that contain at least one direct image file."""
    child_dirs = [path for path in directory.iterdir() if path.is_dir()]
    return sum(1 for path in child_dirs if _contains_supported_image(path, recursive=False))


def resolve_dataset_root(dataset_dir: Path) -> Path | None:
    """Resolve dataset root for torchvision ImageFolder, supporting wrappers."""
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return None

    candidates: list[tuple[int, int, Path]] = []

    # Search shallow levels and pick the path with the most plausible class dirs.
    first_level_dirs = [path for path in dataset_dir.iterdir() if path.is_dir()]
    search_roots = [dataset_dir]
    search_roots.extend(first_level_dirs)
    for child in first_level_dirs:
        search_roots.extend(path for path in child.iterdir() if path.is_dir())

    for root in search_roots:
        score = _count_direct_image_subdirs(root)
        if score >= 2:
            depth = len(root.relative_to(dataset_dir).parts) if root != dataset_dir else 0
            candidates.append((score, -depth, root))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][2]

    return None


def _validate_kaggle_placeholders(cfg: RunConfig) -> None:
    """Ensure Kaggle placeholders are replaced with real values before download."""
    if cfg.kaggle_dataset.strip() in {"", "OWNER/DATASET_SLUG"}:
        raise ValueError(
            "Set CONFIG.kaggle_dataset to your Kaggle dataset slug, "
            "for example 'owner/dataset-slug'."
        )

    if cfg.kaggle_username.strip().startswith("YOUR_"):
        raise ValueError("Replace CONFIG.kaggle_username placeholder with your username.")

    if cfg.kaggle_key.strip().startswith("YOUR_"):
        raise ValueError("Replace CONFIG.kaggle_key placeholder with your API key.")


def _download_kaggle_dataset_zip(cfg: RunConfig, force: bool = False) -> Path:
    """Download a Kaggle dataset zip using credentials defined in CONFIG."""
    download_dir = cfg.kaggle_download_dir
    download_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = cfg.kaggle_username
    env["KAGGLE_KEY"] = cfg.kaggle_key

    venv_kaggle = Path(sys.executable).with_name("kaggle")
    if venv_kaggle.exists():
        base_command = [str(venv_kaggle)]
    else:
        system_kaggle = shutil.which("kaggle")
        if system_kaggle:
            base_command = [system_kaggle]
        else:
            base_command = [sys.executable, "-m", "kaggle.cli"]

    command = base_command + [
        "datasets",
        "download",
        "-d",
        cfg.kaggle_dataset,
        "-p",
        str(download_dir),
    ]
    if force:
        command.append("--force")

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        if completed.stdout.strip():
            print(completed.stdout.strip())
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"Kaggle download failed: {details}") from exc

    zip_files = sorted(
        download_dir.glob("*.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not zip_files:
        raise FileNotFoundError(
            f"No zip file found in '{download_dir}' after Kaggle download."
        )

    return zip_files[0]


def _find_existing_dataset_zip(cfg: RunConfig) -> Path | None:
    """Find an already downloaded dataset zip that matches the Kaggle slug."""
    download_dir = cfg.kaggle_download_dir
    if not download_dir.exists() or not download_dir.is_dir():
        return None

    slug_name = cfg.kaggle_dataset.strip().split("/")[-1]
    if slug_name:
        exact = download_dir / f"{slug_name}.zip"
        if exact.exists() and exact.is_file():
            return exact

    candidates = sorted(
        [
            path
            for path in download_dir.glob("*.zip")
            if slug_name in path.stem
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    return None


def _dataset_extract_dir(dataset_path: Path) -> Path:
    """Return the directory where dataset images should be available."""
    if dataset_path.suffix.lower() == ".zip":
        return dataset_path.with_suffix("")
    return dataset_path


def _extract_zip_to_directory(zip_path: Path, target_dir: Path) -> None:
    """Extract a zip archive into target_dir, creating it when needed."""
    if not zip_path.exists() or not zip_path.is_file():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)


def ensure_dataset_available(cfg: RunConfig) -> Path:
    """Return a usable dataset root, downloading from Kaggle when configured."""
    configured_dataset_path = cfg.dataset_dir
    dataset_extract_dir = _dataset_extract_dir(configured_dataset_path)

    if configured_dataset_path.suffix.lower() == ".zip":
        print(
            "CONFIG.dataset_dir points to a zip file. "
            f"Using extraction directory: '{dataset_extract_dir}'."
        )

    resolved_dataset_root = resolve_dataset_root(dataset_extract_dir)
    if resolved_dataset_root is not None:
        print(f"Using dataset directory: {resolved_dataset_root}")
        return resolved_dataset_root

    if configured_dataset_path.suffix.lower() == ".zip" and configured_dataset_path.exists():
        print(f"Found local dataset zip at '{configured_dataset_path}'. Extracting...")
        _extract_zip_to_directory(configured_dataset_path, dataset_extract_dir)
        resolved_dataset_root = resolve_dataset_root(dataset_extract_dir)
        if resolved_dataset_root is not None:
            print(f"Dataset prepared at: {resolved_dataset_root}")
            return resolved_dataset_root

    existing_zip = _find_existing_dataset_zip(cfg)
    if existing_zip is not None:
        print(f"Found existing Kaggle zip at '{existing_zip}'. Extracting...")
        try:
            _extract_zip_to_directory(existing_zip, dataset_extract_dir)
        except zipfile.BadZipFile:
            print(
                f"Existing zip '{existing_zip}' appears corrupted. "
                "Re-downloading from Kaggle..."
            )
        else:
            resolved_dataset_root = resolve_dataset_root(dataset_extract_dir)
            if resolved_dataset_root is not None:
                print(f"Dataset prepared at: {resolved_dataset_root}")
                return resolved_dataset_root

    if not cfg.auto_download_from_kaggle:
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_extract_dir}'. "
            "Either place data there or set CONFIG.auto_download_from_kaggle=True."
        )

    _validate_kaggle_placeholders(cfg)
    print(f"Dataset not found at '{dataset_extract_dir}'. Downloading from Kaggle...")

    downloaded_zip = _download_kaggle_dataset_zip(cfg, force=True)
    print(f"Extracting '{downloaded_zip.name}' to '{dataset_extract_dir}'...")
    _extract_zip_to_directory(downloaded_zip, dataset_extract_dir)

    resolved_dataset_root = resolve_dataset_root(dataset_extract_dir)
    if resolved_dataset_root is None:
        raise ValueError(
            "Dataset extracted but no ImageFolder-compatible structure was found. "
            "Expected class subfolders containing images."
        )

    print(f"Dataset prepared at: {resolved_dataset_root}")
    return resolved_dataset_root


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create train and validation preprocessing pipelines."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            AddGaussianNoise(mean=0.0, std=0.03),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, val_transform


def split_indices(
    targets: List[int],
    train_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Create deterministic stratified train/validation indices by class."""
    if not targets:
        raise ValueError("Cannot split an empty dataset.")

    class_to_indices: Dict[int, List[int]] = {}
    for index, target in enumerate(targets):
        class_to_indices.setdefault(target, []).append(index)

    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for class_indices in class_to_indices.values():
        rng.shuffle(class_indices)

        if len(class_indices) == 1:
            train_indices.extend(class_indices)
            continue

        train_size = int(len(class_indices) * train_ratio)
        train_size = min(max(train_size, 1), len(class_indices) - 1)
        train_indices.extend(class_indices[:train_size])
        val_indices.extend(class_indices[train_size:])

    if not val_indices:
        rng.shuffle(train_indices)
        fallback_val_size = max(1, int(len(train_indices) * (1.0 - train_ratio)))
        val_indices = train_indices[-fallback_val_size:]
        train_indices = train_indices[:-fallback_val_size]

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
    class_counts = torch.bincount(target_tensor, minlength=num_classes).float()
    class_counts = torch.clamp(class_counts, min=1.0)

    weights = class_counts.sum() / (num_classes * class_counts)
    return weights / weights.mean()


def create_dataloaders(
    dataset_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, List[str], torch.Tensor]:
    """Build train/val loaders and class weights using a stratified 80/20 split."""
    resolved_dataset_dir = resolve_dataset_root(dataset_dir)
    if resolved_dataset_dir is None:
        raise FileNotFoundError(
            f"Dataset directory not found or invalid ImageFolder structure: {dataset_dir}"
        )

    train_transform, val_transform = build_transforms(image_size=image_size)

    base_dataset = QualityProxyImageFolder(root=str(resolved_dataset_dir))
    if len(base_dataset) == 0:
        raise ValueError(f"No images found in dataset directory: {resolved_dataset_dir}")

    class_names = base_dataset.classes
    train_indices, val_indices = split_indices(
        targets=base_dataset.targets,
        train_ratio=0.8,
        seed=seed,
    )
    train_targets = [base_dataset.targets[index] for index in train_indices]
    class_weights = compute_class_weights(
        targets=train_targets,
        num_classes=len(class_names),
    )

    train_dataset = QualityProxyImageFolder(
        root=str(resolved_dataset_dir),
        transform=train_transform,
        validated_samples=base_dataset.samples,
    )
    val_dataset = QualityProxyImageFolder(
        root=str(resolved_dataset_dir),
        transform=val_transform,
        validated_samples=base_dataset.samples,
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, class_names, class_weights


class MultiTaskProduceNet(nn.Module):
    """ResNet backbone with classification and quality regression heads."""

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim, num_classes),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 3),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)

        logits = self.classifier_head(features)
        quality_scores = self.quality_head(features) * 100.0
        return logits, quality_scores


def build_model(
    num_classes: int,
    device: torch.device,
    use_pretrained: bool = True,
) -> nn.Module:
    """Create transfer-learning multitask model with a ResNet-50 backbone."""
    pretrained_loaded = False

    if use_pretrained:
        try:
            backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            pretrained_loaded = True
        except (URLError, RuntimeError, OSError) as exc:
            retry_error = exc
            cert_error_text = str(exc).upper()
            if "CERTIFICATE_VERIFY_FAILED" in cert_error_text:
                try:
                    import certifi

                    cafile = certifi.where()
                    os.environ.setdefault("SSL_CERT_FILE", cafile)
                    os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)

                    def _create_https_context() -> ssl.SSLContext:
                        return ssl.create_default_context(cafile=cafile)

                    ssl._create_default_https_context = _create_https_context

                    # Ensure urllib (used internally by torch hub) uses the same CA bundle.
                    https_context = ssl.create_default_context(cafile=cafile)
                    opener = urllib.request.build_opener(
                        urllib.request.HTTPSHandler(context=https_context)
                    )
                    urllib.request.install_opener(opener)

                    backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
                    pretrained_loaded = True
                    print("Loaded pretrained ResNet-50 weights after applying certifi SSL certificates.")
                except (ImportError, URLError, RuntimeError, OSError) as cert_exc:
                    retry_error = cert_exc

            if not pretrained_loaded:
                print(
                    "Warning: Could not download pretrained ResNet-50 weights "
                    f"({retry_error}). Falling back to randomly initialized weights."
                )
                backbone = models.resnet50(weights=None)
    else:
        backbone = models.resnet50(weights=None)

    if pretrained_loaded:
        # Freeze early layers but unfreeze layer3 + layer4 for fine-tuning.
        for param in backbone.parameters():
            param.requires_grad = False
        for module in [backbone.layer3, backbone.layer4]:
            for param in module.parameters():
                param.requires_grad = True

    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()

    model = MultiTaskProduceNet(
        backbone=backbone,
        feature_dim=feature_dim,
        num_classes=num_classes,
    )
    return model.to(device)


def _extract_checkpoint_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    """Normalize common checkpoint formats into current model key names."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary.")

    if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
        raw_state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        raw_state_dict = checkpoint["state_dict"]
    else:
        raw_state_dict = checkpoint

    if not isinstance(raw_state_dict, dict) or not raw_state_dict:
        raise ValueError("Checkpoint does not contain a usable state dict.")
    if not all(torch.is_tensor(value) for value in raw_state_dict.values()):
        raise ValueError("Checkpoint state dict values must be tensors.")

    normalized_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in raw_state_dict.items():
        clean_key = key[len("module."):] if key.startswith("module.") else key

        if clean_key.startswith(("backbone.", "classifier_head.", "quality_head.")):
            normalized_state_dict[clean_key] = value
            continue

        if clean_key.startswith("fc."):
            suffix = clean_key[len("fc."):]
            if suffix.startswith("0."):
                suffix = suffix[len("0."):]
            elif suffix.startswith("1."):
                suffix = suffix[len("1."):]
            if suffix in {"weight", "bias"}:
                normalized_state_dict[f"classifier_head.1.{suffix}"] = value
            continue

        normalized_state_dict[f"backbone.{clean_key}"] = value

    return normalized_state_dict


def _remap_classifier_to_dataset_classes(
    state_dict: Dict[str, torch.Tensor],
    checkpoint_class_names: List[str] | None,
    target_class_names: List[str],
) -> Dict[str, torch.Tensor]:
    """Align classifier rows by class name without changing model class count."""
    weight_key = "classifier_head.1.weight"
    bias_key = "classifier_head.1.bias"

    if checkpoint_class_names is None:
        return state_dict
    if weight_key not in state_dict or bias_key not in state_dict:
        return state_dict

    src_weight = state_dict[weight_key]
    src_bias = state_dict[bias_key]
    if src_weight.ndim != 2 or src_bias.ndim != 1:
        return state_dict

    if len(checkpoint_class_names) != src_weight.shape[0] or len(checkpoint_class_names) != src_bias.shape[0]:
        raise RuntimeError(
            "Checkpoint class_names does not match classifier tensor sizes."
        )

    class_to_index = {name: idx for idx, name in enumerate(checkpoint_class_names)}
    missing_classes = [name for name in target_class_names if name not in class_to_index]
    if missing_classes:
        missing_preview = ", ".join(missing_classes[:6])
        raise RuntimeError(
            "Checkpoint is missing one or more dataset classes needed for remapping: "
            f"{missing_preview}."
        )

    remapped_weight = torch.empty(
        (len(target_class_names), src_weight.shape[1]),
        dtype=src_weight.dtype,
        device=src_weight.device,
    )
    remapped_bias = torch.empty(
        (len(target_class_names),),
        dtype=src_bias.dtype,
        device=src_bias.device,
    )

    for target_index, class_name in enumerate(target_class_names):
        source_index = class_to_index[class_name]
        remapped_weight[target_index] = src_weight[source_index]
        remapped_bias[target_index] = src_bias[source_index]

    remapped_state = dict(state_dict)
    remapped_state[weight_key] = remapped_weight
    remapped_state[bias_key] = remapped_bias
    return remapped_state


def _list_dataset_images(dataset_root: Path) -> List[Path]:
    """Return all supported image files under dataset_root."""
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return [
        file_path
        for file_path in dataset_root.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in image_suffixes
    ]


def _mlp_mapper_dependencies_available() -> bool:
    """Return True when optional dependencies for the MLP mapper are installed."""
    try:
        _import_mlp_mapper_dependencies()
    except (ModuleNotFoundError, AttributeError):
        return False
    return True


def _import_mlp_mapper_dependencies() -> Tuple[Any, Any, Any, Any]:
    """Dynamically import optional sklearn/joblib dependencies for mapper usage."""
    joblib_module = importlib.import_module("joblib")
    sklearn_nn = importlib.import_module("sklearn.neural_network")
    sklearn_pipeline = importlib.import_module("sklearn.pipeline")
    sklearn_preprocessing = importlib.import_module("sklearn.preprocessing")
    return (
        joblib_module,
        getattr(sklearn_nn, "MLPRegressor"),
        getattr(sklearn_pipeline, "Pipeline"),
        getattr(sklearn_preprocessing, "StandardScaler"),
    )


def _build_mapper_features_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Build mapper features from CNN classification outputs."""
    probabilities = torch.softmax(logits, dim=1)
    confidence = probabilities.max(dim=1, keepdim=True).values
    return torch.cat([logits, probabilities, confidence], dim=1)


def _extract_mapper_dataset_from_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect features/targets for mapper training from a dataloader."""
    model.eval()
    feature_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []
    remaining = max_samples if max_samples > 0 else None

    with torch.no_grad():
        for images, _labels, quality_targets in loader:
            images = images.to(device)
            logits, _ = model(images)

            feature_batch = (
                _build_mapper_features_from_logits(logits)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            target_batch = quality_targets.cpu().numpy().astype(np.float32)

            if remaining is not None:
                if remaining <= 0:
                    break
                keep = min(remaining, feature_batch.shape[0])
                feature_batch = feature_batch[:keep]
                target_batch = target_batch[:keep]
                remaining -= keep

            if feature_batch.size == 0:
                continue

            feature_chunks.append(feature_batch)
            target_chunks.append(target_batch)

            if remaining is not None and remaining <= 0:
                break

    if not feature_chunks:
        raise ValueError("No samples extracted for MLP quality mapper training.")

    return np.concatenate(feature_chunks, axis=0), np.concatenate(target_chunks, axis=0)


def _compute_quality_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE for colour, size, and ripeness outputs."""
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    return {
        "colour_mae": float(mae[0]),
        "size_mae": float(mae[1]),
        "ripeness_mae": float(mae[2]),
        "overall_mae": float(np.mean(mae)),
    }


def load_mlp_quality_mapper(
    mapper_path: Path,
    expected_input_dim: int,
) -> Any | None:
    """Load a persisted MLP quality mapper when dimensions are compatible."""
    if not mapper_path.exists() or not mapper_path.is_file():
        return None
    if not _mlp_mapper_dependencies_available():
        return None

    try:
        joblib_module, _mlp_cls, _pipeline_cls, _scaler_cls = _import_mlp_mapper_dependencies()
        loaded_obj = joblib_module.load(mapper_path)
    except Exception as exc:
        print(f"Warning: Failed to load MLP quality mapper ({exc}).")
        return None

    if isinstance(loaded_obj, dict):
        mapper = loaded_obj.get("mapper")
        saved_dim = loaded_obj.get("expected_input_dim")
    else:
        mapper = loaded_obj
        saved_dim = getattr(mapper, "n_features_in_", None)

    if mapper is None:
        return None

    if saved_dim is not None and int(saved_dim) != expected_input_dim:
        print(
            "Existing MLP quality mapper feature dimension does not match current model. "
            "Will retrain mapper."
        )
        return None

    return mapper


def train_mlp_quality_mapper(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: RunConfig,
) -> Any:
    """Train and save an MLP mapper from CNN outputs to quality percentages."""
    if not _mlp_mapper_dependencies_available():
        raise RuntimeError(
            "MLP mapper dependencies are missing. Install scikit-learn and joblib."
        )

    joblib_module, mlp_regressor_cls, pipeline_cls, scaler_cls = _import_mlp_mapper_dependencies()

    x_train, y_train = _extract_mapper_dataset_from_loader(
        model=model,
        loader=train_loader,
        device=device,
        max_samples=cfg.mapper_train_max_samples,
    )

    val_cap = max(min(cfg.mapper_train_max_samples // 4, 4000), 1000)
    x_val, y_val = _extract_mapper_dataset_from_loader(
        model=model,
        loader=val_loader,
        device=device,
        max_samples=val_cap,
    )

    mapper = pipeline_cls(
        [
            ("scaler", scaler_cls()),
            (
                "mlp",
                mlp_regressor_cls(
                    hidden_layer_sizes=cfg.mapper_hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=cfg.mapper_max_iter,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    random_state=cfg.seed,
                ),
            ),
        ]
    )
    mapper.fit(x_train, y_train)

    y_val_pred = np.clip(mapper.predict(x_val), 0.0, 100.0)
    mae_report = _compute_quality_mae(y_val, y_val_pred)
    print(
        "MLP quality mapper validation MAE "
        f"(colour/size/ripeness/overall): "
        f"{mae_report['colour_mae']:.2f}/"
        f"{mae_report['size_mae']:.2f}/"
        f"{mae_report['ripeness_mae']:.2f}/"
        f"{mae_report['overall_mae']:.2f}"
    )

    payload = {
        "mapper": mapper,
        "expected_input_dim": int(x_train.shape[1]),
        "target_names": ["colour", "size", "ripeness"],
        "algorithm": "sklearn_mlp_regressor",
    }
    cfg.quality_mapper_path.parent.mkdir(parents=True, exist_ok=True)
    joblib_module.dump(payload, cfg.quality_mapper_path)
    print(f"Saved MLP quality mapper to: {cfg.quality_mapper_path}")
    return mapper


def predict_quality_with_mlp_mapper(
    model: nn.Module,
    mapper: Any,
    image_path: Path,
    image_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """Predict quality scores using the MLP mapper on CNN outputs."""
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits, _ = model(image_tensor)
        mapper_features = (
            _build_mapper_features_from_logits(logits)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    prediction = np.asarray(mapper.predict(mapper_features), dtype=np.float32).reshape(-1)
    if prediction.shape[0] < 3:
        raise RuntimeError("MLP mapper prediction is invalid; expected 3 outputs.")

    prediction = np.clip(prediction[:3], 0.0, 100.0)
    return {
        "colour": round(float(prediction[0]), 2),
        "size": round(float(prediction[1]), 2),
        "ripeness": round(float(prediction[2]), 2),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cls_criterion: nn.Module,
    quality_criterion: nn.Module,
    quality_loss_weight: float,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Run multitask evaluation on a data loader."""
    model.eval()
    total_loss_sum = 0.0
    cls_loss_sum = 0.0
    quality_loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, quality_targets in loader:
            images = images.to(device)
            labels = labels.to(device)
            quality_targets = quality_targets.to(device)

            logits, quality_preds = model(images)
            cls_loss = cls_criterion(logits, labels)
            quality_loss = quality_criterion(quality_preds, quality_targets)
            total_loss = cls_loss + quality_loss_weight * quality_loss

            batch_size = labels.size(0)
            total_loss_sum += total_loss.item() * batch_size
            cls_loss_sum += cls_loss.item() * batch_size
            quality_loss_sum += quality_loss.item() * batch_size

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    avg_total_loss = total_loss_sum / max(total, 1)
    avg_cls_loss = cls_loss_sum / max(total, 1)
    avg_quality_loss = quality_loss_sum / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)

    return avg_total_loss, accuracy, avg_cls_loss, avg_quality_loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cls_criterion: nn.Module,
    quality_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    quality_loss_weight: float,
    quality_warmup_epochs: int,
    device: torch.device,
    num_epochs: int,
    patience: int,
    early_stop_min_delta: float,
) -> Tuple[nn.Module, History]:
    """Train multitask model with early stopping on validation cls loss."""
    history = History(
        train_total_loss=[],
        val_total_loss=[],
        train_cls_loss=[],
        val_cls_loss=[],
        train_quality_loss=[],
        val_quality_loss=[],
        train_acc=[],
        val_acc=[],
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_cls_loss = float("inf")
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        effective_quality_loss_weight = (
            quality_loss_weight if epoch >= quality_warmup_epochs else 0.0
        )
        train_total_loss_sum = 0.0
        train_cls_loss_sum = 0.0
        train_quality_loss_sum = 0.0
        running_correct = 0
        running_total = 0

        for images, labels, quality_targets in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            quality_targets = quality_targets.to(device)

            optimizer.zero_grad()
            logits, quality_preds = model(images)

            cls_loss = cls_criterion(logits, labels)
            quality_loss = quality_criterion(quality_preds, quality_targets)
            total_loss = cls_loss + effective_quality_loss_weight * quality_loss

            total_loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_total_loss_sum += total_loss.item() * batch_size
            train_cls_loss_sum += cls_loss.item() * batch_size
            train_quality_loss_sum += quality_loss.item() * batch_size

            predictions = torch.argmax(logits, dim=1)
            running_correct += (predictions == labels).sum().item()
            running_total += batch_size

        train_total_loss = train_total_loss_sum / max(running_total, 1)
        train_cls_loss = train_cls_loss_sum / max(running_total, 1)
        train_quality_loss = train_quality_loss_sum / max(running_total, 1)
        train_acc = 100.0 * running_correct / max(running_total, 1)

        val_total_loss, val_acc, val_cls_loss, val_quality_loss = evaluate(
            model=model,
            loader=val_loader,
            cls_criterion=cls_criterion,
            quality_criterion=quality_criterion,
            quality_loss_weight=quality_loss_weight,
            device=device,
        )

        history.train_total_loss.append(train_total_loss)
        history.val_total_loss.append(val_total_loss)
        history.train_cls_loss.append(train_cls_loss)
        history.val_cls_loss.append(val_cls_loss)
        history.train_quality_loss.append(train_quality_loss)
        history.val_quality_loss.append(val_quality_loss)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Total: {train_total_loss:.4f} | Train Cls: {train_cls_loss:.4f} | "
            f"Train Q: {train_quality_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Total: {val_total_loss:.4f} | Val Cls: {val_cls_loss:.4f} | "
            f"Val Q: {val_quality_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"QWeight: {effective_quality_loss_weight:.3f}"
        )

        improved_loss = val_cls_loss < (best_val_cls_loss - early_stop_min_delta)
        improved_acc_tie_break = (
            abs(val_cls_loss - best_val_cls_loss) <= early_stop_min_delta
            and val_acc > best_val_acc
        )

        if improved_loss or improved_acc_tie_break:
            best_val_cls_loss = val_cls_loss
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_model_wts)
    return model, history


def plot_learning_curves(history: History, save_path: Path | None = None) -> None:
    """Plot multitask training curves."""
    epochs = range(1, len(history.train_total_loss) + 1)

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, history.train_total_loss, label="Train Total Loss")
    plt.plot(epochs, history.val_total_loss, label="Val Total Loss")
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history.train_cls_loss, label="Train Classification Loss")
    plt.plot(epochs, history.val_cls_loss, label="Val Classification Loss")
    plt.title("Classification Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history.train_quality_loss, label="Train Quality Loss")
    plt.plot(epochs, history.val_quality_loss, label="Val Quality Loss")
    plt.title("Quality Regression Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history.train_acc, label="Train Accuracy")
    plt.plot(epochs, history.val_acc, label="Val Accuracy")
    plt.title("Classification Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def predict_single_image(
    model: nn.Module,
    image_path: Path,
    class_names: List[str],
    image_size: int,
    device: torch.device,
) -> Tuple[str, float, Dict[str, float]]:
    """Predict class label, confidence, and quality percentages for one image."""
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits, quality_predictions = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probabilities, dim=1)

    predicted_label = class_names[pred_idx.item()]
    confidence_pct = confidence.item() * 100.0

    quality_np = torch.clamp(quality_predictions.squeeze(0), 0.0, 100.0).cpu().numpy()
    quality_scores = {
        "colour": round(float(quality_np[0]), 2),
        "size": round(float(quality_np[1]), 2),
        "ripeness": round(float(quality_np[2]), 2),
    }
    return predicted_label, confidence_pct, quality_scores


def main() -> None:
    """Run end-to-end multitask training, evaluation, plotting, and inference."""
    cfg = CONFIG
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_root = ensure_dataset_available(cfg)

    train_loader, val_loader, class_names, class_weights = create_dataloaders(
        dataset_dir=dataset_root,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(
        num_classes=len(class_names),
        device=device,
        use_pretrained=not cfg.no_pretrained,
    )

    cls_criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=cfg.label_smoothing,
    )
    quality_criterion = nn.SmoothL1Loss()

    # Differential learning rates: backbone fine-tuned layers get lower LR.
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "backbone" in n]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "backbone" not in n]
    optimizer = Adam(
        [
            {"params": backbone_params, "lr": cfg.learning_rate * 0.1},
            {"params": head_params, "lr": cfg.learning_rate},
        ],
        weight_decay=cfg.weight_decay,
    )
    trained_now = False
    has_quality_head_weights = True
    history = History(
        train_total_loss=[],
        val_total_loss=[],
        train_cls_loss=[],
        val_cls_loss=[],
        train_quality_loss=[],
        val_quality_loss=[],
        train_acc=[],
        val_acc=[],
    )

    if cfg.save_model_path.exists():
        checkpoint = torch.load(cfg.save_model_path, map_location=device)
        checkpoint_class_names: List[str] | None = None
        if isinstance(checkpoint, dict):
            raw_class_names = checkpoint.get("class_names")
            if isinstance(raw_class_names, list) and raw_class_names and all(
                isinstance(name, str) for name in raw_class_names
            ):
                checkpoint_class_names = raw_class_names

        checkpoint_state_dict = _extract_checkpoint_state_dict(checkpoint)
        checkpoint_state_dict = _remap_classifier_to_dataset_classes(
            state_dict=checkpoint_state_dict,
            checkpoint_class_names=checkpoint_class_names,
            target_class_names=class_names,
        )

        try:
            incompatible = model.load_state_dict(checkpoint_state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint could not be loaded into the current dataset class layout. "
                "If this checkpoint uses different classes, retrain a new model."
            ) from exc

        disallowed_missing = [
            key for key in incompatible.missing_keys if not key.startswith("quality_head.")
        ]
        if disallowed_missing or incompatible.unexpected_keys:
            missing_preview = ", ".join(disallowed_missing[:6]) or "none"
            unexpected_preview = ", ".join(incompatible.unexpected_keys[:6]) or "none"
            raise RuntimeError(
                "Checkpoint is incompatible with the current model. "
                f"Missing keys: {missing_preview}. Unexpected keys: {unexpected_preview}."
            )

        has_quality_head_weights = not any(
            key.startswith("quality_head.") for key in incompatible.missing_keys
        )
        print(f"Loaded existing model from: {cfg.save_model_path}")
        if not has_quality_head_weights:
            print(
                "Checkpoint has no trained quality head. "
                "Using image-based fallback quality scoring for post-processing."
            )
    else:
        print("No existing model checkpoint found. Training from scratch.")
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cls_criterion=cls_criterion,
            quality_criterion=quality_criterion,
            optimizer=optimizer,
            quality_loss_weight=cfg.quality_loss_weight,
            quality_warmup_epochs=cfg.quality_warmup_epochs,
            device=device,
            num_epochs=cfg.epochs,
            patience=cfg.patience,
            early_stop_min_delta=cfg.early_stop_min_delta,
        )
        trained_now = True

    if trained_now:
        train_total, train_acc, train_cls, train_quality = evaluate(
            model=model,
            loader=train_loader,
            cls_criterion=cls_criterion,
            quality_criterion=quality_criterion,
            quality_loss_weight=cfg.quality_loss_weight,
            device=device,
        )
        val_total, val_acc, val_cls, val_quality = evaluate(
            model=model,
            loader=val_loader,
            cls_criterion=cls_criterion,
            quality_criterion=quality_criterion,
            quality_loss_weight=cfg.quality_loss_weight,
            device=device,
        )

        print("\nFinal Evaluation Metrics")
        print(f"Train Total Loss: {train_total:.4f}")
        print(f"Train Classification Loss: {train_cls:.4f}")
        print(f"Train Quality Loss: {train_quality:.4f}")
        print(f"Train Accuracy: {train_acc:.2f}%")
        print(f"Validation Total Loss: {val_total:.4f}")
        print(f"Validation Classification Loss: {val_cls:.4f}")
        print(f"Validation Quality Loss: {val_quality:.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": cfg.image_size,
                "architecture": "resnet50_multitask_transfer_learning",
                "quality_head": "deep_regression_head",
            },
            cfg.save_model_path,
        )
        print(f"Saved model weights to: {cfg.save_model_path}")

        plot_learning_curves(history=history, save_path=cfg.save_plot_path)
        print(f"Saved learning curves to: {cfg.save_plot_path}")
    else:
        print("Skipped full evaluation, saving, and plotting because training was not run.")

    quality_mapper = None
    if cfg.use_mlp_quality_mapper:
        if _mlp_mapper_dependencies_available():
            expected_mapper_dim = (2 * len(class_names)) + 1
            quality_mapper = load_mlp_quality_mapper(
                mapper_path=cfg.quality_mapper_path,
                expected_input_dim=expected_mapper_dim,
            )
            if quality_mapper is not None:
                print(f"Loaded existing MLP quality mapper from: {cfg.quality_mapper_path}")
            else:
                print("No compatible MLP quality mapper found. Training mapper...")
                quality_mapper = train_mlp_quality_mapper(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    cfg=cfg,
                )
        else:
            print(
                "Warning: scikit-learn/joblib not installed; "
                "using existing quality-score path without MLP mapper."
            )

    prediction_image = cfg.predict_image
    if prediction_image is None:
        dataset_images = _list_dataset_images(dataset_root)
        if not dataset_images:
            raise FileNotFoundError(
                f"No supported image files found for prediction in: {dataset_root}"
            )
        prediction_image = random.SystemRandom().choice(dataset_images)
        print(f"Selected random dataset image for prediction: {prediction_image}")

    if not prediction_image.exists() or not prediction_image.is_file():
        raise FileNotFoundError(f"Prediction image not found: {prediction_image}")

    label, confidence_pct, quality_scores = predict_single_image(
        model=model,
        image_path=prediction_image,
        class_names=class_names,
        image_size=cfg.image_size,
        device=device,
    )
    if quality_mapper is not None:
        mapped_quality_scores = predict_quality_with_mlp_mapper(
            model=model,
            mapper=quality_mapper,
            image_path=prediction_image,
            image_size=cfg.image_size,
            device=device,
        )
        result = process_prediction(
            label=label,
            confidence=confidence_pct / 100.0,
            quality_scores=mapped_quality_scores,
            quality_source_override="cnn_output_mlp_mapper",
        )
    elif has_quality_head_weights:
        result = process_prediction(
            label=label,
            confidence=confidence_pct / 100.0,
            quality_scores=quality_scores,
        )
    else:
        result = process_prediction(
            label=label,
            confidence=confidence_pct / 100.0,
            image_path=prediction_image,
        )

    print(
        f"Prediction for '{prediction_image}': {label} "
        f"(confidence: {confidence_pct:.2f}%)"
    )
    print(f"Quality Scores: {result['quality_scores']}")
    print(f"Overall Grade: {result['overall_grade']}")
    print(f"Inventory Action: {result['inventory_action']}")


if __name__ == "__main__":
    main()
