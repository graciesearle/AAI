"""Train and evaluate a fresh-vs-rotten produce classifier using PyTorch.

This script builds an image classification pipeline for the
"Fruit and Vegetable Disease (Healthy vs Rotten)" dataset, where classes are
stored in subdirectories (e.g., Apple__Healthy, Apple__Rotten).

Features:
- Data loading with train/validation split.
- Data augmentation (flip, rotation, color jitter, Gaussian noise).
- Transfer learning with a frozen pre-trained ResNet-50 backbone.
- Custom classification head for multi-class prediction.
- Early stopping on validation loss.
- Training/validation loss and accuracy reporting.
- Learning curve plotting.
- Single-image prediction with confidence score.
"""

from __future__ import annotations

import copy
import random
from urllib.error import URLError
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ROOT = REPO_ROOT / "models"
DEFAULT_MODEL_NAME = "produce-quality"
DEFAULT_MODEL_VERSION = "auto"
DEFAULT_PLOT_OUTPUT_DIR = REPO_ROOT / "docs" / "task2"


def _parse_semver(version: str) -> Tuple[int, int, int] | None:
    parts = version.split(".")
    if len(parts) != 3:
        return None

    try:
        major, minor, patch = (int(part) for part in parts)
    except ValueError:
        return None

    if major < 0 or minor < 0 or patch < 0:
        return None

    return major, minor, patch


def _next_model_version(*, model_root: Path, model_name: str) -> str:
    model_dir = model_root / model_name
    if not model_dir.exists() or not model_dir.is_dir():
        return "1.0.0"

    parsed_versions: list[Tuple[int, int, int]] = []
    for child in model_dir.iterdir():
        if not child.is_dir():
            continue
        parsed = _parse_semver(child.name)
        if parsed is not None:
            parsed_versions.append(parsed)

    if not parsed_versions:
        return "1.0.0"

    major, minor, patch = max(parsed_versions)
    return f"{major}.{minor}.{patch + 1}"


def resolve_output_paths(cfg: "RunConfig") -> tuple[str, Path, Path]:
    configured_version = cfg.model_version.strip()
    if configured_version.lower() == "auto":
        resolved_version = _next_model_version(model_root=cfg.model_root, model_name=cfg.model_name)
    else:
        resolved_version = configured_version

    model_output_path = (
        cfg.model_root / cfg.model_name / resolved_version / "artifacts" / "model.pth"
    )
    plot_output_path = (
        cfg.plot_output_dir
        / f"task2_learning_curves_{cfg.model_name}_{resolved_version}.png"
    )
    return resolved_version, model_output_path, plot_output_path


@dataclass
class History:
    """Stores training and validation metrics per epoch.

    Attributes:
        train_loss: List of training loss values for each epoch.
        val_loss: List of validation loss values for each epoch.
        train_acc: List of training accuracy values for each epoch.
        val_acc: List of validation accuracy values for each epoch.
    """

    train_loss: List[float]
    val_loss: List[float]
    train_acc: List[float]
    val_acc: List[float]


@dataclass
class RunConfig:
    """Editable training configuration for local runs/Colab notebooks.

    Update these values directly in code instead of passing CLI arguments.
    """

    dataset_dir: Path = Path("FruitAndVegetableDataset")
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    image_size: int = 224
    patience: int = 5
    num_workers: int = 2
    seed: int = 42
    no_pretrained: bool = False
    model_root: Path = DEFAULT_MODEL_ROOT
    model_name: str = DEFAULT_MODEL_NAME
    model_version: str = DEFAULT_MODEL_VERSION
    plot_output_dir: Path = DEFAULT_PLOT_OUTPUT_DIR
    predict_image: Path | None = None


# Edit this block directly when running in Colab.
# Example dataset path after Kaggle download/unzip is often: /content/FruitAndVegetableDataset
CONFIG = RunConfig(
    dataset_dir=Path("FruitAndVegetableDataset"),
    epochs=20,
    batch_size=32,
    learning_rate=1e-3,
    image_size=224,
    patience=5,
    num_workers=2,
    seed=42,
    no_pretrained=False,
    model_root=DEFAULT_MODEL_ROOT,
    model_name=DEFAULT_MODEL_NAME,
    model_version=DEFAULT_MODEL_VERSION,
    plot_output_dir=DEFAULT_PLOT_OUTPUT_DIR,
    predict_image=None,
)


class AddGaussianNoise:
    """Apply additive Gaussian noise to an image tensor.

    This transform expects a tensor image and adds random noise sampled from
    a normal distribution. Values are clamped to keep pixel intensities valid.

    Args:
        mean: Mean of the Gaussian distribution.
        std: Standard deviation of the Gaussian distribution.

    Returns:
        A noisy tensor with values clamped to [0, 1].
    """

    def __init__(self, mean: float = 0.0, std: float = 0.03) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to a tensor image.

        Args:
            tensor: Input image tensor with shape [C, H, W].

        Returns:
            Transformed tensor with additive noise.
        """
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value for random generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create train and validation preprocessing pipelines.

    Args:
        image_size: Target size for both height and width.

    Returns:
        A tuple of (train_transform, val_transform).
    """
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
    num_samples: int,
    train_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Create deterministic train/validation indices.

    Args:
        num_samples: Total number of dataset samples.
        train_ratio: Fraction of samples assigned to training.
        seed: Random seed for reproducible shuffling.

    Returns:
        A tuple containing (train_indices, val_indices).
    """
    indices = list(range(num_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_size = int(num_samples * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return train_indices, val_indices


def create_dataloaders(
    dataset_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Build train and validation data loaders using an 80/20 split.

    Args:
        dataset_dir: Root directory containing class subfolders.
        image_size: Target image dimension used for resizing.
        batch_size: Number of samples per training batch.
        num_workers: Number of worker processes for data loading.
        seed: Random seed used for deterministic splitting.

    Returns:
        A tuple of (train_loader, val_loader, class_names).

    Raises:
        FileNotFoundError: If the dataset directory does not exist.
        ValueError: If no images are found in the dataset directory.
    """
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    train_transform, val_transform = build_transforms(image_size=image_size)

    base_dataset = datasets.ImageFolder(root=str(dataset_dir))
    if len(base_dataset) == 0:
        raise ValueError(f"No images found in dataset directory: {dataset_dir}")

    class_names = base_dataset.classes
    train_indices, val_indices = split_indices(
        num_samples=len(base_dataset),
        train_ratio=0.8,
        seed=seed,
    )

    train_dataset = datasets.ImageFolder(
        root=str(dataset_dir),
        transform=train_transform,
    )
    val_dataset = datasets.ImageFolder(
        root=str(dataset_dir),
        transform=val_transform,
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

    return train_loader, val_loader, class_names


def build_model(
    num_classes: int,
    device: torch.device,
    use_pretrained: bool = True,
) -> nn.Module:
    """Create a transfer-learning classifier with frozen ResNet-50 features.

    Args:
        num_classes: Number of output classes.
        device: Target device where model is allocated.
        use_pretrained: Whether to initialize with ImageNet pretrained weights.

    Returns:
        A configured and device-mapped PyTorch model.
    """
    if use_pretrained:
        try:
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        except (URLError, RuntimeError, OSError) as exc:
            print(
                "Warning: Could not download pretrained ResNet-50 weights "
                f"({exc}). Falling back to randomly initialized weights."
            )
            model = models.resnet50(weights=None)
    else:
        model = models.resnet50(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    return model.to(device)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run model evaluation on a data loader.

    Args:
        model: Neural network model.
        loader: DataLoader for evaluation data.
        criterion: Loss function used for scoring.
        device: Compute device.

    Returns:
        A tuple containing (average_loss, accuracy_percent).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    patience: int,
) -> Tuple[nn.Module, History]:
    """Train the model with early stopping based on validation loss.

    Args:
        model: Model to train.
        train_loader: DataLoader for training samples.
        val_loader: DataLoader for validation samples.
        criterion: Loss function.
        optimizer: Optimizer used for parameter updates.
        device: Compute device.
        num_epochs: Maximum number of training epochs.
        patience: Number of epochs to wait for val loss improvement.

    Returns:
        A tuple of (best_model, history), where best_model contains the
        parameters with the lowest validation loss.
    """
    history = History(train_loss=[], val_loss=[], train_acc=[], val_acc=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predictions = torch.argmax(logits, dim=1)
            running_correct += (predictions == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc = 100.0 * running_correct / max(running_total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
    """Plot training and validation loss/accuracy curves.

    Args:
        history: Training history containing metric values per epoch.
        save_path: Optional output path for saving the plot image.
    """
    epochs = range(1, len(history.train_loss) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.train_loss, label="Train Loss")
    plt.plot(epochs, history.val_loss, label="Val Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.train_acc, label="Train Accuracy")
    plt.plot(epochs, history.val_acc, label="Val Accuracy")
    plt.title("Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def predict_single_image(
    model: nn.Module,
    image_path: Path,
    class_names: List[str],
    image_size: int,
    device: torch.device,
) -> Tuple[str, float]:
    """Predict class label and confidence score for one image.

    Args:
        model: Trained model in evaluation mode.
        image_path: Path to input image.
        class_names: Ordered class names matching model outputs.
        image_size: Target size used during model training.
        device: Compute device.

    Returns:
        A tuple containing:
        - Predicted class label.
        - Confidence score as a percentage.
    """
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
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probabilities, dim=1)

    predicted_label = class_names[pred_idx.item()]
    confidence_pct = confidence.item() * 100.0
    return predicted_label, confidence_pct


def main() -> None:
    """Run end-to-end training, evaluation, plotting, and optional inference."""
    cfg = CONFIG
    set_seed(cfg.seed)
    resolved_version, save_model_path, save_plot_path = resolve_output_paths(cfg)

    print(f"Resolved model version: {resolved_version}")
    print(f"Model output path: {save_model_path}")
    print(f"Plot output path: {save_plot_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = create_dataloaders(
        dataset_dir=cfg.dataset_dir,
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
    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
    )

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=cfg.epochs,
        patience=cfg.patience,
    )

    final_train_loss, final_train_acc = evaluate(model, train_loader, criterion, device)
    final_val_loss, final_val_acc = evaluate(model, val_loader, criterion, device)

    print("\nFinal Evaluation Metrics")
    print(f"Train Loss: {final_train_loss:.4f}")
    print(f"Train Accuracy: {final_train_acc:.2f}%")
    print(f"Validation Loss: {final_val_loss:.4f}")
    print(f"Validation Accuracy: {final_val_acc:.2f}%")

    save_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": cfg.image_size,
            "model_name": cfg.model_name,
            "model_version": resolved_version,
        },
        save_model_path,
    )
    print(f"Saved model weights to: {save_model_path}")

    plot_learning_curves(history=history, save_path=save_plot_path)
    print(f"Saved learning curves to: {save_plot_path}")

    if cfg.predict_image is not None:
        label, confidence = predict_single_image(
            model=model,
            image_path=cfg.predict_image,
            class_names=class_names,
            image_size=cfg.image_size,
            device=device,
        )
        print(
            f"Prediction for '{cfg.predict_image}': "
            f"{label} (confidence: {confidence:.2f}%)"
        )


if __name__ == "__main__":
    main()
