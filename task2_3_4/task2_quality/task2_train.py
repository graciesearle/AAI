"""Task 2 training script: train EfficientNetV2-S gatekeeper classifier.

Trains a single-task transfer-learning model for Fresh/Rotten classification.
Saves model checkpoint and learning curves on completion.

Usage:
    python task2_train.py
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from task2_model import (
    CONFIG,
    History,
    _extract_checkpoint_state_dict,
    build_model,
    create_dataloaders,
    resolve_dataset_root,
    set_seed,
)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cls_criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Run single-task classification evaluation on a data loader."""
    model.eval()
    total_loss_sum = cls_loss_sum = 0.0
    correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            cls_loss = cls_criterion(logits, labels)
            total_loss = cls_loss

            batch_size = labels.size(0)
            total_loss_sum += total_loss.item() * batch_size
            cls_loss_sum += cls_loss.item() * batch_size
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total += batch_size

    n = max(total, 1)
    return total_loss_sum / n, 100.0 * correct / n, cls_loss_sum / n


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cls_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    num_epochs: int,
    patience: int,
    early_stop_min_delta: float,
) -> Tuple[nn.Module, History]:
    """Train single-task model with early stopping on validation cls loss."""
    history = History([], [], [], [], [], [])
    best_wts = copy.deepcopy(model.state_dict())
    best_val_cls_loss = float("inf")
    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_total_sum = train_cls_sum = 0.0
        correct = total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            cls_loss = cls_criterion(logits, labels)
            loss = cls_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = labels.size(0)
            train_total_sum += loss.item() * batch_size
            train_cls_sum += cls_loss.item() * batch_size
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total += batch_size

        n = max(total, 1)
        train_total = train_total_sum / n
        train_cls = train_cls_sum / n
        train_acc = 100.0 * correct / n

        val_total, val_acc, val_cls = evaluate(
            model,
            val_loader,
            cls_criterion,
            device,
        )

        history.train_total_loss.append(train_total)
        history.val_total_loss.append(val_total)
        history.train_cls_loss.append(train_cls)
        history.val_cls_loss.append(val_cls)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Total: {train_total:.4f} | Train Cls: {train_cls:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Total: {val_total:.4f} | Val Cls: {val_cls:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        improved = val_cls < (best_val_cls_loss - early_stop_min_delta)
        tie_break = abs(val_cls - best_val_cls_loss) <= early_stop_min_delta and val_acc > best_val_acc

        if improved or tie_break:
            best_val_cls_loss, best_val_acc = val_cls, val_acc
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
    """Plot total loss and classification accuracy curves."""
    epochs = range(1, len(history.train_total_loss) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.train_total_loss, label="Train Total")
    plt.plot(epochs, history.val_total_loss, label="Val Total")
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run end-to-end gatekeeper training and evaluation."""
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

    # Differential learning rates: backbone fine-tuned layers get lower LR.
    backbone_params = [
        p for name, p in model.named_parameters() if p.requires_grad and "backbone" in name
    ]
    head_params = [
        p for name, p in model.named_parameters() if p.requires_grad and "backbone" not in name
    ]
    optimizer = Adam(
        [
            {"params": backbone_params, "lr": cfg.learning_rate * 0.01},
            {"params": head_params, "lr": cfg.learning_rate},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=1e-6,
    )

    if cfg.save_model_path.exists():
        checkpoint = torch.load(cfg.save_model_path, map_location=device)
        state_dict = _extract_checkpoint_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded existing model from: {cfg.save_model_path}")
        print("Skipping training - delete checkpoint to retrain.")
    else:
        print("No existing model checkpoint found. Training from scratch.")
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cls_criterion=cls_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=cfg.epochs,
            patience=cfg.patience,
            early_stop_min_delta=cfg.early_stop_min_delta,
        )

        # Final evaluation
        train_total, train_acc, train_cls = evaluate(
            model,
            train_loader,
            cls_criterion,
            device,
        )
        val_total, val_acc, val_cls = evaluate(
            model,
            val_loader,
            cls_criterion,
            device,
        )
        print("\nFinal Evaluation Metrics")
        print(
            f"Train Total Loss: {train_total:.4f} | Cls: {train_cls:.4f} | "
            f"Accuracy: {train_acc:.2f}%"
        )
        print(
            f"Val   Total Loss: {val_total:.4f} | Cls: {val_cls:.4f} | "
            f"Accuracy: {val_acc:.2f}%"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": cfg.image_size,
                "architecture": "efficientnetv2s_gatekeeper_transfer_learning",
            },
            cfg.save_model_path,
        )
        print(f"Saved model weights to: {cfg.save_model_path}")

        plot_learning_curves(history=history, save_path=cfg.save_plot_path)
        print(f"Saved learning curves to: {cfg.save_plot_path}")


if __name__ == "__main__":
    main()
