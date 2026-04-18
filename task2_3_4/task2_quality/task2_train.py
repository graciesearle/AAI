"""Task 2 training script: train multitask EfficientNetV2-S CNN.

Trains a multitask transfer-learning model for Fresh/Rotten classification
and Colour/Size/Ripeness quality regression. Saves model checkpoint and
learning curves on completion.

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
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run end-to-end multitask training and evaluation."""
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

    if cfg.save_model_path.exists():
        checkpoint = torch.load(cfg.save_model_path, map_location=device)
        state_dict = _extract_checkpoint_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded existing model from: {cfg.save_model_path}")
        print("Skipping training — delete checkpoint to retrain.")
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

        # Final evaluation
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


if __name__ == "__main__":
    main()
