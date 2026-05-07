"""HITL fine-tune for Task 2 regression head using override exports.

This script consumes the DESD retraining export (ZIP or extracted folder)
with metadata.csv and images/, then fine-tunes only the quality head of
EfficientNetV2-S using human overrides.
"""

from __future__ import annotations

import argparse
import copy
import csv
import random
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset

from task2_model import (
    _extract_checkpoint_state_dict,
    build_model,
    build_transforms,
    clamp,
    set_seed,
)


@dataclass(frozen=True)
class OverrideSample:
    image_path: Path
    targets: Tuple[float, float, float]


class OverrideDataset(Dataset):
    def __init__(self, samples: Sequence[OverrideSample], transform=None) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        targets = torch.tensor(sample.targets, dtype=torch.float32)
        return image, targets


def _parse_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    return None


def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _resolve_score(
    *,
    override_value: Optional[float],
    accepted_flag: Optional[bool],
    base_value: Optional[float],
    accepted_fallback: Optional[bool],
) -> Optional[float]:
    if override_value is not None:
        return override_value
    accepted = accepted_flag if accepted_flag is not None else accepted_fallback
    if accepted is True and base_value is not None:
        return base_value
    return None


def _build_samples(metadata_path: Path, images_root: Path) -> List[OverrideSample]:
    samples: List[OverrideSample] = []
    missing_images = 0
    missing_labels = 0

    with metadata_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_name = (row.get("image_filename") or "").strip()
            if not image_name:
                missing_images += 1
                continue
            image_path = images_root / image_name
            if not image_path.exists():
                missing_images += 1
                continue

            base_color = _parse_float(row.get("color_score"))
            base_size = _parse_float(row.get("size_score"))
            base_ripeness = _parse_float(row.get("ripeness_score"))

            override_color = _parse_float(row.get("override_color_score"))
            override_size = _parse_float(row.get("override_size_score"))
            override_ripeness = _parse_float(row.get("override_ripeness_score"))

            accepted_fallback = _parse_bool(row.get("accepted_recommendation"))
            color_accepted = _parse_bool(row.get("color_accepted"))
            size_accepted = _parse_bool(row.get("size_accepted"))
            ripeness_accepted = _parse_bool(row.get("ripeness_accepted"))

            color_score = _resolve_score(
                override_value=override_color,
                accepted_flag=color_accepted,
                base_value=base_color,
                accepted_fallback=accepted_fallback,
            )
            size_score = _resolve_score(
                override_value=override_size,
                accepted_flag=size_accepted,
                base_value=base_size,
                accepted_fallback=accepted_fallback,
            )
            ripeness_score = _resolve_score(
                override_value=override_ripeness,
                accepted_flag=ripeness_accepted,
                base_value=base_ripeness,
                accepted_fallback=accepted_fallback,
            )

            if color_score is None or size_score is None or ripeness_score is None:
                missing_labels += 1
                continue

            samples.append(
                OverrideSample(
                    image_path=image_path,
                    targets=(
                        clamp(color_score, 0.0, 100.0),
                        clamp(size_score, 0.0, 100.0),
                        clamp(ripeness_score, 0.0, 100.0),
                    ),
                )
            )

    if missing_images:
        print(f"Skipped {missing_images} row(s) with missing images.")
    if missing_labels:
        print(f"Skipped {missing_labels} row(s) without usable override labels.")

    return samples


def _split_indices(count: int, train_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split_point = max(1, int(count * train_ratio))
    if split_point >= count:
        split_point = count - 1
    return indices[:split_point], indices[split_point:]


def _freeze_for_quality_finetune(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.quality_head.parameters():
        param.requires_grad = True


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Tuple[float, float]:
    running_loss = 0.0
    running_mae = 0.0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        _, preds = model(images)
        loss = criterion(preds, targets)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_mae += torch.mean(torch.abs(preds - targets)).item() * batch_size
        total += batch_size

    total = max(total, 1)
    return running_loss / total, running_mae / total


def fine_tune_quality_head(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
) -> nn.Module:
    criterion = nn.SmoothL1Loss()
    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_weights = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_mae = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        model.eval()
        with torch.no_grad():
            val_loss, val_mae = _run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
            )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} MAE: {train_mae:.3f} | "
            f"Val Loss: {val_loss:.4f} MAE: {val_mae:.3f}"
        )

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_weights)
    return model


def _resolve_export_root(export_zip: Optional[Path], export_dir: Optional[Path]):
    if export_zip:
        temp_dir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(export_zip, "r") as zip_handle:
            zip_handle.extractall(temp_dir.name)
        return Path(temp_dir.name), temp_dir
    if export_dir:
        return export_dir, None
    raise ValueError("Either --export-zip or --export-dir is required.")


def main() -> None:
    parser = argparse.ArgumentParser(description="HITL fine-tune for Task 2 quality head")
    parser.add_argument("--export-zip", type=Path, default=None, help="Path to DESD retraining export ZIP")
    parser.add_argument("--export-dir", type=Path, default=None, help="Path to extracted export folder")
    parser.add_argument("--images-dir", type=Path, default=None, help="Optional override for images directory")
    parser.add_argument("--base-model", type=Path, required=True, help="Path to base Task 2 checkpoint")
    parser.add_argument("--output-model", type=Path, required=True, help="Path to save fine-tuned checkpoint")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    export_root, temp_dir = _resolve_export_root(args.export_zip, args.export_dir)
    try:
        metadata_path = export_root / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.csv not found in {export_root}")

        images_root = args.images_dir or (export_root / "images")
        if not images_root.exists():
            raise FileNotFoundError(f"Images directory not found: {images_root}")

        if not args.base_model.exists():
            raise FileNotFoundError(f"Base model checkpoint not found: {args.base_model}")

        samples = _build_samples(metadata_path, images_root)
        if not samples:
            raise ValueError("No usable override samples found in export.")

        train_idx, val_idx = _split_indices(len(samples), args.train_ratio, args.seed)
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        print(f"Training samples: {len(train_samples)} | Validation samples: {len(val_samples)}")

        checkpoint = torch.load(args.base_model, map_location=device)
        class_names = checkpoint.get("class_names", ["Fresh", "Rotten"])
        image_size = checkpoint.get("image_size", 224)

        model = build_model(
            num_classes=len(class_names),
            device=device,
            use_pretrained=False,
        )
        state_dict = _extract_checkpoint_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=False)

        _freeze_for_quality_finetune(model)

        train_tf, val_tf = build_transforms(image_size=image_size)
        train_ds = OverrideDataset(train_samples, transform=train_tf)
        val_ds = OverrideDataset(val_samples, transform=val_tf)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model = fine_tune_quality_head(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            min_delta=args.min_delta,
        )

        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": image_size,
                "architecture": "efficientnetv2s_multitask_hitl_quality_head",
                "hitl_finetune": {
                    "export_source": str(args.export_zip or args.export_dir),
                    "train_samples": len(train_samples),
                    "val_samples": len(val_samples),
                },
            },
            args.output_model,
        )
        print(f"Saved fine-tuned model to: {args.output_model}")
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
