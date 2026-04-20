"""Task 2 prediction script: run inference and quality grading.

Loads a trained multitask EfficientNetV2-S checkpoint and performs
single-image prediction with quality scoring, grading, and inventory
action recommendations.

Usage:
    python task2_predict.py                          # random dataset image
    python task2_predict.py --image path/to/image.jpg  # specific image
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from task2_model import (
    CONFIG,
    _extract_checkpoint_state_dict,
    _list_dataset_images,
    build_model,
    clamp,
    process_prediction,
    resolve_dataset_root,
    set_seed,
)


# ---------------------------------------------------------------------------
# Single-image prediction
# ---------------------------------------------------------------------------

def _compute_quality_from_image(image: Image.Image) -> Dict[str, float]:
    """Compute quality scores directly from image pixels (improved formula).

    Uses HSV colour analysis with a balanced formula that works well
    for both vibrant and pale produce. This bypasses the model's quality
    head, avoiding the need for retraining when the formula changes.
    """
    import numpy as np

    rgb = image.convert("RGB").resize((256, 256))
    hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255.0
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    mask = ((sat > 0.12) | (val < 0.92)) & (val > 0.08)
    if float(np.mean(mask)) < 0.03:
        mask = np.ones_like(mask, dtype=bool)

    sat_pixels, val_pixels = sat[mask], val[mask]
    mean_sat = float(np.mean(sat_pixels)) if sat_pixels.size else 0.0
    mean_val = float(np.mean(val_pixels)) if val_pixels.size else 0.0
    val_std = float(np.std(val_pixels)) if val_pixels.size else 0.0

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
    freshness_score = 100.0 * (1.0 - brown_ratio)
    colour = clamp(0.3 * saturation_score + 0.3 * brightness_score + 0.4 * freshness_score)

    size = clamp(
        0.8 * (100.0 * clamp((area_ratio - 0.08) / 0.52, 0.0, 1.0))
        + 0.2 * (100.0 * clamp(fill_ratio, 0.0, 1.0))
    )

    uniformity_score = 100.0 * clamp(1.0 - min(val_std / 0.25, 1.0), 0.0, 1.0)
    ripeness = clamp(
        0.45 * colour + 0.30 * uniformity_score + 0.25 * freshness_score
    )

    return {
        "colour": round(colour, 2),
        "size": round(size, 2),
        "ripeness": round(ripeness, 2),
    }


def predict_single_image(
    model: nn.Module, image_path: Path, class_names: List[str],
    image_size: int, device: torch.device,
) -> Tuple[str, float, Dict[str, float]]:
    """Predict class label via CNN, quality scores via direct pixel analysis."""
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits, _ = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    label = class_names[pred_idx.item()]
    quality_scores = _compute_quality_from_image(image)
    return label, confidence.item() * 100.0, quality_scores


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run single-image prediction with quality grading."""
    parser = argparse.ArgumentParser(description="Task 2 — Predict produce quality")
    parser.add_argument("--image", type=Path, default=None,
                        help="Path to image file (default: random from dataset)")
    parser.add_argument("--model", type=Path, default=None,
                        help="Path to model checkpoint (default: from CONFIG)")
    args = parser.parse_args()

    cfg = CONFIG
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = args.model or cfg.save_model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{model_path}'. "
            "Run task2_train.py first to train the model."
        )

    # Load checkpoint to get class names and image size.
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get("class_names")
    image_size = checkpoint.get("image_size", cfg.image_size)

    if class_names is None:
        raise ValueError("Checkpoint does not contain 'class_names'. Retrain with task2_train.py.")

    # Build model and load weights.
    model = build_model(
        num_classes=len(class_names), device=device,
        use_pretrained=False,  # weights come from checkpoint
    )
    state_dict = _extract_checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded model from: {model_path}")
    print(f"Classes ({len(class_names)}): {class_names}")

    # Resolve prediction image.
    prediction_image = args.image
    if prediction_image is None:
        dataset_root = resolve_dataset_root(cfg.dataset_dir)
        if dataset_root is None:
            raise FileNotFoundError(
                f"Dataset not found at '{cfg.dataset_dir}' and no --image provided."
            )
        dataset_images = _list_dataset_images(dataset_root)
        if not dataset_images:
            raise FileNotFoundError(f"No image files found in: {dataset_root}")
        prediction_image = random.SystemRandom().choice(dataset_images)
        print(f"Selected random dataset image: {prediction_image}")

    if not prediction_image.exists() or not prediction_image.is_file():
        raise FileNotFoundError(f"Prediction image not found: {prediction_image}")

    # Run prediction.
    label, confidence_pct, quality_scores = predict_single_image(
        model, prediction_image, class_names, image_size, device,
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
