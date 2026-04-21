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
    process_prediction,
    resolve_dataset_root,
    set_seed,
)


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
