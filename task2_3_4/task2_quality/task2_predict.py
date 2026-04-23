"""Task 2 prediction script: run chained gatekeeper and zero-shot grading.

Loads a trained EfficientNetV2-S gatekeeper checkpoint for Fresh/Rotten
classification. Fresh items are then routed to CLIP zero-shot quality grading
for Colour/Shape/Ripeness with prompt-based justifications.

Usage:
    python task2_predict.py
    python task2_predict.py --image path/to/image.jpg
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
import cv2
import numpy as np

from task2_model import (
    CONFIG,
    _extract_checkpoint_state_dict,
    _list_dataset_images,
    build_model,
    normalize_label,
    process_prediction,
    resolve_dataset_root,
    set_seed,
)


def evaluate_quality_with_clip(image: Image.Image, predicted_class: str, clip_model, clip_processor, device) -> dict:
    base_item = predicted_class.split("_")[0].lower()

    specific_prompts = {
        "banana": {
            "ripeness": [
                ("a photo of a perfectly ripe and fresh banana", 100),
                ("a photo of a standard, edible banana", 75),
                ("a photo of an unripe or overripe, mealy banana", 30)
            ],
            "colour": [
                ("a banana with vibrant, bright, and uniform colour", 100),
                ("a banana with natural, standard colour and minor blemishes", 75),
                ("a banana with dull, uneven, or significantly discoloured skin", 20)
            ]
        },
        "apple": {
            "ripeness": [
                ("a photo of a perfectly ripe and fresh apple", 100),
                ("a photo of a standard, edible apple", 75),
                ("a photo of an unripe or overripe, mealy apple", 30)
            ],
            "colour": [
                ("an apple with vibrant, bright, and uniform colour", 100),
                ("an apple with natural, standard colour and minor blemishes", 75),
                ("an apple with dull, uneven, or significantly discoloured skin", 20)
            ]
        }
    }

    fallback_prompts = {
        "ripeness": [
            (f"a photo of a perfectly ripe and fresh {base_item}", 100),
            (f"a photo of a standard, edible {base_item}", 75),
            (f"a photo of an unripe or overripe, mealy {base_item}", 30)
        ],
        "colour": [
            (f"a {base_item} with vibrant, bright, and uniform colour", 100),
            (f"a {base_item} with natural, standard colour and minor blemishes", 75),
            (f"a {base_item} with dull, uneven, or significantly discoloured skin", 20)
        ]
    }

    shape_prompts = [
        (f"a perfectly symmetrical, retail-standard {base_item}", 100),
        (f"a normal, whole, and intact {base_item} with minor organic variations", 75),
        (f"a severely deformed, broken, or crushed {base_item}", 20)
    ]
    
    item_prompts = specific_prompts.get(base_item, fallback_prompts)

    def get_weighted_score(prompt_data):
        texts = [p[0] for p in prompt_data]
        weights = [p[1] for p in prompt_data]

        inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1).squeeze().cpu().tolist()
        final_score = sum(prob * weight for prob, weight in zip(probs, weights))
        max_prob_idx = probs.index(max(probs))
        winning_prompt = texts[max_prob_idx]

        return {"score": round(final_score, 2), "justification": winning_prompt}

    return {
        "colour": get_weighted_score(item_prompts["colour"]),
        "shape": get_weighted_score(shape_prompts),
        "ripeness": get_weighted_score(item_prompts["ripeness"])
    }


# ---------------------------------------------------------------------------
# Single-image prediction
# ---------------------------------------------------------------------------



def predict_single_image(
    model: nn.Module,
    image_path: Path,
    class_names: List[str],
    image_size: int,
    device: torch.device,
    clip_model,
    clip_processor,
) -> Tuple[str, float, Dict[str, Dict[str, object]]]:
    """Predict with Denoising and Multi-View Averaging for maximum accuracy."""
    
    # 1. Load and Denoise
    raw_pil = Image.open(image_path).convert("RGB")
    cv_img = cv2.cvtColor(np.array(raw_pil), cv2.COLOR_RGB2BGR)
    # Median filter is the gold standard for salt-and-pepper noise
    denoised_cv = cv2.medianBlur(cv_img, 3) 
    clean_pil = Image.fromarray(cv2.cvtColor(denoised_cv, cv2.COLOR_BGR2RGB))

    # 2. Gatekeeper Logic (EfficientNet)
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = preprocess(raw_pil).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    label = class_names[pred_idx.item()]
    normalized_label = normalize_label(label)
    confidence_pct = confidence.item() * 100.0

    if normalized_label == "rotten":
        return label, confidence_pct, {
            "colour": {"score": 0.0, "justification": None},
            "shape": {"score": 0.0, "justification": None},
            "ripeness": {"score": 0.0, "justification": None},
        }

    # 3. Multi-View CLIP Grading (TTA)
    # We evaluate the clean image and a horizontal flip to average out noise bias
    views = [clean_pil, clean_pil.transpose(Image.FLIP_LEFT_RIGHT)]
    
    all_scores = []
    for view in views:
        scores = evaluate_quality_with_clip(
            image=view,
            predicted_class=label,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=device,
        )
        all_scores.append(scores)

    # Image Output for Debugging (optional)
    raw_pil = Image.open(image_path).convert("RGB")
    cv_img = cv2.cvtColor(np.array(raw_pil), cv2.COLOR_RGB2BGR)
    denoised_cv = cv2.medianBlur(cv_img, 3) 
    clean_pil = Image.fromarray(cv2.cvtColor(denoised_cv, cv2.COLOR_BGR2RGB))

    # --- ADD THIS DEBUG BLOCK ---
    debug_dir = Path("debug_output")
    debug_dir.mkdir(exist_ok=True)
    
    # Save a comparison image
    comparison = np.hstack((np.array(raw_pil), np.array(clean_pil)))
    Image.fromarray(comparison).save(debug_dir / f"debug_{image_path.name}")
    # -----------------------------

    # Average the scores across both views
    final_quality = {}
    for metric in ["colour", "shape", "ripeness"]:
        avg_score = sum(s[metric]["score"] for s in all_scores) / len(all_scores)
        # Use the justification from the first (standard) view
        final_quality[metric] = {
            "score": round(avg_score, 2),
            "justification": all_scores[0][metric]["justification"]
        }

    return label, confidence_pct, final_quality


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _normalize_cli_args(argv: List[str]) -> List[str]:
    """Expand shorthand run-count arguments, e.g. '-10' to '--runs 10'."""
    normalized: List[str] = []
    for arg in argv:
        match = re.fullmatch(r"-(\d+)", arg)
        if match:
            normalized.extend(["--runs", match.group(1)])
        else:
            normalized.append(arg)
    return normalized


def main() -> None:
    """Run single-image chained prediction with quality grading."""
    parser = argparse.ArgumentParser(description="Task 2 - Predict produce quality")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to image file (default: random from dataset)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to model checkpoint (default: from CONFIG)",
    )
    parser.add_argument(
        "-n",
        "--runs",
        type=int,
        default=1,
        help="Number of prediction runs to execute (supports shorthand like -10).",
    )
    args = parser.parse_args(_normalize_cli_args(sys.argv[1:]))

    if args.runs < 1:
        raise ValueError("--runs must be a positive integer.")

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

    # Build gatekeeper model and load weights.
    model = build_model(
        num_classes=len(class_names),
        device=device,
        use_pretrained=False,
    )
    state_dict = _extract_checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded gatekeeper model from: {model_path}")
    print(f"Classes ({len(class_names)}): {class_names}")

    # Load CLIP components after loading the gatekeeper model.
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    print("Loaded CLIP model: openai/clip-vit-base-patch32")

    # Resolve prediction image(s).
    if args.image is not None:
        if not args.image.exists() or not args.image.is_file():
            raise FileNotFoundError(f"Prediction image not found: {args.image}")
        prediction_images = [args.image for _ in range(args.runs)]
    else:
        dataset_root = resolve_dataset_root(cfg.dataset_dir)
        if dataset_root is None:
            raise FileNotFoundError(
                f"Dataset not found at '{cfg.dataset_dir}' and no --image provided."
            )
        dataset_images = _list_dataset_images(dataset_root)
        if not dataset_images:
            raise FileNotFoundError(f"No image files found in: {dataset_root}")
        rng = random.SystemRandom()
        prediction_images = [rng.choice(dataset_images) for _ in range(args.runs)]

    # Run chained prediction(s).
    for run_idx, prediction_image in enumerate(prediction_images, start=1):
        label, confidence_pct, quality_scores = predict_single_image(
            model=model,
            image_path=prediction_image,
            class_names=class_names,
            image_size=image_size,
            device=device,
            clip_model=clip_model,
            clip_processor=clip_processor,
        )
        result = process_prediction(
            label=label,
            confidence=confidence_pct / 100.0,
            quality_scores=quality_scores,
        )

        if args.runs > 1:
            print(f"\nRun [{run_idx}/{args.runs}]")
        print(
            f"Prediction for '{prediction_image.name}': {label} "
            f"(confidence: {confidence_pct:.2f}%)"
        )
        print(f"Quality Scores: {result['quality_scores']}")
        print(f"Overall Grade: {result['overall_grade']}")
        print(f"Inventory Action: {result['inventory_action']}")


if __name__ == "__main__":
    main()
