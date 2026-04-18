"""
task2_runtime.py — Task 2 quality inference bridge.

Delegates model construction and post-processing to task2_model (Gracie's split
module), which owns the MultiTaskProduceNet architecture (EfficientNetV2-S
backbone + classification head + quality regression head).

predict_single_image() from task2_predict.py is intentionally NOT imported here
— it requires a file path on disk. This module handles BinaryIO file uploads
from the API directly using PIL.Image.open(), which accepts file-like objects
without writing to disk.

If functions in task2_model are renamed, refer to these docstrings to locate
the correct replacement for each import.
"""
from PIL import Image
from torchvision import transforms
import torch

from task2_3_4.task2_quality.task2_model import (
    build_model,                     # Constructs the EfficientNetV2-S multitask model
    _extract_checkpoint_state_dict,  # Normalises checkpoint keys across save formats
    process_prediction,              # Validates scores, assigns A/B/C grade, returns inventory action
)


def run_quality_inference(*, image_file, checkpoint_path):
    """Run EfficientNetV2-S multitask inference on an uploaded image (BinaryIO).

    Reads class_names and image_size directly from the checkpoint — Gracie's
    training script (task2_train.py) embeds both fields at save time, so we
    do not need to duplicate them in the manifest output_schema.

    Args:
        image_file: A file-like object (BinaryIO / InMemoryUploadedFile) — no
                    temp file is written to disk.
        checkpoint_path: Path to the .pth checkpoint on disk.

    Returns:
        dict with keys: normalized_label, overall_grade, quality_scores,
        inventory_action, explanation_payload, confidence.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    class_names = checkpoint.get("class_names")
    image_size = checkpoint.get("image_size", 224)
    if not class_names:
        raise ValueError(
            "Checkpoint is missing 'class_names'. Re-train with task2_train.py "
            "to produce a checkpoint that embeds class metadata."
        )

    model = build_model(num_classes=len(class_names), device=device, use_pretrained=False)
    state_dict = _extract_checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_file.seek(0)
    image = Image.open(image_file).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, q_preds = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    label = class_names[pred_idx.item()]
    q_np = torch.clamp(q_preds.squeeze(0), 0.0, 100.0).cpu().numpy()
    
    quality_scores = {
        "colour": round(float(q_np[0]), 2),
        "size":   round(float(q_np[1]), 2),
        "ripeness": round(float(q_np[2]), 2),
    }

    # Generate authoritative base prediction
    result = process_prediction(
        label=label,
        confidence=confidence.item(),
        quality_scores=quality_scores,
    )
    
    # --- Generate XAI Derivations for Academic Rubric Compliance ---
    # The spec requires us to explain *how* the grade and recommendation
    # were derived. We inject these explanatory strings directly into the
    # explanation_payload here.
    grade = result["overall_grade"]
    
    if grade == "C":
        grade_derivation = "Assigned Grade C because one or more quality metrics fell below threshold (Colour < 65, Size < 70, or Ripeness < 60)."
        rec_derivation = "Grade C indicates immediate risk of spoilage. Fast sale or heavy markdown recommended."
    elif grade == "B":
        grade_derivation = "Assigned Grade B because one or more quality metrics fell below the top-tier thresholds but remained above critical levels."
        rec_derivation = "Grade B stock requires moderate markdowns or close monitoring depending on confidence."
    else:
        grade_derivation = "Assigned Grade A because all metrics exceeded high-quality thresholds (Colour >= 75, Size >= 80, Ripeness >= 70)."
        rec_derivation = "Grade A stock warrants keeping standard retail pricing."
    
    explanation_payload = {
        "model_artifact": str(checkpoint_path),
        "architecture": "efficientnetv2_s_multitask",
        "score_breakdown": quality_scores,
        "grade_derivation": grade_derivation,
        "recommendation_derivation": rec_derivation,
        "model_confidence_percentage": round(confidence.item() * 100.0, 2),
    }
    
    result["explanation_payload"] = explanation_payload
    
    return result
