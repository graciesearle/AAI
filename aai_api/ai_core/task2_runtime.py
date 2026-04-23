"""
task2_runtime.py — Task 2 quality inference bridge.

Delegates model construction and post-processing to task2_model. 
Handles BinaryIO file uploads from the API directly using PIL.Image.open().
"""
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

from task2_3_4.task2_quality.task2_model import (
    build_model,                     
    _extract_checkpoint_state_dict,  
    process_prediction,              
    normalize_label
)

def evaluate_quality_with_clip(image: Image.Image, predicted_class: str, clip_model, clip_processor, device) -> dict:
    """Zero-shot CLIP grading using the calibrated 3-tier prompt system."""
    base_item = predicted_class.split("_")[0].lower()

    specific_prompts = {
        "banana": {
            "ripeness": [
                ("a photo of a perfectly ripe and fresh banana", 100),
                ("a photo of a standard, edible banana", 90),
                ("a photo of an unripe or overripe, mealy banana", 30)
            ],
            "colour": [
                ("a banana with vibrant, bright, and uniform colour", 100),
                ("a banana with natural, standard colour and minor blemishes", 90),
                ("a banana with dull, uneven, or significantly discoloured skin", 20)
            ]
        },
        "apple": {
            "ripeness": [
                ("a photo of a perfectly ripe and fresh apple", 100),
                ("a photo of a standard, edible apple", 90),
                ("a photo of an unripe or overripe, mealy apple", 30)
            ],
            "colour": [
                ("an apple with vibrant, bright, and uniform colour", 100),
                ("an apple with natural, standard colour and minor blemishes", 90),
                ("an apple with dull, uneven, or significantly discoloured skin", 20)
            ]
        }
    }

    fallback_prompts = {
        "ripeness": [
            (f"a photo of a perfectly ripe and fresh {base_item}", 100),
            (f"a photo of a standard, edible {base_item}", 90),
            (f"a photo of an unripe or overripe, mealy {base_item}", 30)
        ],
        "colour": [
            (f"a {base_item} with vibrant, bright, and uniform colour", 100),
            (f"a {base_item} with natural, standard colour and minor blemishes", 90),
            (f"a {base_item} with dull, uneven, or significantly discoloured skin", 20)
        ]
    }

    shape_prompts = [
        (f"a perfectly symmetrical, retail-standard {base_item}", 100),
        (f"a normal, whole, and intact {base_item} with minor organic variations", 90),
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


def run_quality_inference(*, image_file, checkpoint_path: str) -> dict:
    """Run chained EfficientNetV2-S and CLIP inference on an uploaded image stream."""
    print("[AAI PIPELINE LOG] Starting quality inference for Task 2...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the EfficientNet Gatekeeper
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names")
    image_size = checkpoint.get("image_size", 224)
    
    if not class_names:
        raise ValueError("Checkpoint is missing 'class_names'.")

    model = build_model(num_classes=len(class_names), device=device, use_pretrained=False)
    model.load_state_dict(_extract_checkpoint_state_dict(checkpoint), strict=False)
    model.eval()

    # 2. Load the CLIP Model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    # 3. Read and Process the Image stream
    image_file.seek(0)
    raw_pil = Image.open(image_file).convert("RGB")
    
    # Denoise for the Grader (Strips salt-and-pepper artifacts)
    cv_img = cv2.cvtColor(np.array(raw_pil), cv2.COLOR_RGB2BGR)
    denoised_cv = cv2.medianBlur(cv_img, 3)
    clean_pil = Image.fromarray(cv2.cvtColor(denoised_cv, cv2.COLOR_BGR2RGB))

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Gatekeeper uses the raw image
    tensor = preprocess(raw_pil).unsqueeze(0).to(device)

    # 4. Gatekeeper Inference
    with torch.no_grad():
        logits = model(tensor) 
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    label = class_names[pred_idx.item()]
    normalized_label = normalize_label(label)
    confidence_pct = confidence.item() * 100.0

    # 5. Conditional Routing
    if normalized_label == "rotten":
        quality_scores = {
            "colour": {"score": 0.0, "justification": None},
            "shape": {"score": 0.0, "justification": None},
            "ripeness": {"score": 0.0, "justification": None},
        }
    else:
        # TTA (Test-Time Augmentation): Average scores across original & flipped views
        views = [clean_pil, clean_pil.transpose(Image.FLIP_LEFT_RIGHT)]
        all_scores = []
        for view in views:
            scores = evaluate_quality_with_clip(view, label, clip_model, clip_processor, device)
            all_scores.append(scores)

        quality_scores = {}
        for metric in ["colour", "shape", "ripeness"]:
            avg_score = sum(s[metric]["score"] for s in all_scores) / len(all_scores)
            quality_scores[metric] = {
                "score": round(avg_score, 2),
                "justification": all_scores[0][metric]["justification"]
            }

    # 6. Post-process and package payload
    result = process_prediction(
        label=label,
        confidence=confidence_pct / 100.0,
        quality_scores=quality_scores,
    )

    # 7. Generate XAI Derivations for Academic Rubric Compliance
    grade = result["overall_grade"]
    
    if grade == "C":
        if normalized_label == "rotten":
            grade_derivation = "Assigned Grade C because the classification model detected severe defects/rot."
        else:
            grade_derivation = "Assigned Grade C because one or more quality metrics fell below threshold."
        rec_derivation = "Grade C indicates immediate risk of spoilage. Fast sale or heavy markdown recommended."
    elif grade == "B":
        grade_derivation = "Assigned Grade B because one or more quality metrics fell below the top-tier thresholds but remained above critical levels."
        rec_derivation = "Grade B stock requires moderate markdowns or close monitoring depending on confidence."
    else:
        grade_derivation = "Assigned Grade A because all metrics exceeded high-quality thresholds."
        rec_derivation = "Grade A stock warrants keeping standard retail pricing."
    
    explanation_payload = {
        "model_artifact": str(checkpoint_path),
        "architecture": "efficientnetv2_s_multitask + clip_zero_shot",
        "score_breakdown": quality_scores,
        "grade_derivation": grade_derivation,
        "recommendation_derivation": rec_derivation,
        "model_confidence_percentage": round(confidence_pct, 2),
    }
    
    result["explanation_payload"] = explanation_payload
    return result