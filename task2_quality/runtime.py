from __future__ import annotations

from pathlib import Path
from typing import Any

from task2_quality.model_inference import infer_binary_from_artifact
from task2_quality.postprocess import process_prediction


def run_quality_inference(
    *,
    image_file,
    model_root: Path,
    model_name: str,
    model_version: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if manifest.get("task_profile") != "task2_quality":
        raise ValueError("Selected model bundle is not task2_quality profile")

    model_output = infer_binary_from_artifact(
        image_file=image_file,
        model_root=model_root,
        model_name=model_name,
        model_version=model_version,
        manifest=manifest,
    )
    predicted_class = model_output["predicted_class"]
    confidence_01 = model_output["confidence_01"]
    class_probabilities = model_output["class_probabilities"]
    explanation_payload = {
        "note": "trained-model-inference-v1",
        "model_name": model_name,
        "model_version": model_version,
        "artifact_path": model_output["artifact_path"],
        "raw_class_probabilities": model_output.get("raw_class_probabilities", {}),
    }

    graded = process_prediction(label=predicted_class.title(), confidence=confidence_01)
    quality_scores = graded["quality_scores"]

    return {
        "color_score": float(quality_scores["colour"]),
        "size_score": float(quality_scores["size"]),
        "ripeness_score": float(quality_scores["ripeness"]),
        "confidence": round(confidence_01 * 100.0, 2),
        "predicted_class": predicted_class,
        "overall_grade": graded["overall_grade"],
        "class_probabilities": class_probabilities,
        "explanation_payload": explanation_payload,
        "transparency_refs": [f"model://{model_name}/{model_version}"],
        "model_version_used": model_version,
    }
