from __future__ import annotations

from pathlib import Path
from typing import Any

from task2_3_4.task2_quality.model_inference import infer_binary_from_artifact


def run_quality_inference(
    *,
    image_file,
    model_root,
    model_name: str,
    model_version: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    model_root_path = Path(model_root)
    model_bundle_root = model_root_path / model_name / model_version

    artifacts = manifest.get("artifacts") or []
    if not artifacts:
        raise ValueError("Manifest has no artifacts entry for task2 inference.")

    artifact_rel_path = artifacts[0].get("path")
    if not artifact_rel_path:
        raise ValueError("Manifest artifact path is missing.")

    artifact_path = model_bundle_root / artifact_rel_path
    predicted_class, confidence_01, class_probabilities, _ = infer_binary_from_artifact(
        image_file=image_file,
        artifact_path=artifact_path,
    )

    explanation_payload = {
        "note": "task2-blank-slate",
        "model_artifact": str(artifact_rel_path),
    }

    return {
        # Blank-slate defaults until Task 2 post-processing is reimplemented.
        "color_score": 0.0,
        "size_score": 0.0,
        "ripeness_score": 0.0,
        "confidence": round(confidence_01 * 100.0, 2),
        "predicted_class": predicted_class,
        "overall_grade": "UNSET",
        "class_probabilities": class_probabilities,
        "explanation_payload": explanation_payload,
        "transparency_refs": [f"model://{model_name}/{model_version}"],
        "model_version_used": model_version,
    }
