from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from task2_raw_files.postprocess_quality_grading import process_prediction


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _extract_image_signals(image_file) -> dict[str, float]:
    image_file.seek(0)
    image = Image.open(image_file).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32) / 255.0

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    max_channel = np.max(rgb, axis=2)
    min_channel = np.min(rgb, axis=2)

    saturation = np.mean((max_channel - min_channel) / np.maximum(max_channel, 1e-6))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    brightness = float(np.mean(luminance))
    dark_ratio = float(np.mean(luminance < 0.22))
    texture_signal = float(np.std(luminance))

    green_strength = float(np.mean(green))
    brownness = float(np.mean(np.clip(red - green, 0.0, 1.0)))

    freshness_signal = (
        0.45 * (1.0 - dark_ratio)
        + 0.20 * saturation
        + 0.20 * _clamp01(green_strength - brownness + 0.35)
        + 0.15 * _clamp01(texture_signal / 0.35)
    )

    return {
        "freshness_signal": _clamp01(float(freshness_signal)),
        "brightness": _clamp01(brightness),
        "saturation": _clamp01(float(saturation)),
        "dark_ratio": _clamp01(dark_ratio),
        "texture_signal": _clamp01(texture_signal),
        "green_strength": _clamp01(green_strength),
        "brownness": _clamp01(brownness),
    }


def _infer_label_and_confidence(signals: dict[str, float]) -> tuple[str, float, dict[str, float]]:
    freshness_signal = signals["freshness_signal"]
    predicted_class = "fresh" if freshness_signal >= 0.5 else "rotten"

    margin = abs(freshness_signal - 0.5) * 2.0
    confidence_01 = 0.55 + 0.40 * _clamp01(margin)

    fresh_probability = _clamp01(0.05 + 0.90 * freshness_signal)
    class_probabilities = {
        "fresh": round(fresh_probability, 6),
        "rotten": round(1.0 - fresh_probability, 6),
    }
    return predicted_class, confidence_01, class_probabilities


def run_quality_inference(
    *,
    image_file,
    model_name: str,
    model_version: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if manifest.get("task_profile") != "task2_quality":
        raise ValueError("Selected model bundle is not task2_quality profile")

    signals = _extract_image_signals(image_file)
    predicted_class, confidence_01, class_probabilities = _infer_label_and_confidence(signals)

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
        "explanation_payload": {
            "note": "image-signal-inference-v1",
            "model_name": model_name,
            "model_version": model_version,
            "signals": {key: round(value, 6) for key, value in signals.items()},
        },
        "transparency_refs": [f"model://{model_name}/{model_version}"],
        "model_version_used": model_version,
    }
