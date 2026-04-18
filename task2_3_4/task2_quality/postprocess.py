"""Post-process binary produce predictions into quality attributes and grades.

This module translates a model output (Fresh/Rotten + confidence) into three
proxy quality attributes:
- Colour
- Size
- Ripeness

Then it assigns an overall grade using fixed threshold rules and simulates
inventory actions, including discount recommendations for lower-grade stock.
"""

from __future__ import annotations

from typing import Dict
from task2_3_4.shared.quality_rules import (
    QualityScores,
    assign_overall_grade,
    clamp,
    normalize_label,
    update_inventory_and_discount,
)


def generate_quality_attributes(label: str, confidence: float) -> QualityScores:
    """Generate Colour, Size, and Ripeness percentages from model output.

    The heuristic uses a freshness factor derived from the predicted label and
    confidence score:
    - If prediction is Fresh, freshness factor increases with confidence.
    - If prediction is Rotten, freshness factor decreases with confidence.

    Args:
        label: Predicted class label (Fresh or Rotten).
        confidence: Model confidence for the predicted class in [0.0, 1.0].

    Returns:
        QualityScores with Colour, Size, and Ripeness percentages.
    """
    normalized_label = normalize_label(label)
    conf = clamp(confidence, lower=0.0, upper=1.0)

    # Convert confidence into a freshness proxy in [0, 1].
    # High-confidence Fresh -> freshness_factor near 1.
    # High-confidence Rotten -> freshness_factor near 0.
    freshness_factor = conf if normalized_label == "fresh" else 1.0 - conf

    # Map freshness proxy into each attribute's scale.
    # Distinct ranges keep attributes realistic while preserving correlation.
    colour = 40.0 + 60.0 * freshness_factor
    size = 48.0 + 50.0 * freshness_factor
    ripeness = 35.0 + 65.0 * freshness_factor

    # Add a small penalty when the model is very confident the item is rotten.
    # This sharpens lower-end scores for clearly poor-quality samples.
    if normalized_label == "rotten" and conf > 0.8:
        penalty = (conf - 0.8) * 25.0
        colour -= penalty
        size -= penalty * 0.8
        ripeness -= penalty * 1.1

    return QualityScores(
        colour=round(clamp(colour), 2),
        size=round(clamp(size), 2),
        ripeness=round(clamp(ripeness), 2),
    )


def process_prediction(label: str, confidence: float) -> Dict[str, object]:
    """End-to-end post-processing for a single model prediction.

    This orchestration function keeps the grading layer and inventory action
    layer modular while exposing a simple interface for callers.

    Args:
        label: Predicted class label (Fresh or Rotten).
        confidence: Confidence score in the range [0.0, 1.0].

    Returns:
        A dictionary with:
        - input_label
        - input_confidence
        - quality_scores
        - overall_grade
        - inventory_action
    """
    scores = generate_quality_attributes(label=label, confidence=confidence)
    grade = assign_overall_grade(scores)
    inventory_action = update_inventory_and_discount(grade)

    return {
        "input_label": label,
        "input_confidence": round(clamp(confidence, 0.0, 1.0), 4),
        "quality_scores": {
            "colour": scores.colour,
            "size": scores.size,
            "ripeness": scores.ripeness,
        },
        "overall_grade": grade,
        "inventory_action": inventory_action,
    }


def main() -> None:
    """Run a small demonstration of the post-processing pipeline."""
    example_predictions = [
        ("Fresh", 0.96),
        ("Fresh", 0.74),
        ("Rotten", 0.81),
        ("Rotten", 0.96),
    ]

    for label, confidence in example_predictions:
        result = process_prediction(label=label, confidence=confidence)
        print("-" * 72)
        print(f"Input: label={result['input_label']}, confidence={result['input_confidence']}")
        print(f"Quality Scores: {result['quality_scores']}")
        print(f"Overall Grade: {result['overall_grade']}")
        print(f"Inventory Action: {result['inventory_action']}")


if __name__ == "__main__":
    main()