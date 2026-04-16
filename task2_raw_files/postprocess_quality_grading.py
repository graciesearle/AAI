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

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class QualityScores:
    """Container for generated quality percentages.

    Attributes:
        colour: Colour quality as a percentage in the range [0, 100].
        size: Size quality as a percentage in the range [0, 100].
        ripeness: Ripeness quality as a percentage in the range [0, 100].
    """

    colour: float
    size: float
    ripeness: float


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    """Clamp a numeric value to a closed interval.

    Args:
        value: Input value to clamp.
        lower: Minimum allowed value.
        upper: Maximum allowed value.

    Returns:
        The clamped value such that lower <= result <= upper.
    """
    return max(lower, min(value, upper))


def normalize_label(label: str) -> str:
    """Normalize and validate a predicted class label.

    Args:
        label: Raw label string, expected to represent Fresh or Rotten.

    Returns:
        A normalized lowercase label: "fresh" or "rotten".

    Raises:
        ValueError: If label is not Fresh or Rotten.
    """
    normalized = label.strip().lower()
    if normalized not in {"fresh", "rotten"}:
        raise ValueError("label must be either 'Fresh' or 'Rotten'.")
    return normalized


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


def assign_overall_grade(scores: QualityScores) -> str:
    """Assign an overall grade using fixed threshold rules.

    Threshold policy (strict):
    - Grade C if any metric falls below Grade C limits:
      Colour < 65, Size < 70, or Ripeness < 60.
    - Grade B if not Grade C, and any metric falls below Grade B limits:
      Colour < 75, Size < 80, or Ripeness < 70.
    - Grade A otherwise.

    Args:
        scores: QualityScores with Colour, Size, and Ripeness percentages.

    Returns:
        Overall grade as "A", "B", or "C".
    """
    if scores.colour < 65.0 or scores.size < 70.0 or scores.ripeness < 60.0:
        return "C"

    if scores.colour < 75.0 or scores.size < 80.0 or scores.ripeness < 70.0:
        return "B"

    return "A"


def update_inventory_and_discount(grade: str) -> Dict[str, object]:
    """Simulate inventory update and discount recommendation by grade.

    Business policy:
    - Grade A: Keep standard pricing.
    - Grade B: Recommend moderate discount.
    - Grade C: Recommend aggressive discount.

    Args:
        grade: Overall grade string ("A", "B", or "C").

    Returns:
        Dictionary containing simulated inventory action fields:
        - status: Inventory quality status text.
        - discount_recommended: Boolean flag for discount action.
        - discount_percent: Suggested markdown percentage.
        - action_note: Human-readable operational recommendation.

    Raises:
        ValueError: If grade is not one of A, B, C.
    """
    normalized_grade = grade.strip().upper()
    if normalized_grade not in {"A", "B", "C"}:
        raise ValueError("grade must be one of: 'A', 'B', 'C'.")

    if normalized_grade == "A":
        return {
            "status": "premium_stock",
            "discount_recommended": False,
            "discount_percent": 0,
            "action_note": "Keep at regular price and standard shelf placement.",
        }

    if normalized_grade == "B":
        return {
            "status": "lower_grade_stock",
            "discount_recommended": True,
            "discount_percent": 10,
            "action_note": "Apply mild discount and prioritize near-term sale.",
        }

    return {
        "status": "lower_grade_stock",
        "discount_recommended": True,
        "discount_percent": 25,
        "action_note": "Apply strong discount and move to clearance channel.",
    }


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