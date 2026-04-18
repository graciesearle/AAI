from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class QualityScores:
    colour: float
    size: float
    ripeness: float


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(value, upper))


def normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized in {"fresh", "rotten"}:
        return normalized

    if "fresh" in normalized or "healthy" in normalized:
        return "fresh"
    if "rotten" in normalized or "disease" in normalized or "spoiled" in normalized:
        return "rotten"

    raise ValueError(
        "label must map to Fresh or Rotten "
        "(examples: Fresh, Rotten, Healthy, Apple__Rotten)."
    )


def validate_quality_scores(quality_scores: Mapping[str, float] | QualityScores) -> QualityScores:
    if isinstance(quality_scores, QualityScores):
        return QualityScores(
            colour=round(clamp(float(quality_scores.colour)), 2),
            size=round(clamp(float(quality_scores.size)), 2),
            ripeness=round(clamp(float(quality_scores.ripeness)), 2),
        )

    required_keys = {"colour", "size", "ripeness"}
    missing = required_keys - set(quality_scores.keys())
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"quality_scores is missing required keys: {missing_keys}")

    return QualityScores(
        colour=round(clamp(float(quality_scores["colour"])), 2),
        size=round(clamp(float(quality_scores["size"])), 2),
        ripeness=round(clamp(float(quality_scores["ripeness"])), 2),
    )


def assign_overall_grade(scores: QualityScores) -> str:
    if scores.colour < 65.0 or scores.size < 70.0 or scores.ripeness < 60.0:
        return "C"

    if scores.colour < 75.0 or scores.size < 80.0 or scores.ripeness < 70.0:
        return "B"

    return "A"


def update_inventory_and_discount(grade: str) -> dict[str, object]:
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
