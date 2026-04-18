from typing import Any


def build_recommendations(*, model_name: str, model_version: str, recent_items: list[str] | None = None, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    if manifest and manifest.get("task_profile") != "task1_recommendation":
        raise ValueError("Selected model bundle is not task1_recommendation profile")

    seed_items = recent_items or []
    base = ["Tomatoes", "Apples", "Carrots"]
    recommendations = list(dict.fromkeys(seed_items + base))[:5]

    return {
        "recommended_items": recommendations,
        "confidence": 0.76,
        "model_version_used": model_version,
        "schema_version": "task1-reco-v1",
        "explanation_payload": {
            "note": "stub-response",
            "strategy": "recent-items-plus-popular-baseline",
            "model_name": model_name,
        },
    }
