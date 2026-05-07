from typing import Any
from .fbt import recommend

def build_recommendations(*, model_name: str, model_version: str, recent_items: list[str] | None = None, top_n: int = 5, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    if manifest and manifest.get("task_profile") != "task1_recommendation":
        raise ValueError("Selected model bundle is not task1_recommendation profile")

    seed_items = recent_items or []
    base = ["Organic Tomatoes", "Organic Carrots", "Organic Potatoes", "Organic Lettuce"]
    recommendations = list(dict.fromkeys(seed_items + base))[:top_n]

    # Call FP-Growth
    results = recommend(basket=recent_items or [], top_n=top_n, use_db=True)
    
    # Fallback to base items if FP-Growth yields no results
    if not results:
        results = [
            {"item": item, "confidence": 0.8, "because_you_bought": "Seasonal Trending", "lift": 1.0}
            for item in base[:top_n]
        ]

    return {
        "recommended_items": [r["item"] for r in results],
        "confidence": round(sum(r["confidence"] for r in results) / len(results), 3) if results else 0.0,
        "explanation_payload": {
            r["item"]: {
                "because_you_bought": r["because_you_bought"],
                "confidence": r["confidence"],
                "lift": r["lift"],
            } for r in results
        },
        "model_version_used": model_version,
        "schema_version": "1.0",
    }
