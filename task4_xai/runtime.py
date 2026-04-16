from typing import Any


def build_explanation(*, model_name: str, model_version: str, context: dict[str, Any] | None = None, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    if manifest and manifest.get("task_profile") != "task4_xai":
        raise ValueError("Selected model bundle is not task4_xai profile")

    return {
        "model_version_used": model_version,
        "schema_version": "task4-xai-v1",
        "transparency_refs": ["xai://saliency-placeholder", "xai://rules-placeholder"],
        "explanation_payload": {
            "note": "stub-response",
            "model_name": model_name,
            "context": context or {},
            "derivation": {
                "quality_rules": "kept in DESD policy layer",
                "model_reasoning": "placeholder attribution",
            },
        },
    }
