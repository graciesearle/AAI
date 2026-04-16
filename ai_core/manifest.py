import json
from pathlib import Path
from typing import Any


REQUIRED_MANIFEST_KEYS = {
    "model_name",
    "model_version",
    "task_profile",
    "schema_version",
    "framework",
    "entrypoint",
    "artifacts",
    "input_schema",
    "output_schema",
    "metrics",
    "created_at",
}


class ManifestError(ValueError):
    pass


def get_manifest_path(model_root: Path, model_name: str, model_version: str) -> Path:
    return model_root / model_name / model_version / "manifest.json"


def load_manifest(model_root: Path, model_name: str, model_version: str) -> dict[str, Any]:
    path = get_manifest_path(model_root, model_name, model_version)
    if not path.exists():
        raise ManifestError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    missing = REQUIRED_MANIFEST_KEYS.difference(payload.keys())
    if missing:
        missing_values = ", ".join(sorted(missing))
        raise ManifestError(f"Manifest is missing required keys: {missing_values}")

    return payload
