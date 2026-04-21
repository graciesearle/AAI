import json
from pathlib import Path
from typing import Any

from aai_api.ai_core.utils import sha256_file


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

REQUIRED_ARTIFACT_KEYS = {
    "type",
    "path",
    "checksum",
}


class ManifestError(ValueError):
    pass


def get_manifest_path(model_root: Path, model_name: str, model_version: str) -> Path:
    return model_root / model_name / model_version / "manifest.json"


def get_bundle_root(model_root: Path, model_name: str, model_version: str) -> Path:
    return model_root / model_name / model_version


def _validate_artifacts(
    *,
    model_root: Path,
    model_name: str,
    model_version: str,
    artifacts: Any,
) -> None:
    if not isinstance(artifacts, list) or not artifacts:
        raise ManifestError("Manifest field 'artifacts' must be a non-empty list")

    bundle_root = get_bundle_root(model_root, model_name, model_version)
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise ManifestError(f"Manifest artifact at index {index} must be an object")

        missing = REQUIRED_ARTIFACT_KEYS.difference(artifact.keys())
        if missing:
            missing_values = ", ".join(sorted(missing))
            raise ManifestError(
                f"Manifest artifact at index {index} is missing required keys: {missing_values}"
            )

        relative_path = str(artifact.get("path", "")).strip()
        if not relative_path:
            raise ManifestError(f"Manifest artifact at index {index} has an empty path")

        artifact_path = (bundle_root / relative_path).resolve()
        try:
            artifact_path.relative_to(bundle_root.resolve())
        except ValueError as exc:
            raise ManifestError(
                f"Manifest artifact at index {index} resolves outside model bundle"
            ) from exc

        if not artifact_path.exists() or not artifact_path.is_file():
            raise ManifestError(f"Manifest artifact not found: {artifact_path}")

        checksum = str(artifact.get("checksum", "")).strip().lower()
        if not checksum:
            raise ManifestError(f"Manifest artifact at index {index} has an empty checksum")

        actual_checksum = sha256_file(artifact_path)
        if actual_checksum != checksum:
            raise ManifestError(
                "Manifest artifact checksum mismatch for "
                f"'{relative_path}' (expected {checksum}, got {actual_checksum})"
            )


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

    manifest_model_name = str(payload.get("model_name", "")).strip()
    manifest_model_version = str(payload.get("model_version", "")).strip()
    if manifest_model_name != model_name:
        raise ManifestError(
            f"Manifest model_name mismatch (expected '{model_name}', got '{manifest_model_name}')"
        )
    if manifest_model_version != model_version:
        raise ManifestError(
            "Manifest model_version mismatch "
            f"(expected '{model_version}', got '{manifest_model_version}')"
        )

    _validate_artifacts(
        model_root=model_root,
        model_name=model_name,
        model_version=model_version,
        artifacts=payload.get("artifacts"),
    )

    return payload
