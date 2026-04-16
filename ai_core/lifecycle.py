from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_core.manifest import ManifestError, load_manifest

REGISTRY_SCHEMA_VERSION = "task3-lifecycle-v1"
REGISTRY_FILE_NAME = "_lifecycle_registry.json"


class LifecycleError(ValueError):
    pass


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _registry_template() -> dict[str, Any]:
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "models": {},
    }


def get_registry_path(model_root: Path) -> Path:
    return model_root / REGISTRY_FILE_NAME


def _read_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _registry_template()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LifecycleError(f"Lifecycle registry is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise LifecycleError("Lifecycle registry root must be an object")

    payload.setdefault("schema_version", REGISTRY_SCHEMA_VERSION)
    payload.setdefault("models", {})
    if not isinstance(payload["models"], dict):
        raise LifecycleError("Lifecycle registry field 'models' must be an object")

    return payload


def _write_registry(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def _iter_manifest_paths(model_root: Path):
    if not model_root.exists():
        return

    for model_dir in model_root.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        for version_dir in model_dir.iterdir():
            if not version_dir.is_dir():
                continue
            manifest_path = version_dir / "manifest.json"
            if manifest_path.exists() and manifest_path.is_file():
                yield model_dir.name, version_dir.name


def _ensure_registry(model_root: Path) -> dict[str, Any]:
    path = get_registry_path(model_root)
    payload = _read_registry(path)

    changed = False
    models = payload["models"]

    for model_name, model_version in _iter_manifest_paths(model_root):
        try:
            manifest = load_manifest(model_root, model_name, model_version)
        except ManifestError:
            continue

        model_bucket = models.setdefault(
            model_name,
            {
                "active_version": None,
                "versions": {},
                "activation_history": [],
            },
        )

        versions = model_bucket.setdefault("versions", {})
        if model_version in versions:
            continue

        versions[model_version] = {
            "model_name": manifest["model_name"],
            "model_version": manifest["model_version"],
            "task_profile": manifest.get("task_profile", ""),
            "schema_version": manifest.get("schema_version", ""),
            "framework": manifest.get("framework", ""),
            "checksum": (manifest.get("artifacts") or [{}])[0].get("checksum", ""),
            "artifact_path": (manifest.get("artifacts") or [{}])[0].get("path", ""),
            "manifest_path": f"{model_name}/{model_version}/manifest.json",
            "created_at": manifest.get("created_at") or _utc_iso(),
            "source": "manifest-discovery",
        }
        changed = True

    if changed:
        _write_registry(path, payload)

    return payload


def list_model_versions(model_root: Path, *, task_profile: str | None = None) -> list[dict[str, Any]]:
    payload = _ensure_registry(model_root)

    results: list[dict[str, Any]] = []
    for model_name, model_bucket in payload["models"].items():
        active_version = model_bucket.get("active_version")
        versions = model_bucket.get("versions", {})
        if not isinstance(versions, dict):
            continue

        for model_version, info in versions.items():
            if not isinstance(info, dict):
                continue
            if task_profile and info.get("task_profile") != task_profile:
                continue

            item = {
                "model_name": model_name,
                "model_version": model_version,
                "is_active": active_version == model_version,
                **info,
            }
            results.append(item)

    results.sort(key=lambda item: (item.get("created_at", ""), item["model_version"]), reverse=True)
    return results


def register_model_version(
    model_root: Path,
    *,
    manifest: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    model_name = str(manifest.get("model_name", "")).strip()
    model_version = str(manifest.get("model_version", "")).strip()
    if not model_name or not model_version:
        raise LifecycleError("Manifest must include model_name and model_version")

    payload = _ensure_registry(model_root)
    models = payload["models"]

    model_bucket = models.setdefault(
        model_name,
        {
            "active_version": None,
            "versions": {},
            "activation_history": [],
        },
    )

    versions = model_bucket.setdefault("versions", {})
    if model_version in versions:
        existing = versions[model_version]
        return {
            "model_name": model_name,
            "model_version": model_version,
            "is_active": model_bucket.get("active_version") == model_version,
            **existing,
        }

    artifacts = manifest.get("artifacts") or []
    first_artifact = artifacts[0] if artifacts and isinstance(artifacts[0], dict) else {}

    versions[model_version] = {
        "model_name": model_name,
        "model_version": model_version,
        "task_profile": manifest.get("task_profile", ""),
        "schema_version": manifest.get("schema_version", ""),
        "framework": manifest.get("framework", ""),
        "checksum": first_artifact.get("checksum", ""),
        "artifact_path": first_artifact.get("path", ""),
        "manifest_path": f"{model_name}/{model_version}/manifest.json",
        "created_at": manifest.get("created_at") or _utc_iso(),
        "source": source,
    }

    _write_registry(get_registry_path(model_root), payload)
    return {
        "model_name": model_name,
        "model_version": model_version,
        "is_active": False,
        **versions[model_version],
    }


def get_active_model_version(model_root: Path, model_name: str) -> str | None:
    payload = _ensure_registry(model_root)
    model_bucket = payload.get("models", {}).get(model_name)
    if not isinstance(model_bucket, dict):
        return None

    active_version = model_bucket.get("active_version")
    if isinstance(active_version, str) and active_version.strip():
        return active_version
    return None


def set_active_model_version(
    model_root: Path,
    *,
    model_name: str,
    model_version: str,
    source: str,
) -> dict[str, Any]:
    payload = _ensure_registry(model_root)
    model_bucket = payload.get("models", {}).get(model_name)
    if not isinstance(model_bucket, dict):
        raise LifecycleError("Model name not found")

    versions = model_bucket.get("versions", {})
    if not isinstance(versions, dict) or model_version not in versions:
        raise LifecycleError("Model version not found")

    previous_active = model_bucket.get("active_version")
    activated_at = _utc_iso()

    model_bucket["active_version"] = model_version
    history = model_bucket.setdefault("activation_history", [])
    if isinstance(history, list):
        history.append(
            {
                "model_version": model_version,
                "activated_at": activated_at,
                "source": source,
            }
        )

    version_payload = versions[model_version]
    if isinstance(version_payload, dict):
        version_payload["activated_at"] = activated_at

    _write_registry(get_registry_path(model_root), payload)
    return {
        "model_name": model_name,
        "model_version": model_version,
        "previous_active_model_version": previous_active,
        "activated_at": activated_at,
    }


def rollback_model_version(
    model_root: Path,
    *,
    model_name: str,
    target_model_version: str | None = None,
) -> dict[str, Any]:
    payload = _ensure_registry(model_root)
    model_bucket = payload.get("models", {}).get(model_name)
    if not isinstance(model_bucket, dict):
        raise LifecycleError("Model name not found")

    versions = model_bucket.get("versions", {})
    if not isinstance(versions, dict) or not versions:
        raise LifecycleError("No versions found for model")

    current_active = model_bucket.get("active_version")

    if target_model_version:
        if target_model_version not in versions:
            raise LifecycleError("Rollback target model version not found")
        target = target_model_version
    else:
        history = model_bucket.get("activation_history", [])
        target = None
        if isinstance(history, list):
            for item in reversed(history):
                if not isinstance(item, dict):
                    continue
                candidate = item.get("model_version")
                if candidate and candidate != current_active and candidate in versions:
                    target = candidate
                    break
        if not target:
            sorted_candidates = sorted(versions.keys(), reverse=True)
            for candidate in sorted_candidates:
                if candidate != current_active:
                    target = candidate
                    break

    if not target:
        raise LifecycleError("No rollback target available")

    return set_active_model_version(
        model_root,
        model_name=model_name,
        model_version=target,
        source="rollback",
    )
