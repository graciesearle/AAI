from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from aai_api.ai_core.config import get_service_config
from aai_api.ai_core.lifecycle import list_model_versions, register_model_version
from aai_api.ai_core.manifest import ManifestError, load_manifest
from aai_api.ai_core.utils import sha256_file, utc_iso_now


class Command(BaseCommand):
    help = "Create and verify a lifecycle model bundle for persistence smoke checks."

    def add_arguments(self, parser):
        parser.add_argument(
            "--model-name",
            default="produce-quality",
            help="Model name to prepare/verify.",
        )
        parser.add_argument(
            "--model-version",
            required=True,
            help="Model version to prepare/verify.",
        )
        parser.add_argument(
            "--task-profile",
            default="task2_quality",
            help="Task profile to store in generated manifest.",
        )
        parser.add_argument(
            "--verify-only",
            action="store_true",
            help="Only verify an existing bundle/registry entry; do not create files.",
        )

    def handle(self, *args, **options):
        model_name = str(options["model_name"]).strip()
        model_version = str(options["model_version"]).strip()
        task_profile = str(options["task_profile"]).strip()
        verify_only = bool(options.get("verify_only", False))

        if not model_name or not model_version:
            raise CommandError("--model-name and --model-version are required")

        cfg = get_service_config()

        if not verify_only:
            self._prepare_bundle(
                model_root=cfg.model_root,
                model_name=model_name,
                model_version=model_version,
                task_profile=task_profile,
            )

        summary = self._verify_bundle(
            model_root=cfg.model_root,
            model_name=model_name,
            model_version=model_version,
        )

        self.stdout.write(self.style.SUCCESS(json.dumps(summary, indent=2, sort_keys=True)))

    def _prepare_bundle(self, *, model_root: Path, model_name: str, model_version: str, task_profile: str) -> None:
        bundle_root = model_root / model_name / model_version
        artifacts_dir = bundle_root / "artifacts"
        artifact_path = artifacts_dir / "smoke.bin"

        artifacts_dir.mkdir(parents=True, exist_ok=True)

        if not artifact_path.exists():
            payload = f"persistence-smoke|{model_name}|{model_version}|{utc_iso_now()}".encode("utf-8")
            artifact_path.write_bytes(payload)

        checksum = sha256_file(artifact_path)

        manifest = {
            "model_name": model_name,
            "model_version": model_version,
            "task_profile": task_profile,
            "schema_version": "task2-quality-v1",
            "framework": "pytorch",
            "entrypoint": "aai_api.ai_core.task2_runtime:run_quality_inference",
            "artifacts": [
                {
                    "type": "model_weights",
                    "path": "artifacts/smoke.bin",
                    "checksum": checksum,
                }
            ],
            "input_schema": {
                "image": "multipart-file",
                "producer_id": "int",
                "product_id": "int?",
                "model_version": "str?",
            },
            "output_schema": {
                "overall_grade": "str",
                "confidence": "float",
                "model_version_used": "str",
            },
            "metrics": {
                "smoke": True,
            },
            "created_at": utc_iso_now(),
        }

        manifest_path = bundle_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        try:
            loaded_manifest = load_manifest(model_root, model_name, model_version)
        except ManifestError as exc:
            raise CommandError(f"Unable to validate generated manifest: {exc}") from exc

        register_model_version(
            model_root,
            manifest=loaded_manifest,
            source="persistence-smoke",
        )

    def _verify_bundle(self, *, model_root: Path, model_name: str, model_version: str) -> dict[str, object]:
        try:
            manifest = load_manifest(model_root, model_name, model_version)
        except ManifestError as exc:
            raise CommandError(f"Unable to verify model bundle: {exc}") from exc

        artifacts = manifest.get("artifacts") or []
        if not isinstance(artifacts, list) or not artifacts:
            raise CommandError("Manifest has no artifacts to verify")

        bundle_root = model_root / model_name / model_version
        missing_artifacts: list[str] = []
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            relative_path = str(artifact.get("path", "")).strip()
            if not relative_path:
                continue
            artifact_path = (bundle_root / relative_path).resolve()
            if not artifact_path.exists() or not artifact_path.is_file():
                missing_artifacts.append(relative_path)

        if missing_artifacts:
            raise CommandError(
                "Missing artifact files: " + ", ".join(sorted(set(missing_artifacts)))
            )

        entries = list_model_versions(model_root)
        in_registry = any(
            item.get("model_name") == model_name and item.get("model_version") == model_version
            for item in entries
            if isinstance(item, dict)
        )
        if not in_registry:
            raise CommandError("Model version is not present in lifecycle registry")

        return {
            "status": "ok",
            "model_root": str(model_root),
            "model_name": model_name,
            "model_version": model_version,
            "artifact_count": len(artifacts),
            "registry_entry_found": True,
        }
