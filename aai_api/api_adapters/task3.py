from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from aai_api.ai_core.config import get_service_config
from aai_api.ai_core.lifecycle import (
    LifecycleError,
    list_model_versions,
    register_model_version,
    rollback_model_version,
    set_active_model_version,
)
from aai_api.ai_core.manifest import ManifestError, load_manifest
from task2_3_4.task3_lifecycle.serializers import (
    LifecycleModelActivateSerializer,
    LifecycleModelListSerializer,
    LifecycleModelRollbackSerializer,
    LifecycleModelUploadSerializer,
)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_uploaded_artifact(*, model_root: Path, model_name: str, model_version: str, artifact_file) -> tuple[str, str]:
    artifacts_dir = model_root / model_name / model_version / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(str(getattr(artifact_file, "name", "model.bin"))).name
    artifact_path = artifacts_dir / filename

    with artifact_path.open("wb") as out:
        for chunk in artifact_file.chunks():
            out.write(chunk)

    checksum = _sha256(artifact_path)
    bundle_root = model_root / model_name / model_version
    relative_path = str(artifact_path.relative_to(bundle_root).as_posix())
    return relative_path, checksum


def _build_manifest_from_upload(*, payload: dict[str, Any], artifact_path: str, checksum: str) -> dict[str, Any]:
    model_name = str(payload["model_name"]).strip()
    model_version = str(payload["model_version"]).strip()

    return {
        "model_name": model_name,
        "model_version": model_version,
        "task_profile": str(payload.get("task_profile", "task2_quality")),
        "schema_version": str(payload.get("schema_version", "task2-quality-v1")),
        "framework": str(payload.get("framework", "")),
        "entrypoint": "task2_3_4.task2_quality.runtime:run_quality_inference",
        "artifacts": [
            {
                "type": "model_weights",
                "path": artifact_path,
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
            "color_score": "float",
            "size_score": "float",
            "ripeness_score": "float",
            "confidence": "float",
            "predicted_class": "str",
            "overall_grade": "str",
            "class_probabilities": "object",
            "explanation_payload": "object",
            "transparency_refs": "array",
            "model_version_used": "str",
        },
        "metrics": payload.get("metrics") or {},
        "created_at": _utc_iso(),
    }


class LifecycleModelListAdapterView(APIView):
    authentication_classes = []
    permission_classes = []

    def get(self, request):
        cfg = get_service_config()
        task_profile = request.query_params.get("task_profile")
        items = list_model_versions(cfg.model_root, task_profile=task_profile)
        serializer = LifecycleModelListSerializer(items, many=True)
        return Response({"count": len(serializer.data), "results": serializer.data})


class LifecycleModelUploadAdapterView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        serializer = LifecycleModelUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        cfg = get_service_config()
        data = serializer.validated_data

        manifest = data.get("manifest_json")
        model_name = str(data.get("model_name") or manifest.get("model_name")).strip()
        model_version = str(data.get("model_version") or manifest.get("model_version")).strip()

        if manifest is None:
            artifact_path, checksum = _write_uploaded_artifact(
                model_root=cfg.model_root,
                model_name=model_name,
                model_version=model_version,
                artifact_file=data["artifact"],
            )
            manifest = _build_manifest_from_upload(payload=data, artifact_path=artifact_path, checksum=checksum)
        else:
            manifest = dict(manifest)

        bundle_root = cfg.model_root / model_name / model_version
        bundle_root.mkdir(parents=True, exist_ok=True)
        manifest_path = bundle_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        try:
            loaded_manifest = load_manifest(cfg.model_root, model_name, model_version)
            record = register_model_version(
                cfg.model_root,
                manifest=loaded_manifest,
                source="upload-api",
            )
        except (ManifestError, LifecycleError) as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        return Response(record, status=status.HTTP_201_CREATED)


class LifecycleModelActivateAdapterView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        serializer = LifecycleModelActivateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        cfg = get_service_config()
        data = serializer.validated_data

        try:
            load_manifest(cfg.model_root, data["model_name"], data["model_version"])
            result = set_active_model_version(
                cfg.model_root,
                model_name=data["model_name"],
                model_version=data["model_version"],
                source="activate",
            )
        except (ManifestError, LifecycleError) as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        return Response({"detail": "Model activated", **result})


class LifecycleModelRollbackAdapterView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        serializer = LifecycleModelRollbackSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        cfg = get_service_config()
        data = serializer.validated_data

        try:
            result = rollback_model_version(
                cfg.model_root,
                model_name=data["model_name"],
                target_model_version=data.get("target_model_version"),
            )
        except LifecycleError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        return Response({"detail": "Rollback complete", **result})
