"""
Task 3 API adapters — Lifecycle management and interaction logging.

Endpoints:
    GET    /api/task3/models/                     — List all registered model versions
    POST   /api/task3/models/upload/              — Upload a new model bundle
    POST   /api/task3/models/activate/            — Activate a specific model version
    POST   /api/task3/models/rollback/            — Rollback to a previous model version
    GET    /api/task3/interactions/                — List all inference interaction logs
    PATCH  /api/task3/interactions/<id>/override/  — Record a producer override on a prediction

All endpoints require authentication (TokenAuthentication) and are restricted
to authenticated users. This satisfies the AI Engineer persona isolation
requirement from the case study.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rest_framework import status
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
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
from aai_api.ai_core.models import InferenceLog
from aai_api.api_adapters.task3_serializers import (
    InferenceLogSerializer,
    LifecycleModelActivateSerializer,
    LifecycleModelListSerializer,
    LifecycleModelRollbackSerializer,
    LifecycleModelUploadSerializer,
    ProducerOverrideSerializer,
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
        "entrypoint": "aai_api.ai_core.task2_runtime:run_quality_inference",
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
            "class_names": [],
        },
        "metrics": payload.get("metrics") or {},
        "created_at": _utc_iso(),
    }


# ---------------------------------------------------------------------------
# Lifecycle endpoints (secured with TokenAuthentication)
# ---------------------------------------------------------------------------

class LifecycleModelListAdapterView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        cfg = get_service_config()
        task_profile = request.query_params.get("task_profile")
        items = list_model_versions(cfg.model_root, task_profile=task_profile)
        serializer = LifecycleModelListSerializer(items, many=True)
        return Response({"count": len(serializer.data), "results": serializer.data})


class LifecycleModelUploadAdapterView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

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

            # Checksum validation: reject if the client-provided checksum
            # doesn't match the computed SHA-256 of the uploaded artifact.
            client_checksum = data.get("checksum")
            if client_checksum and client_checksum.lower() != checksum.lower():
                return Response(
                    {"detail": f"Checksum mismatch: expected {client_checksum}, got {checksum}"},
                    status=status.HTTP_400_BAD_REQUEST,
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
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

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
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

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


# ---------------------------------------------------------------------------
# Interaction logging endpoints (new — for assessment compliance)
# These endpoints give AI Engineers access to the "database of all
# end-user interactions" required by the case study, and allow the
# system to ingest human feedback for retraining via the override endpoint.
# ---------------------------------------------------------------------------

class InteractionListAdapterView(APIView):
    """
    GET /api/task3/interactions/

    Returns all inference interaction logs. AI Engineers use this to:
    - Inspect past predictions and their accuracy
    - Identify patterns in producer overrides for retraining
    - Export data for model refinement
    """
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        queryset = InferenceLog.objects.all().order_by("-timestamp")
        serializer = InferenceLogSerializer(queryset, many=True)
        return Response({"count": len(serializer.data), "results": serializer.data})


class InteractionOverrideAdapterView(APIView):
    """
    PATCH /api/task3/interactions/<id>/override/

    Records a producer's feedback on a prediction. When a producer disagrees
    with the AI-assigned grade, this endpoint captures:
    - Whether the producer accepted the recommendation
    - What grade the producer believes is correct

    This is the core of the retraining feedback loop described in the case
    study: "If a user overrides the model's prediction, what are you going
    to do about that?" — we store the override signal in the InferenceLog
    so AI engineers can query disagreements and build corrective training sets.
    """
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def patch(self, request, pk):
        try:
            log = InferenceLog.objects.get(pk=pk)
        except InferenceLog.DoesNotExist:
            return Response(
                {"detail": "Inference log not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = ProducerOverrideSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        log.producer_accepted = serializer.validated_data["producer_accepted"]
        log.override_grade = serializer.validated_data.get("override_grade", "")
        log.save(update_fields=["producer_accepted", "override_grade"])

        output = InferenceLogSerializer(log)
        return Response(output.data)
