from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from aai_api.ai_core.config import get_service_config
from aai_api.ai_core.lifecycle import get_active_model_version
from aai_api.ai_core.manifest import ManifestError, load_manifest
from task2_3_4.task2_quality.model_inference import ModelInferenceError
from task2_3_4.task2_quality.runtime import run_quality_inference
from task2_3_4.task2_quality.serializers import (
    QualityPredictRequestSerializer,
    QualityPredictResponseSerializer,
)


class QualityPredictAdapterView(APIView):
    def post(self, request):
        serializer = QualityPredictRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        cfg = get_service_config()
        model_name = cfg.default_model_name
        requested_model_version = serializer.validated_data.get("model_version")
        model_version = (
            requested_model_version
            or get_active_model_version(cfg.model_root, model_name)
            or cfg.default_model_version
        )

        try:
            manifest = load_manifest(cfg.model_root, model_name, model_version)
        except ManifestError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        try:
            payload = run_quality_inference(
                image_file=serializer.validated_data["image"],
                model_root=cfg.model_root,
                model_name=model_name,
                model_version=model_version,
                manifest=manifest,
            )
        except ModelInferenceError as exc:
            return Response(
                {"detail": f"Model inference failed for '{model_name}/{model_version}': {exc}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        output = QualityPredictResponseSerializer(payload)
        return Response(output.data, status=status.HTTP_200_OK)
