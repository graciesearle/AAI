from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from ai_core.config import get_service_config
from ai_core.manifest import ManifestError, load_manifest
from task2_quality.runtime import run_quality_inference
from task2_quality.serializers import (
    QualityPredictRequestSerializer,
    QualityPredictResponseSerializer,
)


class QualityPredictView(APIView):
    def post(self, request):
        serializer = QualityPredictRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        cfg = get_service_config()
        model_name = cfg.default_model_name
        model_version = serializer.validated_data.get("model_version", cfg.default_model_version)

        manifest = None
        try:
            manifest = load_manifest(cfg.model_root, model_name, model_version)
        except ManifestError:
            # Keep endpoint usable during early setup; enforce strictly later.
            manifest = None

        try:
            payload = run_quality_inference(
                model_name=model_name,
                model_version=model_version,
                manifest=manifest,
            )
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        output = QualityPredictResponseSerializer(payload)
        return Response(output.data, status=status.HTTP_200_OK)
