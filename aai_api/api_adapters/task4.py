from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from aai_api.ai_core.config import get_service_config
from aai_api.ai_core.manifest import ManifestError, load_manifest
from task2_3_4.task4_xai.runtime import build_explanation
from task2_3_4.task4_xai.serializers import (
    ExplainRequestSerializer,
    ExplainResponseSerializer,
)


class ExplainAdapterView(APIView):
    def post(self, request):
        serializer = ExplainRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        cfg = get_service_config()
        model_name = serializer.validated_data.get("model_name", "xai-engine")
        model_version = serializer.validated_data.get("model_version", "0.1.0")

        manifest = None
        try:
            manifest = load_manifest(cfg.model_root, model_name, model_version)
        except ManifestError:
            manifest = None

        try:
            payload = build_explanation(
                model_name=model_name,
                model_version=model_version,
                context=serializer.validated_data.get("context", {}),
                manifest=manifest,
            )
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        output = ExplainResponseSerializer(payload)
        return Response(output.data, status=status.HTTP_200_OK)
