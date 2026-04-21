from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from aai_api.ai_core.config import get_service_config
from aai_api.ai_core.manifest import ManifestError, load_manifest, get_bundle_root
from aai_api.ai_core.lifecycle import get_active_model_version
from task2_3_4.task4_xai.runtime import build_explanation
from task2_3_4.task4_xai.serializers import (
    ExplainRequestSerializer,
    ExplainResponseSerializer,
)


class ExplainAdapterView(APIView):
    def post(self, request):
        serializer = ExplainRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        cfg = get_service_config()
        model_name = serializer.validated_data.get("model_name", "produce-quality")
        model_version = serializer.validated_data.get("model_version") or get_active_model_version(cfg.model_root, model_name) or "1.0.0"

        try:
            manifest = load_manifest(cfg.model_root, model_name, model_version)
            artifact = next(a for a in manifest["artifacts"] if a["type"] == "model_weights")
            checkpoint_path = get_bundle_root(cfg.model_root, model_name, model_version) / artifact["path"]
        except (ManifestError, StopIteration):
            return Response({"detail": "Model not found."}, status=404)

        try:
            payload = build_explanation(
                image_file=serializer.validated_data["image"],
                checkpoint_path=checkpoint_path,
                model_name=model_name,
                model_version=model_version,
                manifest=manifest,
            )
            output = ExplainResponseSerializer(payload)
            return Response(output.data, status=status.HTTP_200_OK)

        except Exception as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        