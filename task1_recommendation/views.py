from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from ai_core.config import get_service_config
from ai_core.manifest import ManifestError, load_manifest
from task1_recommendation.runtime import build_recommendations
from task1_recommendation.serializers import (
    RecommendationRequestSerializer,
    RecommendationResponseSerializer,
)


class RecommendationView(APIView):
    def post(self, request):
        serializer = RecommendationRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        cfg = get_service_config()
        model_name = "recommendation-engine"
        model_version = serializer.validated_data.get("model_version", "0.1.0")

        manifest = None
        try:
            manifest = load_manifest(cfg.model_root, model_name, model_version)
        except ManifestError:
            manifest = None

        try:
            payload = build_recommendations(
                model_name=model_name,
                model_version=model_version,
                recent_items=serializer.validated_data.get("recent_items", []),
                manifest=manifest,
            )
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        output = RecommendationResponseSerializer(payload)
        return Response(output.data, status=status.HTTP_200_OK)
