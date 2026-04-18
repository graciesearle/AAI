from rest_framework.response import Response
from rest_framework.views import APIView

from aai_api.ai_core.config import get_service_config


class HealthView(APIView):
    authentication_classes = []
    permission_classes = []

    def get(self, request):
        cfg = get_service_config()
        return Response(
            {
                "status": "ok",
                "service": "advanced-ai-django-service",
                "default_model_name": cfg.default_model_name,
                "default_model_version": cfg.default_model_version,
                "default_task_profile": cfg.default_task_profile,
            }
        )
