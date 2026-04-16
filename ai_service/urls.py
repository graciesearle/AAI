from django.urls import include, path

from ai_core.views import HealthView

urlpatterns = [
    path("api/health/", HealthView.as_view(), name="health"),
    path("api/task1/", include("task1_recommendation.urls")),
    path("api/task2/", include("task2_quality.urls")),
    path("api/task4/", include("task4_xai.urls")),
]
