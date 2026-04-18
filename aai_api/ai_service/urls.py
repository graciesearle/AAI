from django.urls import path

from aai_api.api_adapters.task1 import RecommendationAdapterView
from aai_api.api_adapters.task2 import QualityPredictAdapterView
from aai_api.api_adapters.task3 import (
    LifecycleModelActivateAdapterView,
    LifecycleModelListAdapterView,
    LifecycleModelRollbackAdapterView,
    LifecycleModelUploadAdapterView,
)
from aai_api.api_adapters.task4 import ExplainAdapterView
from aai_api.ai_core.views import HealthView

urlpatterns = [
    path("api/health/", HealthView.as_view(), name="health"),
    path("api/task1/recommend/", RecommendationAdapterView.as_view(), name="task1-recommend"),
    path("api/task2/predict/", QualityPredictAdapterView.as_view(), name="task2-predict"),
    path("api/task3/models/", LifecycleModelListAdapterView.as_view(), name="task3-model-list"),
    path("api/task3/models/upload/", LifecycleModelUploadAdapterView.as_view(), name="task3-model-upload"),
    path("api/task3/models/activate/", LifecycleModelActivateAdapterView.as_view(), name="task3-model-activate"),
    path("api/task3/models/rollback/", LifecycleModelRollbackAdapterView.as_view(), name="task3-model-rollback"),
    path("api/task4/explain/", ExplainAdapterView.as_view(), name="task4-explain"),
]
