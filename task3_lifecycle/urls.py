from django.urls import path

from task3_lifecycle.views import (
    LifecycleModelActivateView,
    LifecycleModelListView,
    LifecycleModelRollbackView,
    LifecycleModelUploadView,
)

urlpatterns = [
    path("models/", LifecycleModelListView.as_view(), name="task3-model-list"),
    path("models/upload/", LifecycleModelUploadView.as_view(), name="task3-model-upload"),
    path("models/activate/", LifecycleModelActivateView.as_view(), name="task3-model-activate"),
    path("models/rollback/", LifecycleModelRollbackView.as_view(), name="task3-model-rollback"),
]
