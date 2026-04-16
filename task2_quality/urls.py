from django.urls import path

from task2_quality.views import QualityPredictView

urlpatterns = [
    path("predict/", QualityPredictView.as_view(), name="task2-predict"),
]
