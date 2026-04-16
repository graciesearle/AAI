from django.urls import path

from task4_xai.views import ExplainView

urlpatterns = [
    path("explain/", ExplainView.as_view(), name="task4-explain"),
]
