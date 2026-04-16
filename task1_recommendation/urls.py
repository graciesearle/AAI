from django.urls import path

from task1_recommendation.views import RecommendationView

urlpatterns = [
    path("recommend/", RecommendationView.as_view(), name="task1-recommend"),
]
