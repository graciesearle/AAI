from rest_framework import serializers


class RecommendationRequestSerializer(serializers.Serializer):
    customer_id = serializers.IntegerField(required=False)
    recent_items = serializers.ListField(
        child=serializers.CharField(max_length=120),
        required=False,
        allow_empty=True,
    )
    model_version = serializers.CharField(max_length=64, required=False, allow_blank=False)


class RecommendationResponseSerializer(serializers.Serializer):
    recommended_items = serializers.ListField(child=serializers.CharField())
    confidence = serializers.FloatField()
    explanation_payload = serializers.DictField(required=False)
    model_version_used = serializers.CharField()
    schema_version = serializers.CharField()
