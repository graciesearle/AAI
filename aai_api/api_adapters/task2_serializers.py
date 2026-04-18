from rest_framework import serializers

class QualityPredictRequestSerializer(serializers.Serializer):
    """
    Validates incoming image prediction requests from DESD or direct producers.
    """
    producer_id = serializers.IntegerField(required=True)
    product_id = serializers.IntegerField(required=False)
    model_version = serializers.CharField(max_length=64, required=False, allow_blank=False)
    image = serializers.ImageField(required=True)


class QualityPredictResponseSerializer(serializers.Serializer):
    """
    Validates the outbound ML payload, ensuring compliance with the DESD contract.
    """
    color_score = serializers.FloatField()
    size_score = serializers.FloatField()
    ripeness_score = serializers.FloatField()
    confidence = serializers.FloatField()
    predicted_class = serializers.CharField()

    overall_grade = serializers.CharField(required=True, allow_blank=False)
    
    # Optional but frequently returned fields
    class_probabilities = serializers.DictField(required=False)
    explanation_payload = serializers.DictField(required=False)
    transparency_refs = serializers.ListField(child=serializers.CharField(), required=False)
    model_version_used = serializers.CharField(required=False)
    inventory_action = serializers.DictField(required=False)  # Contains discount logic
