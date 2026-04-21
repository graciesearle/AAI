from rest_framework import serializers


class ExplainRequestSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)
    model_name = serializers.CharField(max_length=120, required=False)
    model_version = serializers.CharField(max_length=64, required=False)
    context = serializers.DictField(required=False)


class ExplainResponseSerializer(serializers.Serializer):
    explanation_payload = serializers.DictField()
    transparency_refs = serializers.ListField(child=serializers.CharField())
    model_version_used = serializers.CharField()
    schema_version = serializers.CharField()
