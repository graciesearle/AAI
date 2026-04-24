"""
Task 3 serializers — Lifecycle management and interaction logging.

Moved from task2_3_4/task3_lifecycle/serializers.py to keep task folders
lightweight. All lifecycle and interaction serializers live here.
"""
from __future__ import annotations

from rest_framework import serializers


# ---------------------------------------------------------------------------
# Lifecycle serializers (moved from task3_lifecycle/)
# ---------------------------------------------------------------------------

class LifecycleModelUploadSerializer(serializers.Serializer):
    model_name = serializers.CharField(max_length=120, required=False)
    model_version = serializers.CharField(max_length=64, required=False)
    framework = serializers.CharField(max_length=64, required=False, allow_blank=True)
    task_profile = serializers.CharField(max_length=64, required=False, default="task2_quality")
    schema_version = serializers.CharField(max_length=64, required=False, default="task2-quality-v1")
    metrics = serializers.JSONField(required=False, default=dict)

    manifest_json = serializers.JSONField(required=False)
    artifact = serializers.FileField(required=False)
    checksum = serializers.CharField(max_length=128, required=False, allow_blank=False)

    def validate_metrics(self, value):
        if not isinstance(value, dict):
            raise serializers.ValidationError("metrics must be a JSON object")
        return value

    def validate(self, attrs):
        manifest = attrs.get("manifest_json")
        model_name = attrs.get("model_name")
        model_version = attrs.get("model_version")

        if manifest is not None and not isinstance(manifest, dict):
            raise serializers.ValidationError({"manifest_json": "manifest_json must be a JSON object"})

        if manifest is None and not attrs.get("artifact"):
            raise serializers.ValidationError(
                {"artifact": "artifact file is required when manifest_json is not provided"}
            )

        if manifest is not None:
            manifest_name = str(manifest.get("model_name", "")).strip()
            manifest_version = str(manifest.get("model_version", "")).strip()
            if model_name and manifest_name and model_name != manifest_name:
                raise serializers.ValidationError(
                    {"model_name": "model_name does not match manifest_json.model_name"}
                )
            if model_version and manifest_version and model_version != manifest_version:
                raise serializers.ValidationError(
                    {"model_version": "model_version does not match manifest_json.model_version"}
                )

        if not model_name and not (manifest and manifest.get("model_name")):
            raise serializers.ValidationError({"model_name": "model_name is required"})
        if not model_version and not (manifest and manifest.get("model_version")):
            raise serializers.ValidationError({"model_version": "model_version is required"})

        return attrs


class LifecycleModelActivateSerializer(serializers.Serializer):
    model_name = serializers.CharField(max_length=120)
    model_version = serializers.CharField(max_length=64)


class LifecycleModelRollbackSerializer(serializers.Serializer):
    model_name = serializers.CharField(max_length=120)
    target_model_version = serializers.CharField(max_length=64, required=False, allow_blank=False)


class LifecycleModelListSerializer(serializers.Serializer):
    model_name = serializers.CharField(max_length=120)
    model_version = serializers.CharField(max_length=64)
    task_profile = serializers.CharField(max_length=64, required=False, allow_blank=True)
    schema_version = serializers.CharField(max_length=64, required=False, allow_blank=True)
    framework = serializers.CharField(max_length=64, required=False, allow_blank=True)
    checksum = serializers.CharField(max_length=128, required=False, allow_blank=True)
    artifact_path = serializers.CharField(max_length=255, required=False, allow_blank=True)
    manifest_path = serializers.CharField(max_length=255, required=False, allow_blank=True)
    created_at = serializers.CharField(max_length=64, required=False, allow_blank=True)
    activated_at = serializers.CharField(max_length=64, required=False, allow_blank=True)
    source = serializers.CharField(max_length=64, required=False, allow_blank=True)
    is_active = serializers.BooleanField()


# ---------------------------------------------------------------------------
# Interaction / override serializers (new — for assessment compliance)
# ---------------------------------------------------------------------------

class InferenceLogSerializer(serializers.Serializer):
    """Read-only serializer for the InferenceLog model."""
    id = serializers.IntegerField(read_only=True)
    timestamp = serializers.DateTimeField(read_only=True)
    producer_id = serializers.IntegerField()
    product_id = serializers.IntegerField(allow_null=True, required=False)
    model_version = serializers.CharField()
    confidence = serializers.FloatField()
    color_score = serializers.FloatField()
    size_score = serializers.FloatField()
    ripeness_score = serializers.FloatField()
    predicted_grade = serializers.CharField()
    producer_accepted = serializers.BooleanField(allow_null=True, read_only=True)
    override_grade = serializers.CharField(read_only=True, allow_blank=True)


class ProducerOverrideSerializer(serializers.Serializer):
    """
    Accepts producer feedback on a prediction.
    This is the core of the retraining feedback loop: if a producer
    disagrees with the model's grade, they can override it. The AI
    engineer can then query these overrides to build a retraining dataset.
    """
    producer_accepted = serializers.BooleanField(
        help_text="True if the producer accepted the AI-assigned grade, False if they rejected it."
    )
    override_grade = serializers.CharField(
        max_length=4, required=False, allow_blank=True, default="",
        help_text="The grade the producer believes is correct (A, B, or C). Only required when producer_accepted is False."
    )

    def validate(self, attrs):
        producer_accepted = attrs["producer_accepted"]
        override_grade = str(attrs.get("override_grade", "")).strip()

        if producer_accepted and override_grade:
            raise serializers.ValidationError(
                {"override_grade": "override_grade must be empty when producer_accepted is True."}
            )

        if not producer_accepted and not override_grade:
            raise serializers.ValidationError(
                {"override_grade": "override_grade is required when producer_accepted is False."}
            )

        attrs["override_grade"] = override_grade
        return attrs
