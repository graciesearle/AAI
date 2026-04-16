from __future__ import annotations

import json

from rest_framework import serializers


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
