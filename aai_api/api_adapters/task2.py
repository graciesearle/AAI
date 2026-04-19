import time
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from aai_api.ai_core.config import get_service_config
from aai_api.ai_core.lifecycle import get_active_model_version
from aai_api.ai_core.manifest import ManifestError, get_bundle_root, load_manifest
from aai_api.ai_core.models import InferenceLog

from aai_api.ai_core.task2_runtime import run_quality_inference
from aai_api.api_adapters.task2_serializers import (
    QualityPredictRequestSerializer,
    QualityPredictResponseSerializer,
)


class QualityPredictAdapterView(APIView):
    def post(self, request):
        start_time = time.time()
        
        serializer = QualityPredictRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        cfg = get_service_config()
        model_name = serializer.validated_data.get("model_name") or cfg.default_model_name
        requested_model_version = serializer.validated_data.get("model_version")
        model_version = (
            requested_model_version
            or get_active_model_version(cfg.model_root, model_name)
            or cfg.default_model_version
        )

        try:
            manifest = load_manifest(cfg.model_root, model_name, model_version)
        except ManifestError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        try:
            artifact_item = next((a for a in manifest.get("artifacts", []) if a.get("type") == "model_weights"), None)
            if not artifact_item:
                raise ValueError("Manifest missing 'model_weights' artifact")

            checkpoint_path = get_bundle_root(cfg.model_root, model_name, model_version) / artifact_item["path"]

            payload = run_quality_inference(
                image_file=serializer.validated_data["image"],
                checkpoint_path=checkpoint_path,
            )
            
            latency_ms = (time.time() - start_time) * 1000.0
            
            if cfg.verbose_inference_logging:
                print(f"\n[AAI PIPELINE LOG] task2 quality inference")
                print(f"  ├─ Producer    : {serializer.validated_data['producer_id']}")
                print(f"  ├─ Model Name  : {model_name}")
                print(f"  ├─ Version     : {model_version}")
                print(f"  ├─ Checkpoint  : {checkpoint_path}")
                print(f"  ├─ Predicted   : Grade {payload['overall_grade']} ({payload['normalized_label']})")
                print(f"  └─ Latency     : {latency_ms:.2f} ms\n")
            
            # process_prediction returns 'input_confidence' (normalised 0-1);
            # DESD expects 'confidence' as a percentage 0-100.
            confidence_pct = round(payload["input_confidence"] * 100.0, 2)
            quality_scores = payload["quality_scores"]
            predicted_class = payload["normalized_label"]
            overall_grade = payload["overall_grade"]

            response_data = {
                "color_score": quality_scores["colour"],
                "size_score": quality_scores["size"],
                "ripeness_score": quality_scores["ripeness"],
                "confidence": confidence_pct,
                "predicted_class": predicted_class,
                "overall_grade": overall_grade,
                "model_name_used": model_name,
                "model_version_used": model_version,
                "explanation_payload": payload.get("explanation_payload", {}),
                "inventory_action": payload.get("inventory_action", {}),
                "latency_ms": latency_ms,
            }

            # Academically crucial: Recording the interaction telemetry
            InferenceLog.objects.create(
                producer_id=serializer.validated_data["producer_id"],
                product_id=serializer.validated_data.get("product_id"),
                model_version=model_version,
                confidence=confidence_pct,
                color_score=quality_scores["colour"],
                size_score=quality_scores["size"],
                ripeness_score=quality_scores["ripeness"],
                predicted_grade=overall_grade
            )

        except Exception as exc:
            return Response(
                {"detail": f"Model inference failed for '{model_name}/{model_version}': {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        output = QualityPredictResponseSerializer(response_data)
        return Response(output.data, status=status.HTTP_200_OK)
