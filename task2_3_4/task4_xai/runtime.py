from typing import Any
import io 
import base64
from PIL import Image
from .aai_explainer import ProduceXAI

def _pil_to_base64(img: Image.Image) -> str:
    """Helper to convert PIL image to base64 for API transmission."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_explanation(*, image_file, checkpoint_path, model_name, model_version, context=None, manifest=None) -> dict[str, Any]:
    """
    The entry point called by the Django API. It takes the uploaded image and the path to the active Task 2 model.
    """
    if manifest and manifest.get("task_profile") not in ["task2_quality", "task4_xai"]:
        raise ValueError("Selected model bundle is not task4_xai profile")
    
    xai = ProduceXAI(model_path=checkpoint_path)
    report_img = xai.generate_master_audit_report(image_file)

    return {
        "model_version_used": model_version,
        "schema_version": "task4-xai-v1",
        "transparency_refs": ["xai://master-audit-report"],
        "explanation_payload": {
            "report_image_base64": _pil_to_base64(report_img),
            "note": f"Visual transparency report for {model_name} v{model_version}"
        },
    }
