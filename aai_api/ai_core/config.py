from dataclasses import dataclass
from pathlib import Path

from django.conf import settings


@dataclass(frozen=True)
class ServiceConfig:
    model_root: Path
    default_model_name: str
    default_model_version: str
    default_task_profile: str
    verbose_inference_logging: bool


def get_service_config() -> ServiceConfig:
    return ServiceConfig(
        model_root=Path(settings.MODEL_ROOT),
        default_model_name=settings.DEFAULT_MODEL_NAME,
        default_model_version=settings.DEFAULT_MODEL_VERSION,
        default_task_profile=settings.DEFAULT_TASK_PROFILE,
        verbose_inference_logging=settings.VERBOSE_INFERENCE_LOGGING,
    )
