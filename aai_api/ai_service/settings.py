from pathlib import Path
import os


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent

SECRET_KEY = env("DJANGO_SECRET_KEY", "dev-only-change-me")
DEBUG = str(env("DJANGO_DEBUG", "1")).lower() in {"1", "true", "yes"}
ALLOWED_HOSTS = [
    host.strip()
    for host in str(env("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,0.0.0.0,ai-service,host.docker.internal")).split(",")
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "aai_api.ai_core.apps.AiCoreConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "aai_api.ai_service.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    }
]

WSGI_APPLICATION = "aai_api.ai_service.wsgi.application"
ASGI_APPLICATION = "aai_api.ai_service.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
        "rest_framework.renderers.BrowsableAPIRenderer",
    ]
}

# Service-level AI runtime config (modular and environment-driven)
MODEL_ROOT = Path(str(env("MODEL_ROOT", str(REPO_ROOT / "models"))))
DEFAULT_MODEL_NAME = str(env("DEFAULT_MODEL_NAME", "produce-quality"))
DEFAULT_MODEL_VERSION = str(env("DEFAULT_MODEL_VERSION", "1.0.0"))
DEFAULT_TASK_PROFILE = str(env("DEFAULT_TASK_PROFILE", "task2_quality"))
VERBOSE_INFERENCE_LOGGING = str(env("VERBOSE_INFERENCE_LOGGING", "1")).lower() in {"1", "true", "yes"}
