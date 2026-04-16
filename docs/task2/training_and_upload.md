# Task 2 Training and Upload Guide

This guide describes how to train a real Task 2 model checkpoint and upload/activate it through Task 3 lifecycle APIs.

## What Is Dynamic Now

`task2_quality/training_pipeline.py` now supports dynamic output resolution.

- `model_version="auto"` chooses the next patch version under `models/produce-quality/`.
- Example:
  - Existing highest version: `1.0.0`
  - Next training run with auto version writes to `1.0.1`
- Model output path is resolved as:
  - `models/<model_name>/<resolved_version>/artifacts/model.pth`
- Plot output path is resolved as:
  - `docs/task2/task2_learning_curves_<model_name>_<resolved_version>.png`

No output is written to `local/`.

## Configure Before Training

Edit `CONFIG` in `task2_quality/training_pipeline.py`:

- `dataset_dir`: folder containing class subdirectories.
- `model_name`: normally `produce-quality`.
- `model_version`: use `auto` or set explicitly (e.g., `1.1.0`).
- `no_pretrained`: set `False` to use ImageNet initialization.

## Train the Model

From repo root (`AAI`):

```powershell
c:/Users/jacob/Documents/Projects/UNI/.venv/Scripts/python.exe task2_quality/training_pipeline.py
```

The script prints:

- Resolved model version
- Model output path
- Plot output path

## Option A: Activate by Folder/Manifest Discovery (No Upload)

If the checkpoint was written into `models/produce-quality/<version>/artifacts/model.pth`, activate directly:

```powershell
c:/Users/jacob/Documents/Projects/UNI/.venv/Scripts/python.exe -c "import os,django; os.environ.setdefault('DJANGO_SETTINGS_MODULE','ai_service.settings'); django.setup(); from django.conf import settings; settings.ALLOWED_HOSTS.append('testserver'); from django.test import Client; c=Client(); v='1.0.1'; print(c.post('/api/task3/models/activate/', {'model_name':'produce-quality','model_version':v}).status_code)"
```

This works when a valid manifest and artifact exist for that version.

## Option B: Upload Through Task 3 API (AI Engineer Flow)

You can upload a checkpoint explicitly and then activate it.

PowerShell example:

```powershell
c:/Users/jacob/Documents/Projects/UNI/.venv/Scripts/python.exe -c "import os,django,uuid; os.environ.setdefault('DJANGO_SETTINGS_MODULE','ai_service.settings'); django.setup(); from django.conf import settings; settings.ALLOWED_HOSTS.append('testserver'); from django.test import Client; from django.core.files.uploadedfile import SimpleUploadedFile; c=Client(); v='qa-'+uuid.uuid4().hex[:8]; data=open('models/produce-quality/1.0.1/artifacts/model.pth','rb').read(); f=SimpleUploadedFile('model.pth', data, content_type='application/octet-stream'); up=c.post('/api/task3/models/upload/', {'model_name':'produce-quality','model_version':v,'framework':'pytorch','artifact':f}); print('upload', up.status_code); act=c.post('/api/task3/models/activate/', {'model_name':'produce-quality','model_version':v}); print('activate', act.status_code)"
```

## Verify Inference Mode

After activation, verify Task 2 output note is model-backed:

- Expected model mode note: `trained-model-inference-v1`
- Fallback note: `image-signal-fallback-v1`

Quick check:

```powershell
c:/Users/jacob/Documents/Projects/UNI/.venv/Scripts/python.exe -c "import os,django; os.environ.setdefault('DJANGO_SETTINGS_MODULE','ai_service.settings'); django.setup(); from django.conf import settings; settings.ALLOWED_HOSTS.append('testserver'); from django.test import Client; from io import BytesIO; from PIL import Image; from django.core.files.uploadedfile import SimpleUploadedFile; c=Client(); b=BytesIO(); Image.new('RGB',(32,32),(120,180,60)).save(b,format='PNG'); b.seek(0); f=SimpleUploadedFile('sample.png', b.read(), content_type='image/png'); r=c.post('/api/task2/predict/', {'producer_id':1,'image':f}); print('status', r.status_code); print('note', r.json().get('explanation_payload',{}).get('note'))"
```
