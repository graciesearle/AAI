# Task 2 and Task 3 Completion Runbook

- Training output is only partially automatic.
  - Automatic: model checkpoint file and learning-curve plot are written.
  - Not automatic: lifecycle registration and active model switch. You must upload and activate through Task 3.

## Expected Outcome

After following this runbook, you will have:

- a newly trained checkpoint under models/produce-quality/<version>/artifacts/model.pth
- a lifecycle-registered version created by Task 3 upload
- that version activated for Task 2 inference
- a verification response showing trained-model-inference-v1

## Preconditions

- Run commands from AAI repo root.
- Virtual environment is activated.
- Dataset exists and is readable.
- Django settings module is ai_service.settings.

## Step-by-Step

1. Train a new Task 2 model.
2. Detect the latest trained semantic version dynamically.
3. Build artifact path from detected version.
4. Upload the artifact to Task 3 with a generated version (no hardcoded version).
5. Activate the uploaded version.
6. Verify lifecycle state (active model).
7. Verify Task 2 inference mode is model-backed.
8. Run regression tests for submission evidence.

## Copy-Paste PowerShell Sequence

Paste this whole block from AAI repo root.

```powershell
$ErrorActionPreference = "Stop"

# 1) Train Task 2 model with auto versioning
python task2_quality/training_pipeline.py --model-version auto --dataset-dir "FruitAndVegetableDataset"

# 2) Detect latest trained semantic version dynamically
$trainedVersion = Get-ChildItem "models/produce-quality" -Directory |
  Where-Object { $_.Name -match '^\d+\.\d+\.\d+$' } |
  Sort-Object { [version]$_.Name } |
  Select-Object -Last 1 -ExpandProperty Name

if (-not $trainedVersion) {
  throw "No trained semantic version found under models/produce-quality"
}

Write-Host "Detected trained version: $trainedVersion"

# 3) Resolve trained artifact path
$artifactPath = Join-Path "models/produce-quality/$trainedVersion/artifacts" "model.pth"
if (-not (Test-Path $artifactPath)) {
  throw "Expected trained artifact missing: $artifactPath"
}

Write-Host "Artifact path: $artifactPath"

# 4) Generate upload version dynamically (no hardcoded version)
$uploadVersion = "qa-$([guid]::NewGuid().ToString('N').Substring(0,8))"
$env:TASK3_ARTIFACT_PATH = (Resolve-Path $artifactPath).Path
$env:TASK3_UPLOAD_VERSION = $uploadVersion

Write-Host "Upload lifecycle version: $uploadVersion"

# 5) Upload + activate through Task 3 API using Django test client
python -c "import json, os, django; os.environ.setdefault('DJANGO_SETTINGS_MODULE','ai_service.settings'); django.setup(); from django.conf import settings; settings.ALLOWED_HOSTS.append('testserver'); from django.test import Client; from django.core.files.uploadedfile import SimpleUploadedFile; c=Client(); p=os.environ['TASK3_ARTIFACT_PATH']; v=os.environ['TASK3_UPLOAD_VERSION']; data=open(p,'rb').read(); f=SimpleUploadedFile('model.pth', data, content_type='application/octet-stream'); up=c.post('/api/task3/models/upload/', {'model_name':'produce-quality','model_version':v,'framework':'pytorch','artifact':f}); print('upload_status', up.status_code); print('upload_body', up.content.decode()); act=c.post('/api/task3/models/activate/', {'model_name':'produce-quality','model_version':v}); print('activate_status', act.status_code); print('activate_body', act.content.decode())"

# 6) Verify active lifecycle version
python -c "import os, json, django; os.environ.setdefault('DJANGO_SETTINGS_MODULE','ai_service.settings'); django.setup(); from django.conf import settings; settings.ALLOWED_HOSTS.append('testserver'); from django.test import Client; c=Client(); r=c.get('/api/task3/models/'); print('list_status', r.status_code); body=r.json(); active=[x for x in body.get('results',[]) if x.get('model_name')=='produce-quality' and x.get('is_active')]; print('active', json.dumps(active[:1], indent=2))"

# 7) Verify Task 2 model-backed inference mode
python -c "import os, django; os.environ.setdefault('DJANGO_SETTINGS_MODULE','ai_service.settings'); django.setup(); from django.conf import settings; settings.ALLOWED_HOSTS.append('testserver'); from django.test import Client; from io import BytesIO; from PIL import Image; from django.core.files.uploadedfile import SimpleUploadedFile; c=Client(); b=BytesIO(); Image.new('RGB',(64,64),(120,180,60)).save(b,format='PNG'); b.seek(0); f=SimpleUploadedFile('sample.png', b.read(), content_type='image/png'); r=c.post('/api/task2/predict/', {'producer_id':1,'image':f}); print('predict_status', r.status_code); j=r.json(); print('model_version_used', j.get('model_version_used')); print('mode_note', (j.get('explanation_payload') or {}).get('note'))"

# 8) Run final regression tests
python manage.py test tests.test_smoke_api tests.test_task2_contract tests.test_task3_lifecycle_api
```

## How Populate Works

What populates automatically after training:

- models/produce-quality/<resolved_version>/artifacts/model.pth
- docs/task2/task2*learning_curves_produce-quality*<resolved_version>.png

What does not populate automatically:

- lifecycle registry entry for your new model
- active model assignment used by Task 2

Those are created by Task 3 upload and activate in the command sequence above.

## Success Criteria Checklist

- Training finished with no runtime error.
- Trained artifact exists.
- Upload API returns HTTP 201.
- Activate API returns HTTP 200.
- Task 3 list shows produce-quality with is_active true for upload version.
- Task 2 predict returns HTTP 200 and explanation_payload.note is trained-model-inference-v1.
- Regression test suite passes.

## If Something Fails

- Upload fails 400: verify artifact path exists and model_name/model_version were sent.
- Activate fails 400: ensure upload succeeded for the same version.
- Predict fails 503: missing or invalid manifest, or model inference artifact/load failure for selected version.
