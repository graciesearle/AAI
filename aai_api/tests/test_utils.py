from __future__ import annotations

from io import BytesIO

from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image


def make_uploaded_png(
    *,
    color: tuple[int, int, int],
    name: str = "sample.png",
    size: tuple[int, int] = (24, 24),
) -> SimpleUploadedFile:
    image_data = BytesIO()
    image = Image.new("RGB", size, color=color)
    image.save(image_data, format="PNG")
    image_data.seek(0)
    return SimpleUploadedFile(
        name=name,
        content=image_data.read(),
        content_type="image/png",
    )
