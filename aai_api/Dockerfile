FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY aai_api/requirements.txt /app/aai_api/requirements.txt
RUN pip install --no-cache-dir -r /app/aai_api/requirements.txt

COPY . /app

EXPOSE 8001

CMD ["python", "manage.py", "runserver", "0.0.0.0:8001"]
