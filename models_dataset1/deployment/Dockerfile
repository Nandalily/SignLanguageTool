FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY models_dataset1/deployment/requirements.txt /tmp/deployment-requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/deployment-requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "models_dataset1.deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]