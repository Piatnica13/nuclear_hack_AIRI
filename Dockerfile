FROM python:3.10-slim

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzstd1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем модель и код
COPY models/ ./models/
COPY codec/ ./codec/
COPY codec_config.json .
COPY inference.py .

ENTRYPOINT ["python", "inference.py"]
