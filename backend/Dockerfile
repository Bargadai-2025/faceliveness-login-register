# Build from repository root:  docker build -f backend/Dockerfile .
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8 weights so it doesn't fail on read-only/no-internet deployments
RUN wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt -O yolov8s.pt

COPY backend/ .

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
