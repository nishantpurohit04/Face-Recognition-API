FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download ArcFace and MTCNN weights at build time
# Using wget directly — more reliable than triggering via Python
RUN mkdir -p /root/.deepface/weights

RUN wget -q --timeout=300 --tries=3 \
    "https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5" \
    -O /root/.deepface/weights/arcface_weights.h5

RUN wget -q --timeout=120 --tries=3 \
    "https://github.com/ipazc/mtcnn/releases/download/v0.1.0/mtcnn_weights.npy" \
    -O /root/.deepface/weights/mtcnn_weights.npy 2>/dev/null || true

COPY app/main.py .
COPY frontend/ ./frontend/

RUN mkdir -p known_faces

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]