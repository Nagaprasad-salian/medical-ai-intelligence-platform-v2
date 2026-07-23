FROM python:3.11-slim

WORKDIR /app

# System deps needed by opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_PATH=/app/saved_models/best_model.pt
ENV GRADCAM_DIR=/app/static/gradcam
ENV DATABASE_URL=sqlite:////app/db/medical_ai.db

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
