FROM python:3.13-slim

# libgomp1 = OpenMP runtime required by xgboost / scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code, model and data
COPY . .

# Hugging Face Spaces serves the container on port 7860
ENV PORT=7860
EXPOSE 7860

# Serve via gunicorn (honours $PORT if the platform overrides it)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 2 --timeout 120 app:app"]
