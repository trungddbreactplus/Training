FROM python:3.11-slim

# Tranh Python buffer output (de xem log trong Docker)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Cai cac thu vien he thong can thiet cho torch va FlagEmbedding
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements truoc de tan dung Docker layer cache
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy toan bo code
COPY app.py .
EXPOSE 8501

# Healthcheck de docker-compose biet khi nao app ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]