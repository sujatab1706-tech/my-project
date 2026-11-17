# Container for the compliance API.
# Build:  docker build -t image-compliance .
# Run:    docker run -p 8000:8000 image-compliance   -> open http://127.0.0.1:8000/docs
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2

# Install packages first: Docker caches this layer, so code changes rebuild fast.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/rules.py src/inference.py src/
COPY api/ api/
COPY models/ models/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
