# The deployed container runs app.py (the API server) only.
# ingest.py stays a script you run locally/manually to populate Qdrant --
# it's not a long-running service, so it doesn't belong in this image.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]