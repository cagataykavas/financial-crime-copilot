FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FC_DATABASE_PATH=/data/financial_crime.db

WORKDIR /app
COPY pyproject.toml ./
COPY copilot.py ./
COPY service ./service
RUN pip install --no-cache-dir .

RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8000

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
