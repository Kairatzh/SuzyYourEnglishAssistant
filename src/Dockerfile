FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip setuptools wheel

COPY ../requirements.txt .

RUN pip install --no-cache-dir torch==2.2.2 transformers==4.39.3 sentencepiece==0.1.99 datasets==2.18.0
RUN pip install --no-cache-dir -r requirements.txt

COPY .. .

ENV PYTHONUNBUFFERED=1
ENV PORT=8001
EXPOSE 8001
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
