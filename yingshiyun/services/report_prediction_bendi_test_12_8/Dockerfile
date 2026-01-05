
FROM python:3.11-slim-bookworm as builder

WORKDIR /app

RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update && apt-get install -y build-essential python3-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
COPY . .

FROM python:3.11-slim-bookworm
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/api_main.py .
COPY --from=builder /app/config.py .
COPY --from=builder /app .

RUN find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
RUN find . -type f -name "*.pyc" -delete 2>/dev/null || true

ENV ENVIRONMENT=production
ENV DEBUG=false
ENV LOG_LEVEL=INFO
ENV HOST=0.0.0.0
ENV PORT=7079


ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
EXPOSE 7079

CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "7079"]