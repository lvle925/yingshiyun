FROM python:3.11-slim AS builder

RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources

WORKDIR /app

COPY requirements.txt .

RUN pip install --prefix=/app/python_packages -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /app/python_packages /usr/local

COPY main.py config.py query_intent.py user_info_extractor.py db_query.py validation_rules.py session_manager.py llm_response.py ./
COPY app ./app
COPY prompts ./prompts
COPY assets ./assets
COPY .env .env


RUN mkdir -p /app/logs

EXPOSE 8055

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8055"]

