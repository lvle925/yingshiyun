
FROM python:3.11-slim-bookworm as builder

RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources
RUN apt-get update && apt-get install -y gcc python3-dev build-essential

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

RUN python3 setup.py build_ext --inplace


FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY --from=builder /app/main.py .
COPY --from=builder /app/*.csv ./

COPY --from=builder /app/session_manager.py .
COPY --from=builder /app/config.py . 
COPY --from=builder /app/.env . 
COPY --from=builder /app/queryIntent.py .
COPY --from=builder /app/validation_rules.py . 
COPY --from=builder /app/app ./app

COPY --from=builder /app/*.so ./

COPY --from=builder /app/prompts ./prompts

EXPOSE 8022

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8022"]
