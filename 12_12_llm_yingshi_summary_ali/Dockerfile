
FROM python:3.11-slim as builder


RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources
# 容器内所有后续操作都将在此目录下进行
WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt


COPY ./app .


EXPOSE 8033

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8033"]
