
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app


RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources


RUN apt-get update && apt-get install -y build-essential python3-dev && rm -rf /var/lib/apt/lists/*

# [解释]: 复制 requirements.txt 文件到镜像中
COPY requirements.txt .


RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

COPY . .

# 临时禁用 Cython 编译，使用纯 Python 运行以避免内存损坏问题
# RUN python3 setup.py build_ext --inplace


FROM python:3.11-slim-bookworm

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY --from=builder /usr/local/bin /usr/local/bin

# 复制所有必需的 Python 模块和目录
COPY --from=builder /app/api_main.py .
COPY --from=builder /app/config.py .
COPY --from=builder /app/models.py .
COPY --from=builder /app/monitoring.py .
COPY --from=builder /app/__init__.py .


COPY --from=builder /app/.env ./

# 复制必需的目录（包含所有子文件和 __init__.py）
COPY --from=builder /app/security ./security
COPY --from=builder /app/services ./services
COPY --from=builder /app/clients ./clients
COPY --from=builder /app/database ./database

# 复制提示词模板目录（包含所有 XML 和 J2 模板文件）
COPY --from=builder /app/prompts ./prompts

# 复制其他工具文件（如果 api_main.py 需要导入它们）
# 注意：由于禁用了 Cython 编译，需要确保所有 Python 源文件都被复制
COPY --from=builder /app/utils.py .
COPY --from=builder /app/tiangan_function.py .
COPY --from=builder /app/ziwei_ai_function.py .
COPY --from=builder /app/liushifunction.py .
COPY --from=builder /app/tokenizer.py .
COPY --from=builder /app/prompt_logic.py .

# 确保复制所有 Python 源文件（因为不再使用编译后的 .so 文件）
COPY --from=builder /app/*.py ./

# 清理不需要的文件
RUN find . -type f -name "*.c" -delete || true
RUN find . -type f -name "setup.py" -delete || true
RUN find . -type f -name "test.py" -delete || true
RUN find . -type d -name "__pycache__" -exec rm -rf {} + || true
RUN find . -type d -name "*.egg-info" -exec rm -rf {} + || true



EXPOSE 8044

CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8044"]