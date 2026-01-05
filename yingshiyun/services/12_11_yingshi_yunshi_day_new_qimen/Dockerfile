# Dockerfile

# ---- 第一阶段: 编译环境 (Builder) ----
FROM python:3.11-slim as builder

RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources
# 安装编译所需的依赖
RUN apt-get update && apt-get install -y gcc python3-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 复制所有项目文件
COPY . .

RUN find . -type f -name "*.so" -delete \
    && find . -type f -name "*.pyc" -delete \
    && find . -type d -name "__pycache__" -exec rm -r {} +

# 执行 Cython 编译，生成 .so 文件
# 这会把 app/processing.py 编译成 app/processing.cpython-311-x86_64-linux-gnu.so
RUN python setup.py build_ext --inplace


# ---- 第二阶段: 生产环境 (Final) ----
FROM python:3.11-slim

WORKDIR /app

# 只从编译环境中复制必要的运行时依赖
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/

COPY --from=builder /usr/local/bin/ /usr/local/bin/
# 复制编译好的 .so 文件和不需要编译的 .py 文件
# 我们不复制原始的 .py 源码文件，达到了加密的目的
COPY --from=builder /app/app/*.so /app/app/
COPY --from=builder /app/helper_libs/*.so /app/helper_libs/
COPY app/__init__.py app/__init__.py
COPY app/api.py app/api.py
COPY app/config.py app/config.py
COPY helper_libs/__init__.py helper_libs/__init__.py
COPY assets /app/assets
COPY api_main_attributes.py /app/api_main_attributes.py
COPY api_main_calendar.py /app/api_main_calendar.py
COPY api_main_day.py /app/api_main_day.py
COPY app/prompt_manager.py app/prompt_manager.py
COPY app/monitor.py app/monitor.py
COPY app/prompts.yaml app/prompts.yaml
COPY main.py /app/main.py
COPY .env /app/.env


# 暴露端口
EXPOSE 9001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9001"]