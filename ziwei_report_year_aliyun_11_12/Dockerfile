# =====================================================================
# Stage 1: Builder - 在这里我们安装依赖并编译代码
# 我们使用一个包含完整编译工具链的镜像作为基础
# =====================================================================
FROM python:3.11-slim-bookworm as builder

# 设置工作目录
WORKDIR /app

# [解释]: 安装编译 Cython 扩展所必需的系统级依赖
# build-essential 包含了 gcc 等 C 编译器
# python3-dev 包含了编译 Python C 扩展所需的头文件

RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources


RUN apt-get update && apt-get install -y build-essential python3-dev && rm -rf /var/lib/apt/lists/*

# [解释]: 复制 requirements.txt 文件到镜像中
COPY requirements.txt .

# [解释]: 根据 requirements.txt 安装所有 Python 依赖
# 我们也在这里安装 cython
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
#RUN pip install --no-cache-dir cython

# [解释]: 将您本地项目目录下的所有文件 (除了被 .dockerignore 忽略的)
# 复制到镜像的 /app 目录下。这里的 `.` 代表 Dockerfile 所在的目录，
# 第二个 `.` 代表镜像中的当前工作目录 (`/app`)。
COPY . .

# [解释]: 编译步骤（如果存在setup.py文件的话）
# 注意：当前项目可能不需要编译，如果不需要可以跳过此步骤
# RUN python3 setup.py build_ext --inplace


# =====================================================================
# Stage 2: Final - 这是最终的、轻量级的运行镜像
# =====================================================================
# 我们再次使用 slim 镜像，因为它比包含完整编译工具的镜像小得多
FROM python:3.11-slim-bookworm

WORKDIR /app

# [解释]: 从第一阶段 (builder) 复制已经安装好的 Python 依赖库
# 这样最终镜像就不需要再安装一遍，也不需要包含编译工具
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY --from=builder /usr/local/bin /usr/local/bin
# --- 复制编译后的核心代码 ---

# [解释]: 复制主入口和配置文件，这两个我们保留为明文 .py 文件
COPY --from=builder /app/api_main.py .
COPY --from=builder /app/config.py .

# [解释]: 复制所有编译好的 .so 文件和包结构 (__init__.py)
# 我们使用 find 和 xargs 来智能地复制，避免写死 .so 的具体名字
# 这会自动处理所有子目录
COPY --from=builder /app .

# [解释]: 清理缓存文件，保留所有.py源文件（因为本项目不需要编译）
RUN find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
RUN find . -type f -name "*.pyc" -delete 2>/dev/null || true


# --- 设置环境变量 ---
# [解释]: 设置生产环境的默认配置
# 这些变量可以在运行时通过 docker run -e 参数覆盖
ENV ENVIRONMENT=production
ENV DEBUG=false
ENV LOG_LEVEL=INFO
ENV HOST=0.0.0.0
ENV PORT=7077

# 设置Python运行环境
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# 可以在运行时覆盖的敏感配置
# ENV APP_SECRET=""
# ENV DB_PASSWORD=""
# ENV REDIS_URL=""

# [解释]: 告诉 Docker，容器内的应用程序将会监听 8008 端口
EXPOSE 7077

# [解释]: 这是容器启动时要执行的默认命令。
# 它会使用 uvicorn 来运行 api_main.py 文件中的 app 对象。
CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "7077"]