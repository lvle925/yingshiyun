import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
# 这使得我们可以在开发环境中使用 .env 文件，在 Docker 中使用 --env-file
load_dotenv()

# VLLM 配置
VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "qwen3-next-80b-a3b-instruct")
API_KEY = os.getenv("API_KEY", "sk-54ccd1d837d44d25aafcb92034b95b9b")

# Astro API 配置
ASTRO_API_URL = os.getenv("ASTRO_API_URL", "http://192.168.1.102:3000/astro_with_option")

DB_CONFIG = {
    'user': os.getenv("DB_USER", "root"),
    'password': os.getenv("DB_PASSWORD", "bAm5b&mp"),
    'host': os.getenv("DB_HOST", "192.168.1.106"),
    'port': int(os.getenv("DB_PORT", 3306)),
    'db': os.getenv("DB_NAME", "yingshi"),
    'autocommit': True
}