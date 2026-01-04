# config.py
import os
from dotenv import load_dotenv


load_dotenv()

# Redis配置
REDIS_URL = os.getenv("REDIS_URL", "redis://:h8v@vqekDS@192.168.1.101:6379/0")

# 会话历史保留条数
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))  # 默认保留3轮对话

# 会话数据过期时间
SESSION_TTL = int(os.getenv("SESSION_TTL", "86400"))  # 24小时

VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL", "http://192.168.1.101:6002/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen3-30B-A3B-Instruct-2507")

API_KEY = os.getenv("API_KEY", "sk-81d4dbe056f94030998f0639f709bff4")

DB_CONFIG = {
    'user': os.getenv("DB_USER", "root"),
    'password': os.getenv("DB_PASSWORD", "bAm5b&mp"),
    "port":int(os.getenv("DB_PORT", "3306")),
    'host': os.getenv("DB_HOST", "192.168.1.106"),
    'db': os.getenv("DB_DATABASE", "yingshi"),
    'autocommit': True
}