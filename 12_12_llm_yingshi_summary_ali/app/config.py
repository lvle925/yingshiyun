# config.py
import os
from dotenv import load_dotenv


load_dotenv()
# Redis配置

REDIS_URL = os.getenv("REDIS_URL", "redis://:h8v@vqekDS@192.168.1.101:6379/0")

# 会话历史保留条数（保留最近N轮对话，每轮包含1个用户消息+1个AI消息）
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))  # 默认保留最近3轮对话（6条消息）

# 会话数据过期时间（秒）
SESSION_TTL = int(os.getenv("SESSION_TTL", "86400"))  # 默认24小时

# 数据库配置（MySQL）
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "")
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))  # 连接池大小
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "5"))  # 最大溢出连接数