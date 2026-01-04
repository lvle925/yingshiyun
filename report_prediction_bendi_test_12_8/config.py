"""
项目配置文件
集中管理所有配置信息，支持环境变量覆盖
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """配置类，集中管理所有配置信息"""
    
    # ===== 数据库配置 =====
    # MySQL数据库配置
    DB_HOST: str = os.getenv("DB_HOST", "192.168.1.106")
    DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
    DB_USER: str = os.getenv("DB_USER", "root")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "bAm5b&mp")
    DB_NAME: str = os.getenv("DB_NAME", "xiaojili")
    DB_AUTOCOMMIT: bool = os.getenv("DB_AUTOCOMMIT", "True").lower() == "true"
    
    # 连接池配置
    DB_POOL_MIN_SIZE: int = int(os.getenv("DB_POOL_MIN_SIZE", "5"))
    DB_POOL_MAX_SIZE: int = int(os.getenv("DB_POOL_MAX_SIZE", "20"))
    DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "3600"))
    
    @classmethod
    def get_db_config(cls) -> Dict[str, Any]:
        """获取数据库配置字典"""
        return {
            'host': cls.DB_HOST,
            'port': cls.DB_PORT,
            'user': cls.DB_USER,
            'password': cls.DB_PASSWORD,
            'db': cls.DB_NAME,
            'autocommit': cls.DB_AUTOCOMMIT
        }
    
    # ===== Redis配置 =====
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://:h8v@vqekDS@localhost:6379/0")
    REDIS_POOL_MAX_CONNECTIONS: int = int(os.getenv("REDIS_POOL_MAX_CONNECTIONS", "20"))
    
    # ===== Prometheus监控配置 =====
    PROMETHEUS_METRICS_PATH: str = os.getenv("PROMETHEUS_METRICS_PATH", "/metrics")
    
    # 紫微API配置
    ZIWEI_API_URL: str = os.getenv("ZIWEI_API_URL", "http://192.168.1.102:3000/astro_with_option")
    ZIWEI_API_TIMEOUT_SECONDS: int = int(os.getenv("ZIWEI_API_TIMEOUT_SECONDS", "250"))
    

# 创建全局配置实例
config = Config()

