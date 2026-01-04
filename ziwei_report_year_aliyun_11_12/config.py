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

    # ===== 应用基本配置 =====
    APP_NAME: str = "紫微斗数年度报告系统"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # ===== 数据库配置 =====
    # MySQL数据库配置
    DB_HOST: str = os.getenv("DB_HOST", "rm-bp1bho73kb4uas5xp.mysql.rds.aliyuncs.com")
    DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
    DB_USER: str = os.getenv("DB_USER", "haoyunka_root")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "5r4@ZeU8sBOo")
    DB_NAME: str = os.getenv("DB_NAME", "yingshi")
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

# 创建全局配置实例
config = Config()

if __name__ == "__main__":
    # 测试配置
    print(f"应用名称: {config.APP_NAME}")
