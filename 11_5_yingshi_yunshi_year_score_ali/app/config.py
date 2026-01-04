import os
from dotenv import load_dotenv

load_dotenv()

# Astro API 配置
ASTRO_API_URL = os.getenv("ASTRO_API_URL")
