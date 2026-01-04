# -*- coding: utf-8 -*-
# 这是一个基于 FastAPI 的 LLM 调用服务，支持并发处理请求并与 VLLM 交互。
# 
# 新增功能（2025-10-13更新）：
#   - 违法内容拦截：检测并拒绝回答违法相关问题（破解、盗取、诈骗等）
#   - 历史事件拒绝：对过去时间的运势查询进行拦截，仅推演未来
#   - 乱码智能引导：检测无意义输入并提供运势问题示例引导
#   - 本月运势优化：自动将"本月"查询转换为当前阴历月份进行推演（通过prompt引导大模型）
#   - 知识问答分类：新增knowledge_question意图，路由专业术语解释到紫微服务
#   - 重大决策保护：拦截医疗、法律、财务等重大事项的时间选择问题
#   - 未指定时间处理：当用户询问运势未给定时间时，大模型将分析一生整体运势（通过prompt引导）
#   - 五行分析强化：通过prompt指导大模型准确分析五行信息，避免错误
#   - 未来伴侣查询优化：大模型只分析特质，明确告知无法预测姓氏（通过prompt引导）
# 
# 功能历史：
#   - 使用 Pydantic 模型进行请求体验证，包括可选的卡牌编号池
#   - 实现基于 HMAC-SHA256 的请求签名验证
#   - 路由逻辑：根据用户提问意图决定调用紫微或智慧卡服务
#   - 择时问题专门路由到智慧卡服务

import asyncio
import requests
import os
import random
import pandas as pd
import json
import time
import hmac
import hashlib
import logging
import ast
import sys
import calendar
import re
import uuid
import aiohttp
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any, AsyncGenerator, Literal
# from fastapi import FastAPI, HTTPException, status
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator, ValidationError
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# 数据库连接池（可选，如果配置了数据库则初始化）
try:
    import aiomysql
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

from config import (
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_POOL_SIZE, DB_MAX_OVERFLOW
)

from validation_rules import (
    detect_illegal_content,
    is_gibberish,
    detect_critical_time_selection,
    detect_sensitive_political_content,
    detect_finance_investment_query,
    detect_age_inquiry,
    remove_birth_info,
    detect_reincarnation_query,
)

# 导入session_manager
from session_manager import (
    initialize_session_manager, 
    close_session_manager,
    get_session_history,
    get_last_intent,
    store_last_intent
)

from queryIntent import (
    answer_knowledge_question,
    classify_query_intent_with_llm,
    extract_qimen_with_llm,
    extract_time_range_with_llm,
    get_all_tags_from_db_async,
)

from monitor import StepMonitor, log_step

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - 说: %(message)s')
logger = logging.getLogger(__name__)

# --- 从环境变量加载配置 ---
load_dotenv() # 在本地开发时加载 .env 文件

# 下游服务地址

url_ziwei = os.getenv("URL_ZIWEI", "http://llm_yingshi_ziwei:7008/chat_yingshis")
url_leipai = os.getenv("URL_LEIPAI", "http://llm_yingshi_leinuo:7022/chat_endpoints")
url_qimen = os.getenv("URL_QIMEN", "http://llm_yingshi_qimen:8055/qimen")

# 应用密钥
# 将密钥从字典转换为直接从环境变量获取
APP_SECRET_YINGSHI = os.getenv("APP_SECRET_YINGSHI")
APP_SECRET_TEST = os.getenv("APP_SECRET_TEST")
HOROSCOPE_SECRET_KEY = os.getenv("HOROSCOPE_SECRET_KEY")



LLM_MAX_RETRIES = 5
LLM_RETRY_DELAY_SECONDS = 5.0
LLM_REQUEST_TIMEOUT_SECONDS = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", 1800.0))
async_aiohttp_client: Optional[aiohttp.ClientSession] = None
VLLM_REQUEST_TIMEOUT_SECONDS = 500
# HOROSCOPE_DAILY_STREAM_API_URL = "http://10.210.254.69:8005/horoscope/daily-stream"
HOROSCOPE_APP_ID = "your_test_appid"
HOROSCOPE_SECRET_KEY = "your_super_secret_key_that_matches_the_server"


if not APP_SECRET_YINGSHI:
    raise ValueError("关键环境变量 APP_SECRET_YINGSHI 未设置！")


async_aiohttp_client: Optional[aiohttp.ClientSession] = None
vllm_semaphore: Optional[asyncio.Semaphore] = None

APP_SECRETS: Dict[str, str] = {
    "yingshi_appid": "zhongzhoullm",
    "test_app": "test_secret_key",
    HOROSCOPE_APP_ID: HOROSCOPE_SECRET_KEY
}

renomann_cards_df: Optional[pd.DataFrame] = None
renomann_meanings_df: Optional[pd.DataFrame] = None
app = FastAPI()
VLLM_CONCURRENT_LIMIT = 100
vllm_semaphore: Optional[asyncio.Semaphore] = None
next_request_id_counter = 0

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 或明确写上 "http://192.168.1.101:5500"
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic 请求模型 (保持不变) ---
class UnifiedChatRequest(BaseModel):
    appid: str = Field(..., description="应用ID")
    prompt: str = Field(..., description="用户的问题，将用于生成LLM提示词")
    format: str = Field("json", description="响应格式，默认为json")
    ftime: int = Field(..., description="时间戳 (整数)，用于签名验证")
    sign: str = Field(..., description="请求签名，用于验证请求完整性")
    hl_ymd: Optional[str] = Field(None, description="可选的日期参数")
    card_number_pool: Optional[List[int]] = Field(
        None,
        description="可选的卡牌编号列表，将从这个列表中随机抽取3个数字作为卡牌编号。如果未提供、列表无效，或提供了无法解析的字符串，则从所有可用卡牌中抽取。"
    )
    session_id: Optional[str] = Field(None, description="会话 ID，用于路由到接口2")

    @validator('card_number_pool', pre=True, always=True)
    def parse_and_validate_card_number_pool(cls, v):
        # (此验证器逻辑保持不变)
        if v is None: return None
        if isinstance(v, str):
            try:
                v = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                raise ValueError("输入字符串不是有效的列表字面量。")
        if not isinstance(v, list): raise ValueError('必须是列表或表示列表的有效字符串。')
        if len(v) < 3: raise ValueError('必须包含至少 3 个数字')
        if not all(isinstance(i, int) for i in v): raise ValueError('必须只包含整数')
        return v



def smart_normalize_punctuation(text: str) -> str:
    """
    智能标点符号标准化：保护出生信息格式，只对问题部分进行标准化。
    """
    # 先检查是否包含出生信息
    birth_info_pattern = re.compile(r"(公历|农历)?\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+(男|女)")
    birth_match = birth_info_pattern.search(text)
    
    if not birth_match:
        # 没有出生信息，直接标准化整个文本
        return normalize_punctuation_simple(text)
    
    # 有出生信息，分离处理
    birth_info_end = birth_match.end()
    birth_part = text[:birth_info_end]  # 出生信息部分，保持原样
    question_part = text[birth_info_end:].strip()  # 问题部分，进行标准化
    
    # 只对问题部分进行标准化
    normalized_question = normalize_punctuation_simple(question_part)
    
    # 重新组合
    result = birth_part + " " + normalized_question if question_part else birth_part
    
    # logger.debug(f"智能符号标准化:")
    # logger.debug(f"  原文: '{text}'")
    # logger.debug(f"  出生信息部分: '{birth_part}'")
    # logger.debug(f"  问题部分: '{question_part}' -> '{normalized_question}'")
    # logger.debug(f"  结果: '{result}'")
    
    return result


def normalize_punctuation_simple(text: str) -> str:
    """
    简单的标点符号标准化，将英文标点转换为中文标点。
    """
    # 常见的英文标点转中文标点映射
    punctuation_map = {
        '?': '？',
        '!': '！',
        ',': '，',
        ';': '；',
        # 注意：不转换冒号，因为出生时间需要用英文冒号
        '(': '（',
        ')': '）',
        '[': '［',
        ']': '］',
        '{': '｛',
        '}': '｝',
        '"': '"',  # 转换为中文双引号
        "'": "'",  # 转换为中文单引号
        '.': '。'  # 句号转换（但要小心数字中的小数点）
    }
    
    normalized_text = text
    for en_punct, cn_punct in punctuation_map.items():
        # 对于句号，避免替换数字中的小数点
        if en_punct == '.':
            # 只替换句末的点，不替换数字中的小数点
            normalized_text = re.sub(r'\.(?!\d)', cn_punct, normalized_text)
        else:
            normalized_text = normalized_text.replace(en_punct, cn_punct)
    
    return normalized_text


def contains_formatted_birth_info(prompt: str) -> bool:
    """
    使用正则表达式检查用户的提问中是否包含特定格式的出生信息。
    例如: "公历 2025-08-06 19:00:00 男" 或 "农历 1990-01-15 08:30:00 女".
    """
    # 正则表达式匹配 "公历/农历" (可选) + YYYY-MM-DD HH:MM:SS + "男/女"
    pattern = re.compile(r"(公历|农历)?\s*\d{4}-\d{1,2}-\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+(男|女)")
    return bool(pattern.search(prompt))


def normalize_vague_time_expressions(cleaned_question: str) -> str:
    """
    在第一层意图分类之前，将用户问题中的模糊时间表述替换为具体时间表述。
    
    需求变更：将模糊时间直接替换为“从YYYY年MM月DD日到YYYY年MM月DD日”或包含时间的具体范围，按当前日期动态计算。
    """
    now = datetime.now()
    today = now.date()

    # ---------- 全局模糊时间词数量检查：出现两个及以上直接跳过规范化 ----------
    vague_tokens = [
        "将来", "未来", "近", "这", "后", "后面",
        "近期", "这个阶段", "当前阶段", "现阶段", "当下阶段",
        "接下来", "这段时间", "最近",
        "下个阶段", "下阶段", "下一阶段", "下下阶段",
        "这两天", "马上",
        "年前", "年内",
        "下个月",
        "今年", "明年", "后年", "去年", "前年",
        "下一年", "下年", "后一年",
        "下半年", "上半年",
        "年", "月", "日", "时", "号","天"
    ]

    vague_count = 0
    for token in vague_tokens:
        vague_count += len(re.findall(re.escape(token), cleaned_question))

    # 如果上述模糊/时间相关词整体出现次数 >= 2，则认为时间结构较复杂，避免误改，直接返回原问题
    if vague_count >= 2:
        return cleaned_question

    # ---------- 绝对时间复杂度检查：出现多个或多层级时间时直接跳过规范化 ----------
    # 规则：
    # 1) 如果同时出现“年+月”或“月+日”，为了避免破坏用户明确给定的时间点，直接返回原问题；
    # 2) 如果出现多个“年”或多个“月”或多个“日”（代表有时间范围或多时间点），同样直接返回原问题。

    # 年：包括 2025年 / 1999年 以及 今年/明年/后年/去年/前年
    year_matches = []
    year_matches += re.findall(r'\d{4}年', cleaned_question)
    year_matches += re.findall(r'今年|明年|后年|去年|前年', cleaned_question)

    # 月：只统计带数字的“X月”，不区分是否有“份”
    month_matches = re.findall(r'\d{1,2}月', cleaned_question)

    # 日：统计“X日/号/天”
    day_matches = re.findall(r'\d{1,2}(日|号|天)', cleaned_question)

    year_count = len(year_matches)
    month_count = len(month_matches)
    day_count = len(day_matches)

    has_year_and_month = year_count >= 1 and month_count >= 1
    has_month_and_day = month_count >= 1 and day_count >= 1
    has_multi_year = year_count > 1
    has_multi_month = month_count > 1
    has_multi_day = day_count > 1

    if has_year_and_month or has_month_and_day or has_multi_year or has_multi_month or has_multi_day:
        # 出现明确的复杂绝对时间表达（如“明年3月”“明年3月3日”“2023年到2025年”“3月到5月”等），
        # 为避免误改，直接返回原始问题。
        return cleaned_question

    def fmt_d(d: date) -> str:
        return f"{d.year}年{d.month}月{d.day}日"

    def fmt_dt(dt: datetime) -> str:
        return f"{dt.year}年{dt.month}月{dt.day}日 {dt.hour:02d}:{dt.minute:02d}"

    def fmt_y(year: int) -> str:
        """格式化为“YYYY年”"""
        return f"{year}年"

    def fmt_ym(year: int, month: int) -> str:
        """格式化为“YYYY年M月”"""
        return f"{year}年{month}月"

    def month_add(year: int, month: int, add: int) -> (int, int):
        m = month + add
        y = year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        return y, m

    def last_day(y: int, m: int) -> int:
        return calendar.monthrange(y, m)[1]

    # 解析中文或数字形式的数量，支持 1-30
    def parse_number(num_str: str) -> int:
        num_str = num_str.strip()
        # 纯数字
        if num_str.isdigit():
            try:
                return int(num_str)
            except ValueError:
                return 0

        cn_map = {
            "零": 0, "〇": 0,
            "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
        }

        # 单个数字：一、二、三...
        if num_str in cn_map:
            return cn_map[num_str]

        # 十、十一、十二...十九
        if num_str == "十":
            return 10
        if len(num_str) == 2:
            if num_str[0] == "十" and num_str[1] in cn_map:
                # 十一 ~ 十九
                return 10 + cn_map[num_str[1]]
            if num_str[1] == "十" and num_str[0] in cn_map:
                # 二十、三十
                return cn_map[num_str[0]] * 10

        # 二十一、二十二... 二十九
        if len(num_str) == 3 and num_str[1] == "十" and num_str[0] in cn_map and num_str[2] in cn_map:
            return cn_map[num_str[0]] * 10 + cn_map[num_str[2]]

        return 0

    # -------- 带“几”的模糊时间 --------

    # 未来几天 / 近几天 / 这几天 / 后几天 → 从明天到三天后
    def repl_days(match):
        start = today + timedelta(days=1)
        end = start + timedelta(days=2)
        return f"从{fmt_d(start)}到{fmt_d(end)}"
    cleaned_question = re.sub(r'(将来|未来|近|这|后|后面)几天', repl_days, cleaned_question)

    # 几小时 → 从当前时刻到3小时后
    def repl_hours(match):
        start = now
        end = now + timedelta(hours=3)
        return f"从{fmt_dt(start)}到{fmt_dt(end)}"
    cleaned_question = re.sub(r'(将来|未来|近|这|后|后面)几小时', repl_hours, cleaned_question)

    # 几周 → 从明天到21天后
    def repl_weeks(match):
        start = today + timedelta(days=1)
        end = start + timedelta(days=21 - 1)
        return f"从{fmt_d(start)}到{fmt_d(end)}"
    cleaned_question = re.sub(r'(将来|未来|近|这|后|后面)几周', repl_weeks, cleaned_question)

    # 几个月 / 几月 → 下月1日到下月起三个月的月底
    def repl_months(match):
        start_y, start_m = month_add(today.year, today.month, 1)  # 下月
        end_y, end_m = month_add(start_y, start_m, 2)  # 共3个月
        # 月相关：只精确到“年+月”，不落到日
        return f"从{fmt_ym(start_y, start_m)}到{fmt_ym(end_y, end_m)}"
    cleaned_question = re.sub(r'(将来|未来|近|这|后|后面)几个月?', repl_months, cleaned_question)

    # 几年 → 明年到往后5年的年底
    def repl_years(match):
        start_y = today.year + 1
        end_y = start_y + 5 - 1
        # 年相关：只精确到年份区间
        return f"从{fmt_y(start_y)}到{fmt_y(end_y)}"
    cleaned_question = re.sub(r'(将来|未来|近|这|后|后面|下|接下来|下个|下一|下)几年', repl_years, cleaned_question)

    # -------- 带具体数字的时间：支持 1~30（阿拉伯数字或中文数字） --------
    num_pattern = r'(?P<num>[一二三四五六七八九十零〇两\d]{1,3})'
    prefix_pattern = r'(将来|未来|近|这|后|后面|下|接下来|下个|下一|下)'

    # N天 → 从明天起 N 天
    def repl_days_num(match):
        n = parse_number(match.group("num"))
        if n <= 0:
            return match.group(0)
        n = min(n, 30)
        start = today + timedelta(days=1)
        end = start + timedelta(days=n - 1)
        return f"从{fmt_d(start)}到{fmt_d(end)}"

    cleaned_question = re.sub(
        prefix_pattern + num_pattern + r'天',
        repl_days_num,
        cleaned_question,
    )

    # N小时 → 从当前时刻起 N 小时
    def repl_hours_num(match):
        n = parse_number(match.group("num"))
        if n <= 0:
            return match.group(0)
        n = min(n, 30)
        start = now
        end = now + timedelta(hours=n)
        return f"从{fmt_dt(start)}到{fmt_dt(end)}"

    cleaned_question = re.sub(
        prefix_pattern + num_pattern + r'小时',
        repl_hours_num,
        cleaned_question,
    )

    # N周 → 从明天起 N 周
    def repl_weeks_num(match):
        n = parse_number(match.group("num"))
        if n <= 0:
            return match.group(0)
        n = min(n, 30)
        days = n * 7
        start = today + timedelta(days=1)
        end = start + timedelta(days=days - 1)
        return f"从{fmt_d(start)}到{fmt_d(end)}"

    cleaned_question = re.sub(
        prefix_pattern + num_pattern + r'周',
        repl_weeks_num,
        cleaned_question,
    )

    # N个月 / N月 → 下月起 N 个月
    def repl_months_num(match):
        n = parse_number(match.group("num"))
        if n <= 0:
            return match.group(0)
        n = min(n, 30)
        start_y, start_m = month_add(today.year, today.month, 1)  # 下月
        end_y, end_m = month_add(start_y, start_m, n - 1)
        # 只精确到“年+月”
        return f"从{fmt_ym(start_y, start_m)}到{fmt_ym(end_y, end_m)}"

    cleaned_question = re.sub(
        prefix_pattern + num_pattern + r'个月?',
        repl_months_num,
        cleaned_question,
    )

    # N年 → 明年起 N 年
    def repl_years_num(match):
        n = parse_number(match.group("num"))
        if n <= 0:
            return match.group(0)
        n = min(n, 30)
        start_y = today.year + 1
        end_y = start_y + n - 1
        # 只精确到年份区间
        return f"从{fmt_y(start_y)}到{fmt_y(end_y)}"

    cleaned_question = re.sub(
        prefix_pattern + num_pattern + r'年',
        repl_years_num,
        cleaned_question,
    )

    # 近期 / 最近 / 这段时间 / 下阶段等 → 未来三个月（下月起三个月）
    cleaned_question = re.sub(
        r'近期|这个阶段|当前阶段|现阶段|当下阶段|接下来|这段时间|最近|下个阶段|下阶段|下一阶段|下下阶段',
        repl_months,
        cleaned_question
    )

    # 这两天 / 马上 → 今天到明天
    def repl_two_days(match):
        start = today
        end = today + timedelta(days=1)
        return f"从{fmt_d(start)}到{fmt_d(end)}"
    cleaned_question = re.sub(r'这两天|马上', repl_two_days, cleaned_question)

    # 年前 / 年内 → 当年全年
    def repl_year_inside(match):
        y = today.year
        return fmt_y(y)
    cleaned_question = re.sub(r'年前|年内', repl_year_inside, cleaned_question)

    # 下个月 → 下月1日到下月月底
    def repl_next_month(match):
        start_y, start_m = month_add(today.year, today.month, 1)
        # 单个月份：用“年+月”
        return f"从{fmt_ym(start_y, start_m)}到{fmt_ym(start_y, start_m)}"
    cleaned_question = re.sub(r'下个月', repl_next_month, cleaned_question)

    # 今年 / 当年 / 这一年 / 这年 / 本年：直接替换为具体年份（不再使用“从X年到X年”的区间形式）
    def repl_this_year(match):
        y = today.year
        # 直接输出“YYYY年”
        return fmt_y(y)
    cleaned_question = re.sub(r'今年|当年|这一年|这年|本年', repl_this_year, cleaned_question)

    # 明年 / 下一年 / 下年：直接替换为具体年份
    def repl_next_year(match):
        y = today.year + 1
        return fmt_y(y)
    cleaned_question = re.sub(r'明年|下一年|下年', repl_next_year, cleaned_question)

    # 后年 / 后一年：从后年1月1日到后年12月31日
    def repl_year_after_next(match):
        y = today.year + 2
        return fmt_y(y)
    cleaned_question = re.sub(r'后年|后一年', repl_year_after_next, cleaned_question)

    # 下半年仍保持原逻辑：当年7月1日到12月31日（如需跨年可再调整）
    current_year = today.year
    cleaned_question = re.sub(r'(?<!\\d)下半年', f'{current_year}年7月1日到12月31日', cleaned_question)

    return cleaned_question



# --- 签名生成函数 (保持不变) ---
def generate_signature_daily(params: dict, secret_key: str) -> str:
    params_to_sign = params.copy()
    if 'sign' in params_to_sign: del params_to_sign['sign']
    sorted_items = sorted(params_to_sign.items())
    message = "".join([f"{k}={v}" for k, v in sorted_items])
    return hmac.new(secret_key.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()


def generate_signature(params: Dict[str, Any], app_secret: str) -> str:
    sorted_params = dict(
        sorted({k: str(v) for k, v in params.items() if k not in ['sign', 'card_number_pool', 'hl_ymd', 'skip_intent_check']}.items()))
    string_to_sign = "".join(f"{k}{v}" for k, v in sorted_params.items())
    return hmac.new(app_secret.encode('utf-8'), string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()


# --- 应用启动和关闭事件 (保持不变) ---
@app.on_event("startup")
async def startup_event():
    # (此函数逻辑保持不变)
    logger.info("FastAPI 应用启动中...")
    
    # 【修改2】初始化session_manager
    await initialize_session_manager()
    logger.info("Session Manager 已初始化")
    
    global async_aiohttp_client
    connector = aiohttp.TCPConnector(limit=1000, limit_per_host=1000, enable_cleanup_closed=True, force_close=False,
                                     keepalive_timeout=120)
    async_aiohttp_client = aiohttp.ClientSession(connector=connector,
                                                 timeout=aiohttp.ClientTimeout(connect=VLLM_REQUEST_TIMEOUT_SECONDS,
                                                                               sock_read=None))
    global vllm_semaphore
    vllm_semaphore = asyncio.Semaphore(VLLM_CONCURRENT_LIMIT)
    
    # 【修复】将客户端和信号量存储到 app.state，供 queryIntent.py 使用
    app.state.http_session = async_aiohttp_client
    app.state.vllm_semaphore = vllm_semaphore
    
    # 【新增】初始化数据库连接池（如果配置了数据库）
    if DB_AVAILABLE and DB_NAME:
        try:
            db_pool = await aiomysql.create_pool(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                db=DB_NAME,
                minsize=1,
                maxsize=DB_POOL_SIZE,
                autocommit=True,
                charset='utf8mb4',
                # 不使用 DictCursor，使用普通游标（返回元组），与 queryIntent.py 中的 row[0] 兼容
            )
            app.state.db_pool = db_pool
            logger.info(f"✅ 数据库连接池初始化成功: {DB_HOST}:{DB_PORT}/{DB_NAME}")
        except Exception as e:
            logger.error(f"❌ 数据库连接池初始化失败: {e}", exc_info=True)
            app.state.db_pool = None
    else:
        if not DB_AVAILABLE:
            logger.warning("⚠️ aiomysql 未安装，数据库功能不可用。如需使用数据库，请安装: pip install aiomysql")
        elif not DB_NAME:
            logger.warning("⚠️ DB_NAME 未配置，数据库功能不可用。请在 .env 文件中配置 DB_NAME")
        app.state.db_pool = None
    
    logger.info("FastAPI 应用启动完成。")


@app.on_event("shutdown")
async def shutdown_event():
    # (此函数逻辑保持不变)
    logger.info("FastAPI 应用关闭中...")
    
    # 【修改2】关闭session_manager
    await close_session_manager()
    logger.info("Session Manager 已关闭")
    
    if async_aiohttp_client:
        await async_aiohttp_client.close()
    
    # 【新增】关闭数据库连接池
    if hasattr(app.state, 'db_pool') and app.state.db_pool:
        app.state.db_pool.close()
        await app.state.db_pool.wait_closed()
        logger.info("数据库连接池已关闭")
    
    # 清理 app.state
    if hasattr(app.state, 'http_session'):
        delattr(app.state, 'http_session')
    if hasattr(app.state, 'vllm_semaphore'):
        delattr(app.state, 'vllm_semaphore')
    if hasattr(app.state, 'db_pool'):
        delattr(app.state, 'db_pool')
    
    logger.info("FastAPI 应用关闭完成。")


# --- Helper function: Stream synchronous response to client (保持不变) ---
async def stream_response_with_done(sync_response: requests.Response, prefix_message: Optional[str] = None) -> AsyncGenerator[bytes, None]:
    if prefix_message:
        yield prefix_message.encode('utf-8')
        sys.stdout.flush() 

    sync_iterator = sync_response.iter_content(chunk_size=None)

    while True:
        try:
            chunk_bytes = await asyncio.to_thread(lambda: next(sync_iterator, None))
            if chunk_bytes is None:
                break
            yield chunk_bytes
            sys.stdout.flush()
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError, 
                requests.exceptions.RequestException, StopIteration) as e:
            # 连接提前结束或网络错误，记录日志并正常结束
            logger.warning(f"[流式响应] 连接提前结束或网络错误: {e}")
            break
        except Exception as e:
            # 其他未知错误，记录并中断
            logger.error(f"[流式响应] 读取数据时发生未知错误: {e}", exc_info=True)
            break

    yield b"\n[DONE]\n" 
    logger.info(f"[DONE] 标记已发送到客户端。")
    sys.stdout.flush()


async def stream_response_with_history(
    sync_response: requests.Response, 
    prefix_message: Optional[str],
    session_id: str,
    user_question: str,
    current_intent: str,
    history_manager,
    request_id: str,
) -> AsyncGenerator[bytes, None]:
    """
    【修改2】增强版流式响应，支持保存历史记录和意图
    使用 try-finally 确保意图一定被保存，即使 generator 被取消
    """
    try:
        if prefix_message:
            yield prefix_message.encode('utf-8')
            sys.stdout.flush() 

        sync_iterator = sync_response.iter_content(chunk_size=None)
        
        # 收集完整响应内容用于保存历史
        full_response = []

        while True:
            try:
                chunk_bytes = await asyncio.to_thread(lambda: next(sync_iterator, None))
                if chunk_bytes is None:
                    break
                full_response.append(chunk_bytes)
                yield chunk_bytes
                sys.stdout.flush()
            except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError, 
                    requests.exceptions.RequestException, StopIteration) as e:
                # 连接提前结束或网络错误，记录日志但继续处理已接收的数据
                logger.warning(f"[流式响应] 连接提前结束或网络错误: {e}，已接收 {len(full_response)} 个数据块")
                break
            except Exception as e:
                # 其他未知错误，记录并中断
                logger.error(f"[流式响应] 读取数据时发生未知错误: {e}", exc_info=True)
                break

        # 最后发送 DONE 标记
        yield b"\n[DONE]\n" 
        logger.info(f"[DONE] 标记已发送到客户端。")
        sys.stdout.flush()
        
    finally:
        # 【关键修复】使用 finally 确保意图一定被保存，即使 generator 被取消或出错
        # 注意：历史对话由下游服务（ziwei/leipai）负责保存，summary 只负责保存意图用于路由决策
        try:
            if current_intent:
                await store_last_intent(session_id, current_intent)
                logger.info(f"[Summary-Session] 保存意图: session={session_id}, intent={current_intent}")
            else:
                logger.warning(f"[Summary-Session] 警告：current_intent 为 None，无法保存意图")
        except Exception as e:
            logger.error(f"[Summary-Session] 保存意图时出错: {e}", exc_info=True)
        finally:
            log_step(
                "成功输出给客户端",
                request_id,
                {"session_id": session_id, "intent": current_intent},
            )


# =============================================================================
# === Core Modification: Updated Unified Chat Endpoint ===
# =============================================================================
@app.post("/unified_chat_V12_25")
async def unified_chat(request: Request,request_body: UnifiedChatRequest):
    """
    统一聊天接口，通过前置规则和LLM意图分析，智能决定调用哪个下游服务。
    这个版本手动接收原始请求体，并仅修复换行符问题。
    """
    logger.info("收到客户端请求，开始手动解析...")
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:8]

    # --- 步骤 1: 读取原始请求体，修复换行符，然后重新解析 ---
    with StepMonitor("成功获取请求", request_id, {"path": str(request.url)}):
        try:
            body_bytes = await request.body()
            body_str = body_bytes.decode('utf-8')
            
            # 核心修改：仅替换不合法的换行符
            body_str_fixed = body_str.replace('\n', r'\n').replace('\r', r'\r')
            
            # 使用 json.loads 尝试解析修复后的字符串
            data = json.loads(body_str_fixed)

            # 使用 Pydantic 模型进行最终验证，确保数据格式符合预期
            request_body = UnifiedChatRequest(**data)

        except json.JSONDecodeError as e:
            logger.error(f"请求体无法解析为JSON。原始请求: '{body_str}'. 错误: {e.msg}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"请求数据格式不正确，无法处理。JSON解析错误: {e.msg}"
            )
        except ValidationError as e:
            logger.error(f"请求数据不符合Pydantic模型规范。错误: {e.errors()}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"请求数据不符合规范: {e.errors()}"
            )
        except Exception as e:
            logger.critical(f"处理请求时发生意外错误: {type(e).__name__} - {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"内部服务器错误: {e}"
            )
    
    logger.info(f"成功解析并验证请求体: {request_body.model_dump_json(indent=2)}")

    # 【修改2】获取或生成session_id，获取历史意图
    session_id = request_body.session_id if request_body.session_id else str(uuid.uuid4())
    logger.info(f"[会话: {session_id}] 开始处理请求")
    
    # 获取历史意图，用于智能路由决策
    with StepMonitor("成功获取数据库信息", request_id, {"session_id": session_id}):
        last_intent = await get_last_intent(session_id)
        logger.info(f"[会话: {session_id}] 历史意图: {last_intent}")
        
        # 获取会话历史
        history_manager = await get_session_history(session_id)

    # --- 步骤 2: 保持原有的业务逻辑 ---
    current_time = int(time.time())
    target_url = ""
    payload: Dict[str, Any] = {}
    app_secret = APP_SECRETS["yingshi_appid"]
    headers = {'Content-Type': 'application/json'}
    prefix_message_to_client: Optional[str] = None
    response_media_type: str = 'text/plain'
    current_intent = None  # 记录本次意图

    # 智能标准化用户输入的标点符号（保护出生信息格式）
    normalized_prompt = smart_normalize_punctuation(request_body.prompt)
    
    # 【修改1】统一大限/大运的表述 - 仅处理模糊表述，不影响明确的时间范围
    # 只转换"这个大运"、"这一大运"等模糊表述，不转换"下一个大运"、"后10年大运"等明确表述
    if re.search(r'^(?:这个|这一)大限$', normalized_prompt) or re.search(r'(?:这个|这一)大限(?!.*?(下|未来|今后|后\d+年))', normalized_prompt):
        if '当前大限' not in normalized_prompt and '下' not in normalized_prompt and '未来' not in normalized_prompt:
            normalized_prompt = re.sub(r'(?:这个|这一)大限', '当前大限', normalized_prompt)
            logger.info(f"将模糊大限表述统一为'当前大限': {normalized_prompt}")
    
    if re.search(r'^(?:这个|这一)大运$', normalized_prompt) or (re.search(r'(?:这个|这一)大运', normalized_prompt) and not re.search(r'(下一个|下个|未来|今后|后\d+年|接下来)', normalized_prompt)):
        if '当前大运' not in normalized_prompt:
            normalized_prompt = re.sub(r'(?:这个|这一)大运', '当前大运', normalized_prompt)
            logger.info(f"将模糊大运表述统一为'当前大运': {normalized_prompt}")
    
    # logger.info(f"原始提问: {request_body.prompt}")
    if normalized_prompt != request_body.prompt:
        logger.info(f"标准化后: {normalized_prompt}")
    log_step(
        "成功构建提示词",
        request_id,
        {"session_id": session_id, "prompt_length": len(normalized_prompt)},
    )

    # --- 前置违法内容快速检测 ---
    if detect_illegal_content(normalized_prompt):
        illegal_response = """抱歉，您的问题我无法回答。"""
        
        # 保存历史
        await store_last_intent(session_id, "illegal_refused")
        await history_manager.add_messages([
            HumanMessage(content=normalized_prompt),
            AIMessage(content=illegal_response)
        ])
        
        async def illegal_stream():
                
            yield illegal_response.encode('utf-8')
            yield b"\n[DONE]\n"
        
        log_step(
            "成功输出给客户端",
            request_id,
            {"session_id": session_id, "reason": "illegal_refused"},
        )
        return StreamingResponse(illegal_stream(), media_type='text/plain')
    
    # --- 金融/投资/彩票 固定回复（在重大事件时间点拦截之前） ---
    if detect_finance_investment_query(normalized_prompt):
        finance_response = """您好，我的专长在于通过命理工具为您提供人生趋势的洞察与个人决策的参考，而非金融市场分析，我无法也绝不会提供任何具体的金融投资建议，包括对个股、基金或其他金融产品的买卖建议。理财有风险，投资需谨慎。"""
        
        await store_last_intent(session_id, "finance_investment_refused")
        await history_manager.add_messages([
            HumanMessage(content=normalized_prompt),
            AIMessage(content=finance_response)
        ])

        async def finance_stream():
            yield finance_response.encode('utf-8')
            yield b"\n[DONE]\n"

        log_step(
            "成功输出给客户端",
            request_id,
            {"session_id": session_id, "reason": "finance_investment_refused"},
        )
        return StreamingResponse(finance_stream(), media_type='text/plain')
    
    # --- 政治敏感内容检测 ---
    if detect_sensitive_political_content(normalized_prompt):
        political_response = "抱歉，您的问题我无法回答。根据相关规定，我无法讨论或分析任何与政治、军事相关的话题。我的能力范围仅限于基于紫微斗数的个人运势分析。"
        
        # 保存历史
        await store_last_intent(session_id, "political_refused")
        await history_manager.add_messages([
            HumanMessage(content=normalized_prompt),
            AIMessage(content=political_response)
        ])
        
        async def political_stream():
                
            yield political_response.encode('utf-8')
            yield b"\n[DONE]\n"
        
        log_step(
            "成功输出给客户端",
            request_id,
            {"session_id": session_id, "reason": "political_refused"},
        )
        return StreamingResponse(political_stream(), media_type='text/plain')
    
    # --- 年龄相关问题检测 ---
    if detect_age_inquiry(normalized_prompt):
        age_response = """抱歉，您的提问方式我无法准确回答。

目前系统暂不支持基于具体年龄（如"20岁"、"30岁"）的运势查询。

建议您可以这样提问：
- "我未来5年的运势如何？"
- "我未来10年的事业发展怎么样？"
- "我今年的财运如何？"
- "我明年的感情运势会怎样？"

这样的提问方式能让我为您提供更准确的运势分析。"""
        
        # 保存历史
        await store_last_intent(session_id, "age_inquiry_refused")
        await history_manager.add_messages([
            HumanMessage(content=normalized_prompt),
            AIMessage(content=age_response)
        ])
        
        async def age_stream():
                
            yield age_response.encode('utf-8')
            yield b"\n[DONE]\n"
        
        log_step(
            "成功输出给客户端",
            request_id,
            {"session_id": session_id, "reason": "age_inquiry_refused"},
        )
        return StreamingResponse(age_stream(), media_type='text/plain')
    
    # --- 问题5: 乱码检测 ---
    if is_gibberish(normalized_prompt):
        guidance_response = """抱歉，您的问题我无法回答。我没能理解您的问题。

为了更好地帮助您，请尝试提出清晰的运势相关问题，例如：
- "我今年的事业运势如何？"
- "我最近一个月的财运怎么样？"
- "我在感情方面会有什么变化吗？"
- "我的综合运势趋势如何？"

请重新描述您的问题，我会为您进行专业的运势分析。"""
        
        async def guidance_stream():
                
            yield guidance_response.encode('utf-8')
            yield b"\n[DONE]\n"
        
        log_step(
            "成功输出给客户端",
            request_id,
            {"session_id": session_id, "reason": "gibberish_refused"},
        )
        return StreamingResponse(guidance_stream(), media_type='text/plain')
    
    # --- 来世/下辈子问题拦截 ---
    if detect_reincarnation_query(normalized_prompt):
        reincarnation_response = "很抱歉，我无法预测“下辈子”或“来世”的运势。紫微斗数侧重于对您今生命运和运势的分析。"
        
        await store_last_intent(session_id, "reincarnation_refused")
        await history_manager.add_messages([
            HumanMessage(content=normalized_prompt),
            AIMessage(content=reincarnation_response)
        ])
        
        async def reincarnation_stream():
                
            yield reincarnation_response.encode('utf-8')
            yield b"\n[DONE]\n"
        
        log_step(
            "成功输出给客户端",
            request_id,
            {"session_id": session_id, "reason": "reincarnation_refused"},
        )
        return StreamingResponse(reincarnation_stream(), media_type='text/plain')
    
    # --- 问题11: 重大时间点拦截 ---
    is_critical, category = detect_critical_time_selection(normalized_prompt)
    if is_critical:
        refusal_response = f"""抱歉，您的问题我无法回答。关于【{category}】这类重大事项的具体时间选择，建议您：

1. 咨询相关领域的专业人士（医生、律师、财务顾问等）
2. 综合考虑实际情况和专业建议
3. 不要仅依赖命理推算做出重大决策

命理分析可以作为参考，但涉及健康、法律、重大财务等事项时，专业判断更为重要。

如果您想了解该时期的整体运势趋势，我可以为您分析。"""
        
        async def refusal_stream():
                
            yield refusal_response.encode('utf-8')
            yield b"\n[DONE]\n"
        
        log_step(
            "成功输出给客户端",
            request_id,
            {"session_id": session_id, "reason": f"critical_{category}"},
        )
        return StreamingResponse(refusal_stream(), media_type='text/plain')
    
    # --- 模糊时间表达标准化（在进入意图分类之前） ---
    # normalized_prompt = normalize_vague_time_expressions(normalized_prompt)
    
    # 【修改】不再在这里强制判断，而是让所有请求都通过意图分类
    # 如果确实没有出生信息，由意图分类和兜底逻辑来处理
    has_birth_info = contains_formatted_birth_info(normalized_prompt)
    
    if has_birth_info:
        logger.info("--- 检测到格式化出生信息，开始进行LLM意图分类 ---")
        intent_type: Optional[str] = None
        
        # 【功能1】检测模糊追问（如"我应该怎么做"、"继续"、"呢"等）
        ambiguous_keywords = ["我应该怎么做", "怎么做", "该怎么办", "继续", "呢", "展开", "详细说说"]
        is_ambiguous = any(kw in normalized_prompt for kw in ambiguous_keywords) and len(normalized_prompt) < 30
        
        # 初始化意图变量
        intent_type = None
        intent_reason = None
        # 奇门提取结果（来自并行请求）
        qimen_extraction_result = None
        
        if is_ambiguous and last_intent and last_intent in ["ziwei", "leipai"]:
            logger.info(f"--- 检测到模糊追问，直接使用历史意图 [{last_intent}] ---")
            intent_type = "general_long_term" if last_intent == "ziwei" else "specific_short_term"
            intent_reason = f"模糊追问，继承历史意图{last_intent}"
        else:
            try:
                db_pool = getattr(request.app.state, "db_pool", None)
                # 在进行意图分类、时间范围识别和奇门提取前，先去除出生信息
                question_only = remove_birth_info(normalized_prompt)
                logger.info(f"[意图识别] 原始输入: {normalized_prompt}")
                logger.info(f"[意图识别] 去除出生信息后: {question_only}")
                
                with StepMonitor("串行请求时间范围识别，然后并行请求意图分类和奇门提取", request_id, {"phase": "has_birth_info"}):
                    # 第一步：先串行执行时间范围识别
                    time_range_result = await extract_time_range_with_llm(
                        request.app,
                        question_only,
                    )
                    
                    # 从时间范围识别结果中提取时间跨度类型
                    time_span_type = time_range_result.get("time_span_type") if isinstance(time_range_result, dict) else None
                    logger.info(f"时间范围识别完成，time_span_type={time_span_type}")
                    
                    # 第二步：并行调用意图分类和奇门提取（意图分类使用时间跨度类型）
                    intent_task = classify_query_intent_with_llm(
                        request.app,
                        question_only,
                        db_pool=db_pool,
                        time_span_type=time_span_type,
                    )
                    qimen_task = extract_qimen_with_llm(
                        request.app,
                        question_only,
                        db_pool=db_pool,
                    )
                    query_intent_result, qimen_extraction_result = await asyncio.gather(
                        intent_task, qimen_task, return_exceptions=True
                    )
                
                # 处理意图分类结果
                if isinstance(query_intent_result, Exception):
                    raise query_intent_result
                logger.info(f"LLM 意图分类成功: {query_intent_result}")
                intent_type = query_intent_result.get("query_type")
                intent_reason = query_intent_result.get("reason", "")
                
                # 处理时间范围识别结果
                if isinstance(time_range_result, Exception):
                    logger.warning(f"时间范围识别失败，使用默认结果（非历史事件）: {time_range_result}")
                    time_range_result = {"has_time_range": False, "end_date": None, "time_expression": None, "time_span_type": "long_term", "reason": "提取失败，默认time_span_type为long_term"}
                logger.info(f"时间范围识别结果: {time_range_result}")
                
                # 判断是否为历史事件：如果提取到时间范围，且结束日期早于当前日期，则为历史事件
                is_historical_event = False
                if time_range_result.get("has_time_range") and time_range_result.get("end_date"):
                    try:
                        end_date_str = time_range_result.get("end_date")
                        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
                        current_date = datetime.now().date()
                        if end_date < current_date:
                            is_historical_event = True
                            logger.info(f"判断为历史事件：结束日期 {end_date_str} 早于当前日期 {current_date.strftime('%Y-%m-%d')}")
                        else:
                            logger.info(f"判断为非历史事件：结束日期 {end_date_str} 晚于或等于当前日期 {current_date.strftime('%Y-%m-%d')}")
                    except Exception as e:
                        logger.warning(f"解析时间范围结束日期失败: {e}，默认视为非历史事件")
                
                # 处理奇门提取结果（现在是列表）
                if isinstance(qimen_extraction_result, Exception):
                    logger.warning(f"奇门提取失败，使用默认结果: {qimen_extraction_result}")
                    qimen_extraction_result = [{"is_qimen": False, "qimen_type": None, "matched_tag": None, "reason": "提取失败"}]
                elif not isinstance(qimen_extraction_result, list):
                    # 兼容旧版本（单个结果转为列表）
                    qimen_extraction_result = [qimen_extraction_result] if qimen_extraction_result else [{"is_qimen": False, "qimen_type": None, "matched_tag": None, "reason": "无结果"}]
                logger.info(f"奇门提取结果（共{len(qimen_extraction_result)}次）: {qimen_extraction_result}")
                
            except Exception as e:
                logger.warning(f"网关说：意图分类失败，将启用兜底方案。错误: {e}", exc_info=True)
                # 兜底方案考虑历史意图
                if last_intent:
                    logger.info(f"--- 兜底方案：根据历史意图 [{last_intent}] 进行路由 ---")
                    intent_type = "specific_short_term" if last_intent == "leipai" else "general_long_term"
                    intent_reason = f"意图分类失败，兜底使用历史意图{last_intent}"
                else:
                    logger.info("--- 兜底方案：将请求路由至智慧卡接口 ---")
                    intent_type = "specific_short_term"
                    intent_reason = "意图分类失败，兜底使用智慧卡"
                qimen_extraction_result = {"is_qimen": False, "qimen_type": None, "matched_tag": None, "reason": "兜底"}
        
        # 【新增】助手自我介绍类意图，固定回复
        if intent_type == "self_intro":
            intro_response = "我是您的AI生活小助手，集传统文化智慧与现代AI技术于一体，为您提供传统万年历解读、每日运势宜忌及日常养生指南。让千年智慧融入您的生活，在虚实之间揭开未来的迷雾。"
            await store_last_intent(session_id, "self_intro")
            await history_manager.add_messages([
                HumanMessage(content=normalized_prompt),
                AIMessage(content=intro_response)
            ])

            async def intro_stream():
                yield intro_response.encode('utf-8')
                yield b"\n[DONE]\n"

            log_step(
                "成功输出给客户端",
                request_id,
                {"session_id": session_id, "reason": "self_intro"},
            )
            return StreamingResponse(intro_stream(), media_type='text/plain')
        
        # 【新增】非紫微体系类意图，固定回复
        if intent_type == "non_ziwei_system":
            nzs_response = """抱歉，您的问题我无法回答。我是专注于命理运势分析的AI助手，您提出的问题超出了我的服务范围。

我的专长领域包括：
- 紫微斗数运势分析
- 智慧卡指引
- 命理专业知识解答
- 奇门遁甲事时决策

如果您有关于个人运势、命理方面的问题，欢迎随时向我咨询。"""
            await store_last_intent(session_id, "non_ziwei_system_refused")
            await history_manager.add_messages([
                HumanMessage(content=normalized_prompt),
                AIMessage(content=nzs_response)
            ])

            async def nzs_stream():
                yield nzs_response.encode('utf-8')
                yield b"\n[DONE]\n"

            log_step(
                "成功输出给客户端",
                request_id,
                {"session_id": session_id, "reason": "non_ziwei_system_refused"},
            )
            return StreamingResponse(nzs_stream(), media_type='text/plain')
        
        # 【新增】违法内容检测 - 最高优先级
        if intent_type == "illegal_content":
            logger.info("网关说：--- 查询分类为 [违法犯罪内容], 拒绝服务 ---")
            illegal_response = """抱歉，您的问题我无法回答。"""
            
            # 保存历史
            await store_last_intent(session_id, "illegal_refused")
            await history_manager.add_messages([
                HumanMessage(content=normalized_prompt),
                AIMessage(content=illegal_response)
            ])
            
            async def illegal_stream():
                yield illegal_response.encode('utf-8')
                yield b"\n[DONE]\n"
            
            log_step(
                "成功输出给客户端",
                request_id,
                {"session_id": session_id, "reason": "intent_illegal_refused"},
            )
            return StreamingResponse(illegal_stream(), media_type='text/plain')
        
        # 【新增】常识性知识检测 - 第二优先级
        elif intent_type == "general_knowledge":
            logger.info("网关说：--- 查询分类为 [常识性知识], 拒绝服务 ---")
            knowledge_response = """抱歉，您的问题我无法回答。我是专注于命理运势分析的AI助手，您提出的问题超出了我的服务范围。

我的专长领域包括：
- 紫微斗数运势分析
- 智慧卡指引
- 命理专业知识解答

如果您有关于个人运势、命理方面的问题，欢迎随时向我咨询。"""
            
            # 保存历史
            await store_last_intent(session_id, "general_knowledge_refused")
            await history_manager.add_messages([
                HumanMessage(content=normalized_prompt),
                AIMessage(content=knowledge_response)
            ])
            
            async def knowledge_stream():
                yield knowledge_response.encode('utf-8')
                yield b"\n[DONE]\n"
            
            log_step(
                "成功输出给客户端",
                request_id,
                {"session_id": session_id, "reason": "intent_general_knowledge_refused"},
            )
            return StreamingResponse(knowledge_stream(), media_type='text/plain')
        
        # 【问题4优化】检测历史事件 - 基于时间范围识别结果判断
        if is_historical_event:
            logger.info("网关说：--- 根据时间范围识别判断为 [历史事件], 拒绝推演 ---")
            historical_response = "抱歉，您的问题我无法回答。我专注于未来运势的推演和指导。对于已经发生的历史事件，我无法进行回溯性分析。如果您想了解未来的运势趋势，请提出关于未来时间的问题。"
            
            # 保存历史
            await store_last_intent(session_id, "historical")
            await history_manager.add_messages([
                HumanMessage(content=normalized_prompt),
                AIMessage(content=historical_response)
            ])
            
            async def historical_stream():
                yield historical_response.encode('utf-8')
                yield b"\n[DONE]\n"
            
            log_step(
                "成功输出给客户端",
                request_id,
                {"session_id": session_id, "reason": "time_range_historical_refused", "end_date": time_range_result.get("end_date")},
            )
            return StreamingResponse(historical_stream(), media_type='text/plain')
        
        elif intent_type == "knowledge_question":
            logger.info("网关说：--- 查询分类为 [专业知识问答], 直接在网关调用VLLM回答 ---")
            current_intent = "knowledge"
            
            question_only = remove_birth_info(normalized_prompt)
            logger.info(f"[专业知识问答] 原始提问: {normalized_prompt}")
            logger.info(f"[专业知识问答] 移除出生信息后: {question_only}")
            
            try:
                with StepMonitor("成功请求llm", request_id, {"phase": "knowledge_answer"}):
                    knowledge_answer = await answer_knowledge_question(request.app, question_only)
                
                # 保存意图和历史
                await store_last_intent(session_id, current_intent)
                await history_manager.add_messages([
                    HumanMessage(content=normalized_prompt),
                    AIMessage(content=knowledge_answer)
                ])
                
                async def knowledge_stream():
                    yield knowledge_answer.encode('utf-8')
                    yield b"\n[DONE]\n"
                
                log_step(
                    "成功输出给客户端",
                    request_id,
                    {"session_id": session_id, "reason": "knowledge_answer"},
                )
                return StreamingResponse(knowledge_stream(), media_type='text/plain')

            except Exception as e:
                logger.error(f"专业知识回答失败: {e}", exc_info=True)
                # 如果失败，继续走原来的路由逻辑
                logger.info("专业知识回答失败，回退到紫微服务")
                target_url = url_ziwei
                prefix_message_to_client = "【根据您的询问，我将为您解释相关的命理知识】\n\n"
                current_intent = "ziwei"
                
                payload = {
                    "appid": "yingshi_appid",
                    "query": normalized_prompt,
                    "session_id": session_id,
                    "timestamp": str(current_time),
                    "is_knowledge_query": True,
                    "skip_intent_check": True,
                    "summary_intent_type": intent_type,  # 传递Summary的意图类型
                    "summary_intent_reason": intent_reason  # 传递Summary的分类理由
                }
                payload["sign"] = generate_signature(payload, app_secret)
        elif intent_type == "qimen":
            # 【双重校验】意图分类判断为qimen时，结合奇门提取结果进行二次校验
            should_fallback_to_leipai = False
            selected_result = None
            
            # 处理奇门提取结果列表（5次并行请求的结果）
            qimen_results = qimen_extraction_result if isinstance(qimen_extraction_result, list) else [qimen_extraction_result] if qimen_extraction_result else []
            
            logger.info(f"--- 意图分类=qimen, 奇门提取结果数量: {len(qimen_results)} ---")
            
            # 检查是否有任何一次提取确认为奇门
            qimen_results_filtered = [r for r in qimen_results if r.get("is_qimen", False)]
            
            if not qimen_results_filtered:
                logger.info("--- 所有奇门提取结果都不认为是奇门，降级走雷牌 ---")
                should_fallback_to_leipai = True
            else:
                logger.info(f"--- 有 {len(qimen_results_filtered)} 次提取确认为奇门，开始处理 ---")
                
                # 检查是否有类型3（类型3直接放行）
                type3_results = [r for r in qimen_results_filtered if r.get("qimen_type") == 3]
                if type3_results:
                    logger.info("--- 发现类型3结果，无需标签校验，直接走奇门 ---")
                    selected_result = type3_results[0]
                else:
                    # 检查类型1/2，看是否有能匹配数据库标签的
                    type12_results = [r for r in qimen_results_filtered if r.get("qimen_type") in (1, 2)]
                    if not type12_results:
                        logger.info("--- 没有类型1/2的结果，降级走雷牌 ---")
                        should_fallback_to_leipai = True
                    else:
                        logger.info(f"--- 有 {len(type12_results)} 次类型1/2的结果，开始标签匹配校验 ---")
                        try:
                            db_pool = getattr(request.app.state, "db_pool", None)
                            valid_tags = await get_all_tags_from_db_async(db_pool)
                            logger.info(f"--- 数据库标签列表有 {len(valid_tags)} 个标签 ---")
                            
                            # 查找第一个能匹配数据库标签的结果
                            matched_result = None
                            for result in type12_results:
                                matched_tag = result.get("matched_tag")
                                qimen_type = result.get("qimen_type")
                                if matched_tag and matched_tag in valid_tags:
                                    logger.info(f"--- ✅ 找到匹配结果！类型{qimen_type}，标签 [{matched_tag}] 在数据库中 ---")
                                    matched_result = result
                                    break
                                else:
                                    logger.info(f"--- ❌ 类型{qimen_type}，标签 [{matched_tag}] 不在数据库中 ---")
                            
                            if matched_result:
                                selected_result = matched_result
                                logger.info("--- 使用匹配的结果，放行走奇门 ---")
                            else:
                                logger.info("--- 所有类型1/2的结果都无法匹配数据库标签，降级走雷牌 ---")
                                should_fallback_to_leipai = True
                        except Exception as e:
                            logger.warning(f"标签校验失败: {e}，将降级走雷牌", exc_info=True)
                            should_fallback_to_leipai = True
            
            # 如果还没有选择结果，使用第一个奇门结果（用于日志记录）
            if not selected_result and qimen_results_filtered:
                selected_result = qimen_results_filtered[0]
            
            if selected_result:
                logger.info(f"--- 最终选择的奇门提取结果: type={selected_result.get('qimen_type')}, tag={selected_result.get('matched_tag')} ---")
            
            if should_fallback_to_leipai:
                # 降级到雷牌
                logger.info("--- 奇门降级：调用智慧卡接口 ---")
                target_url = url_leipai
                prefix_message_to_client = "【根据您的询问事项，我们启用智慧卡的方式给您建议】\n\n正在推演中...\n"
                current_intent = "leipai"
                
                payload = {
                    "appid": "yingshi_appid",
                    "prompt": normalized_prompt,
                    "format": "json",
                    "ftime": current_time,
                    "session_id": session_id,
                    "skip_intent_check": 1
                }
                if request_body.card_number_pool:
                    payload["card_number_pool"] = request_body.card_number_pool
                if request_body.hl_ymd:
                    payload["hl_ymd"] = request_body.hl_ymd
                payload["sign"] = generate_signature(payload, app_secret)
            else:
                # 正常走奇门
                logger.info("--- 双重校验通过，调用奇门接口 ---")
                target_url = url_qimen
                prefix_message_to_client = "【根据您的询问，启用奇门遁甲进行择时/择事分析】\n\n正在推演中...\n"
                current_intent = "qimen"

                payload = {
                    "appid": "yingshi_appid",
                    "prompt": normalized_prompt,
                    "format": "json",
                    "ftime": current_time,
                    "session_id": session_id,
                    "skip_intent_check": 1  # 奇门下游也跳过再次意图识别
                }
                if request_body.card_number_pool:
                    payload["card_number_pool"] = request_body.card_number_pool
                if request_body.hl_ymd:
                    payload["hl_ymd"] = request_body.hl_ymd
                payload["sign"] = generate_signature(payload, app_secret)

        elif intent_type == "specific_short_term":
            # 【功能1】检测模型切换，给出提示
            switch_notice = ""
            if last_intent and last_intent == "ziwei":
                switch_notice = "【提示：检测到您的问题更适合智慧卡分析，已为您切换至智慧卡模式】\n\n"
                logger.info(f"[会话: {session_id}] 模型切换: 紫微 → 智慧卡")
            
            logger.info("--- 查询分类为 [具体、短期事件咨询], 调用智慧卡接口 ---")
            target_url = url_leipai
            prefix_message_to_client = switch_notice + "【根据您的询问事项，我们启用智慧卡的方式给您建议】\n\n正在推演中...\n"
            current_intent = "leipai"  # 设置本次意图
            
            payload = {
                "appid": "yingshi_appid",
                "prompt": normalized_prompt,  # 使用标准化后的文本
                "format": "json",
                "ftime": current_time,
                "session_id": session_id,
                "skip_intent_check": 1  # 【修复】跳过智慧卡服务的意图识别，避免重复判断
            }
            if request_body.card_number_pool:
                payload["card_number_pool"] = request_body.card_number_pool
            if request_body.hl_ymd:
                payload["hl_ymd"] = request_body.hl_ymd
            payload["sign"] = generate_signature(payload, app_secret)

        else:  # 默认为 'general_long_term'
            # 【功能1】检测模型切换，给出提示
            switch_notice = ""
            if last_intent and last_intent == "leipai":
                switch_notice = "【提示：根据您的问题涉及运势趋势分析，已为您切换至紫微斗数模式】\n\n"
                logger.info(f"[会话: {session_id}] 模型切换: 智慧卡 → 紫微")
            
            logger.info("--- 查询分类为 [宏观、长期人生解读], 调用紫微斗数接口 ---")
            target_url = url_ziwei
            prefix_message_to_client = switch_notice + "【根据您的询问事项，我们启用紫微斗数进行运势的剖析和事项的前瞻遇事】\n\n正在推演中...\n"
            current_intent = "ziwei"  # 设置本次意图
            
            payload = {
                "appid": "yingshi_appid",
                "query": normalized_prompt,  # 使用标准化后的文本
                "session_id": session_id,
                "timestamp": str(current_time),
                "skip_intent_check": True,
                "summary_intent_type": intent_type,  # 传递Summary的意图类型
                "summary_intent_reason": intent_reason  # 传递Summary的分类理由
            }
            payload["sign"] = generate_signature(payload, app_secret)
    
    else:
        # 【兜底逻辑】没有检测到格式化出生信息
        # 【修复】即使没有出生信息，也要进行LLM意图识别，以检测违法、政治等敏感内容
        logger.info("--- 未检测到格式化出生信息，先进行LLM意图识别检测敏感内容 ---")
        
        # 初始化意图变量
        intent_type: Optional[str] = None
        intent_reason: Optional[str] = None
        # 时间范围识别结果（来自并行请求）
        time_range_result: Optional[Dict[str, Any]] = None
        is_historical_event: bool = False
        # 奇门提取结果（来自并行请求）
        qimen_extraction_result: Optional[Dict[str, Any]] = None
        
        # 检测模糊追问
        ambiguous_keywords = ["我应该怎么做", "怎么做", "该怎么办", "继续", "呢", "展开", "详细说说"]
        is_ambiguous = any(kw in normalized_prompt for kw in ambiguous_keywords) and len(normalized_prompt) < 30
        
        if is_ambiguous and last_intent and last_intent in ["ziwei", "leipai"]:
            logger.info(f"--- 检测到模糊追问，直接使用历史意图 [{last_intent}] ---")
            intent_type = "general_long_term" if last_intent == "ziwei" else "specific_short_term"
            intent_reason = f"模糊追问，继承历史意图{last_intent}"
        else:
            try:
                db_pool = getattr(request.app.state, "db_pool", None)
                # 在进行意图分类、时间范围识别和奇门提取前，先去除出生信息（即使没有检测到格式化出生信息，也可能有非格式化出生信息）
                question_only = remove_birth_info(normalized_prompt)
                logger.info(f"[无出生信息分支-意图识别] 原始输入: {normalized_prompt}")
                logger.info(f"[无出生信息分支-意图识别] 去除出生信息后: {question_only}")
                
                with StepMonitor("串行请求时间范围识别，然后并行请求意图分类和奇门提取", request_id, {"phase": "no_birth_info"}):
                    # 第一步：先串行执行时间范围识别
                    time_range_result = await extract_time_range_with_llm(
                        request.app,
                        question_only,
                    )
                    
                    # 从时间范围识别结果中提取时间跨度类型
                    time_span_type = time_range_result.get("time_span_type") if isinstance(time_range_result, dict) else None
                    logger.info(f"[无出生信息分支] 时间范围识别完成，time_span_type={time_span_type}")
                    
                    # 第二步：并行调用意图分类和奇门提取（意图分类使用时间跨度类型）
                    intent_task = classify_query_intent_with_llm(
                        request.app,
                        question_only,
                        db_pool=db_pool,
                        time_span_type=time_span_type,
                    )
                    qimen_task = extract_qimen_with_llm(
                        request.app,
                        question_only,
                        db_pool=db_pool,
                    )
                    query_intent_result, qimen_extraction_result = await asyncio.gather(
                        intent_task, qimen_task, return_exceptions=True
                    )
                
                # 处理意图分类结果
                if isinstance(query_intent_result, Exception):
                    raise query_intent_result
                logger.info(f"[无出生信息分支] LLM 意图分类成功: {query_intent_result}")
                intent_type = query_intent_result.get("query_type")
                intent_reason = query_intent_result.get("reason", "")
                
                # 处理时间范围识别结果
                if isinstance(time_range_result, Exception):
                    logger.warning(f"[无出生信息分支] 时间范围识别失败，使用默认结果（非历史事件）: {time_range_result}")
                    time_range_result = {"has_time_range": False, "end_date": None, "time_expression": None, "time_span_type": "long_term", "reason": "提取失败，默认time_span_type为long_term"}
                logger.info(f"[无出生信息分支] 时间范围识别结果: {time_range_result}")
                
                # 判断是否为历史事件：如果提取到时间范围，且结束日期早于当前日期，则为历史事件
                is_historical_event = False
                if time_range_result.get("has_time_range") and time_range_result.get("end_date"):
                    try:
                        end_date_str = time_range_result.get("end_date")
                        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
                        current_date = datetime.now().date()
                        if end_date < current_date:
                            is_historical_event = True
                            logger.info(f"[无出生信息分支] 判断为历史事件：结束日期 {end_date_str} 早于当前日期 {current_date.strftime('%Y-%m-%d')}")
                        else:
                            logger.info(f"[无出生信息分支] 判断为非历史事件：结束日期 {end_date_str} 晚于或等于当前日期 {current_date.strftime('%Y-%m-%d')}")
                    except Exception as e:
                        logger.warning(f"[无出生信息分支] 解析时间范围结束日期失败: {e}，默认视为非历史事件")
                
                # 处理奇门提取结果（现在是列表）
                if isinstance(qimen_extraction_result, Exception):
                    logger.warning(f"[无出生信息分支] 奇门提取失败，使用默认结果: {qimen_extraction_result}")
                    qimen_extraction_result = [{"is_qimen": False, "qimen_type": None, "matched_tag": None, "reason": "提取失败"}]
                elif not isinstance(qimen_extraction_result, list):
                    # 兼容旧版本（单个结果转为列表）
                    qimen_extraction_result = [qimen_extraction_result] if qimen_extraction_result else [{"is_qimen": False, "qimen_type": None, "matched_tag": None, "reason": "无结果"}]
                logger.info(f"[无出生信息分支] 奇门提取结果（共{len(qimen_extraction_result)}次）: {qimen_extraction_result}")
                
            except Exception as e:
                logger.warning(f"[无出生信息分支] 意图分类失败，将启用兜底方案。错误: {e}", exc_info=True)
                # 兜底方案考虑历史意图
                if last_intent:
                    intent_type = "specific_short_term" if last_intent == "leipai" else "general_long_term"
                    intent_reason = f"意图分类失败，兜底使用历史意图{last_intent}"
                else:
                    intent_type = "specific_short_term"
                    intent_reason = "意图分类失败，兜底使用智慧卡"
                qimen_extraction_result = {"is_qimen": False, "qimen_type": None, "matched_tag": None, "reason": "兜底"}
        
        # 【新增】助手自我介绍类意图，固定回复（无出生信息分支）
        if intent_type == "self_intro":
            intro_response = "我是您的AI生活小助手，集传统文化智慧与现代AI技术于一体，为您提供传统万年历解读、每日运势宜忌及日常养生指南。让千年智慧融入您的生活，在虚实之间揭开未来的迷雾。"
            await store_last_intent(session_id, "self_intro")
            await history_manager.add_messages([
                HumanMessage(content=normalized_prompt),
                AIMessage(content=intro_response)
            ])

            async def intro_stream():
                yield intro_response.encode('utf-8')
                yield b"\n[DONE]\n"

            log_step(
                "成功输出给客户端",
                request_id,
                {"session_id": session_id, "reason": "self_intro"},
            )
            return StreamingResponse(intro_stream(), media_type='text/plain')
        
        # 【新增】非紫微体系类意图，固定回复（无出生信息分支）
        if intent_type == "non_ziwei_system":
            nzs_response = """抱歉，您的问题我无法回答。我是专注于命理运势分析的AI助手，您提出的问题超出了我的服务范围。

我的专长领域包括：
- 紫微斗数运势分析
- 智慧卡指引
- 命理专业知识解答
- 奇门遁甲事时决策

如果您有关于个人运势、命理方面的问题，欢迎随时向我咨询。"""
            await store_last_intent(session_id, "non_ziwei_system_refused")
            await history_manager.add_messages([
                HumanMessage(content=normalized_prompt),
                AIMessage(content=nzs_response)
            ])

            async def nzs_stream():
                yield nzs_response.encode('utf-8')
                yield b"\n[DONE]\n"

            log_step(
                "成功输出给客户端",
                request_id,
                {"session_id": session_id, "reason": "non_ziwei_system_refused"},
            )
            return StreamingResponse(nzs_stream(), media_type='text/plain')
        
        # 【关键】检测违法内容 - 即使没有出生信息也要检测
        if intent_type == "illegal_content":
            logger.info("网关说：--- [无出生信息分支] 查询分类为 [违法犯罪内容], 拒绝服务 ---")
            illegal_response = """抱歉，您的问题我无法回答。"""
            
            await store_last_intent(session_id, "illegal_refused")
            await history_manager.add_messages([
                HumanMessage(content=normalized_prompt),
                AIMessage(content=illegal_response)
            ])
            
            async def illegal_stream():    
                yield illegal_response.encode('utf-8')
                yield b"\n[DONE]\n"
            
            log_step(
                "成功输出给客户端",
                request_id,
                {"session_id": session_id, "reason": "intent_illegal_refused"},
            )
            return StreamingResponse(illegal_stream(), media_type='text/plain')
        
        # 【新增】常识性知识检测
        elif intent_type == "general_knowledge":
            logger.info("网关说：--- [无出生信息分支] 查询分类为 [常识性知识], 拒绝服务 ---")
            knowledge_response = """抱歉，您的问题我无法回答。我是专注于命理运势分析的AI助手，您提出的问题超出了我的服务范围。

我的专长领域包括：
- 紫微斗数运势分析
- 智慧卡指引
- 命理专业知识解答

如果您有关于个人运势、命理方面的问题，欢迎随时向我咨询。"""
            
            await store_last_intent(session_id, "general_knowledge_refused")
            await history_manager.add_messages([
                HumanMessage(content=normalized_prompt),
                AIMessage(content=knowledge_response)
            ])
            
            async def knowledge_stream():    
                yield knowledge_response.encode('utf-8')
                yield b"\n[DONE]\n"
            
            log_step(
                "成功输出给客户端",
                request_id,
                {"session_id": session_id, "reason": "intent_general_knowledge_refused"},
            )
            return StreamingResponse(knowledge_stream(), media_type='text/plain')
        
        # 【问题4优化】检测历史事件 - 基于时间范围识别结果判断（无出生信息分支）
        if is_historical_event:
            logger.info("网关说：--- [无出生信息分支] 根据时间范围识别判断为 [历史事件], 拒绝推演 ---")
            historical_response = "抱歉，您的问题我无法回答。我专注于未来运势的推演和指导。对于已经发生的历史事件，我无法进行回溯性分析。如果您想了解未来的运势趋势，请提出关于未来时间的问题。"
            
            await store_last_intent(session_id, "historical")
            await history_manager.add_messages([
                HumanMessage(content=normalized_prompt),
                AIMessage(content=historical_response)
            ])
            
            async def historical_stream():
                yield historical_response.encode('utf-8')
                yield b"\n[DONE]\n"
            
            log_step(
                "成功输出给客户端",
                request_id,
                {"session_id": session_id, "reason": "time_range_historical_refused", "end_date": time_range_result.get("end_date")},
            )
            return StreamingResponse(historical_stream(), media_type='text/plain')
        
        # 【修复】根据意图优先决定路由，只在意图不明确时才参考历史
        # 优先级：意图识别结果 > 历史意图
        if intent_type == "qimen":
            # 【双重校验】意图分类判断为qimen时，结合奇门提取结果进行二次校验
            should_fallback_to_leipai = False
            selected_result = None
            
            # 处理奇门提取结果列表（5次并行请求的结果）
            qimen_results = qimen_extraction_result if isinstance(qimen_extraction_result, list) else [qimen_extraction_result] if qimen_extraction_result else []
            
            logger.info(f"[无出生信息分支] 意图分类=qimen, 奇门提取结果数量: {len(qimen_results)}")
            
            # 检查是否有任何一次提取确认为奇门
            qimen_results_filtered = [r for r in qimen_results if r.get("is_qimen", False)]
            
            if not qimen_results_filtered:
                logger.info("[无出生信息分支] 所有奇门提取结果都不认为是奇门，降级走雷牌")
                should_fallback_to_leipai = True
            else:
                logger.info(f"[无出生信息分支] 有 {len(qimen_results_filtered)} 次提取确认为奇门，开始处理")
                
                # 检查是否有类型3（类型3直接放行）
                type3_results = [r for r in qimen_results_filtered if r.get("qimen_type") == 3]
                if type3_results:
                    logger.info("[无出生信息分支] 发现类型3结果，无需标签校验，直接走奇门")
                    selected_result = type3_results[0]
                else:
                    # 检查类型1/2，看是否有能匹配数据库标签的
                    type12_results = [r for r in qimen_results_filtered if r.get("qimen_type") in (1, 2)]
                    if not type12_results:
                        logger.info("[无出生信息分支] 没有类型1/2的结果，降级走雷牌")
                        should_fallback_to_leipai = True
                    else:
                        logger.info(f"[无出生信息分支] 有 {len(type12_results)} 次类型1/2的结果，开始标签匹配校验")
                        try:
                            db_pool = getattr(request.app.state, "db_pool", None)
                            valid_tags = await get_all_tags_from_db_async(db_pool)
                            logger.info(f"[无出生信息分支] 数据库标签列表有 {len(valid_tags)} 个标签")
                            
                            # 查找第一个能匹配数据库标签的结果
                            matched_result = None
                            for result in type12_results:
                                matched_tag = result.get("matched_tag")
                                qimen_type = result.get("qimen_type")
                                if matched_tag and matched_tag in valid_tags:
                                    logger.info(f"[无出生信息分支] ✅ 找到匹配结果！类型{qimen_type}，标签 [{matched_tag}] 在数据库中")
                                    matched_result = result
                                    break
                                else:
                                    logger.info(f"[无出生信息分支] ❌ 类型{qimen_type}，标签 [{matched_tag}] 不在数据库中")
                            
                            if matched_result:
                                selected_result = matched_result
                                logger.info("[无出生信息分支] 使用匹配的结果，放行走奇门")
                            else:
                                logger.info("[无出生信息分支] 所有类型1/2的结果都无法匹配数据库标签，降级走雷牌")
                                should_fallback_to_leipai = True
                        except Exception as e:
                            logger.warning(f"[无出生信息分支] 标签校验失败: {e}，将降级走雷牌", exc_info=True)
                            should_fallback_to_leipai = True
            
            # 如果还没有选择结果，使用第一个奇门结果（用于日志记录）
            if not selected_result and qimen_results_filtered:
                selected_result = qimen_results_filtered[0]
            
            if selected_result:
                logger.info(f"[无出生信息分支] 最终选择的奇门提取结果: type={selected_result.get('qimen_type')}, tag={selected_result.get('matched_tag')}")
            
            if should_fallback_to_leipai:
                # 降级到雷牌
                logger.info(f"[会话: {session_id}] 奇门降级：调用智慧卡接口")
                target_url = url_leipai
                prefix_message_to_client = "【根据您的询问事项，我们启用智慧卡的方式给您建议】\n\n正在推演中...\n"
                current_intent = "leipai"
                
                payload = {
                    "appid": "yingshi_appid",
                    "prompt": normalized_prompt,
                    "format": "json",
                    "ftime": current_time,
                    "session_id": session_id,
                    "skip_intent_check": 1
                }
                if request_body.card_number_pool:
                    payload["card_number_pool"] = request_body.card_number_pool
                if request_body.hl_ymd:
                    payload["hl_ymd"] = request_body.hl_ymd
                payload["sign"] = generate_signature(payload, app_secret)
            else:
                # 正常走奇门
                logger.info(f"[会话: {session_id}] 双重校验通过，使用奇门接口")
            target_url = url_qimen
            prefix_message_to_client = "【根据您的询问，启用奇门遁甲进行择时/择事分析】\n\n正在推演中...\n"
            current_intent = "qimen"

            payload = {
                "appid": "yingshi_appid",
                "prompt": normalized_prompt,
                "format": "json",
                "ftime": current_time,
                "session_id": session_id,
                "skip_intent_check": 1
            }
            if request_body.card_number_pool:
                payload["card_number_pool"] = request_body.card_number_pool
            if request_body.hl_ymd:
                payload["hl_ymd"] = request_body.hl_ymd
            payload["sign"] = generate_signature(payload, app_secret)
        elif intent_type == "general_long_term":
            # LLM明确判定适合紫微
            logger.info(f"[会话: {session_id}] 基于意图识别结果（general_long_term），使用紫微")
            target_url = url_ziwei
            prefix_message_to_client = "【根据您的询问事项，我们启用紫微斗数进行运势的剖析和事项的前瞻遇事】\n\n正在推演中...\n"
            current_intent = "ziwei"
            
            payload = {
                "appid": "yingshi_appid",
                "query": normalized_prompt,
                "session_id": session_id,
                "timestamp": str(current_time),
                "skip_intent_check": True,  # 【修复】添加skip_intent_check参数
                "summary_intent_type": intent_type,  # 传递Summary的意图类型
                "summary_intent_reason": intent_reason  # 传递Summary的分类理由
            }
            payload["sign"] = generate_signature(payload, app_secret)
        elif intent_type == "specific_short_term":
            # LLM明确判定适合智慧卡
            logger.info(f"[会话: {session_id}] 基于意图识别结果（specific_short_term），使用智慧卡，并设置 skip_intent_check=1")
            target_url = url_leipai
            prefix_message_to_client = "【根据您的询问事项，我们启用智慧卡的方式给您建议】\n\n正在推演中...\n"
            current_intent = "leipai"
            
            payload = {
                "appid": "yingshi_appid",
                "prompt": normalized_prompt,
                "format": "json",
                "ftime": current_time,
                "session_id": session_id,
                "skip_intent_check": 1  # 智慧卡使用int类型
            }
            if request_body.card_number_pool:
                payload["card_number_pool"] = request_body.card_number_pool
            if request_body.hl_ymd:
                payload["hl_ymd"] = request_body.hl_ymd
            payload["sign"] = generate_signature(payload, app_secret)
        elif last_intent == "ziwei":
            # 意图不明确，但历史是紫微，继续使用紫微
            logger.info(f"[会话: {session_id}] 意图不明确，基于历史意图（ziwei），使用紫微")
            target_url = url_ziwei
            prefix_message_to_client = "【根据您的询问事项，我们启用紫微斗数进行运势的剖析和事项的前瞻遇事】\n\n正在推演中...\n"
            current_intent = "ziwei"
            
            payload = {
                "appid": "yingshi_appid",
                "query": normalized_prompt,
                "session_id": session_id,
                "timestamp": str(current_time),
                "skip_intent_check": True,  # 【修复】添加skip_intent_check参数
                "summary_intent_type": intent_type if intent_type else "general_long_term",  # 传递Summary的意图类型（如果没有则默认general_long_term）
                "summary_intent_reason": intent_reason if intent_reason else "基于历史意图继续使用紫微"  # 传递Summary的分类理由
            }
            payload["sign"] = generate_signature(payload, app_secret)
        else:
            # 默认使用智慧卡（包括 last_intent 为 None 或 leipai 的情况）
            logger.info(f"[会话: {session_id}] 默认使用智慧卡（无明确意图或历史），并设置 skip_intent_check=1")
            target_url = url_leipai
            prefix_message_to_client = "【根据您的询问事项，我们启用智慧卡的方式给您建议】\n\n正在推演中...\n"
            current_intent = "leipai"
            
            payload = {
                "appid": "yingshi_appid",
                "prompt": normalized_prompt,
                "format": "json",
                "ftime": current_time,
                "session_id": session_id,
                "skip_intent_check": 1  # 智慧卡使用int类型
            }
            if request_body.card_number_pool:
                payload["card_number_pool"] = request_body.card_number_pool
            if request_body.hl_ymd:
                payload["hl_ymd"] = request_body.hl_ymd
            payload["sign"] = generate_signature(payload, app_secret)

    # --- 步骤 3: 发送请求到最终选定的下游服务 ---
    logger.info(f"[会话: {session_id}] 【调试】最终 current_intent = {current_intent}")
    logger.info(f"最终决定将请求转发至: {target_url}")
    logger.info(f"最终请求体: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    if target_url == url_ziwei:
        step_label = "成功调用ziweiapi"
    elif target_url == url_leipai:
        step_label = "成功调用智慧卡API"
    elif target_url == url_qimen:
        step_label = "成功调用奇门API"
    else:
        step_label = "成功调用下游服务"
    try:
        with StepMonitor(step_label, request_id, {"target_url": target_url, "session_id": session_id}):
            response = await asyncio.to_thread(
                requests.post,
                target_url,
                json=payload,
                headers=headers,
                stream=True,
                timeout=LLM_REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()

        log_step(
            "成功收到llm返回信息",
            request_id,
            {"status_code": response.status_code, "target_url": target_url},
        )

        logger.info(f"网关说：成功从 {target_url} 收到响应，开始流式返回...")
        
        # 【修改2】使用增强版流式响应，支持保存历史
        return StreamingResponse(
            stream_response_with_history(
                response, 
                prefix_message_to_client,
                session_id,
                normalized_prompt,
                current_intent,
                history_manager,
                request_id,
            ),
            media_type=response_media_type
        )

    except requests.exceptions.HTTPError as e:
        logger.error(f"网关说：下游服务调用失败: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"下游服务错误: {e.response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"网关说：下游服务调用期间发生网络或请求错误: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"无法连接到下游服务: {e}")
    except Exception as e:
        logger.critical(f"网关说：发生意外错误: {type(e).__name__} - {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"内部服务器错误: {e}")