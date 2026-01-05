# -*- coding: utf-8 -*-
# 这是一个基于 FastAPI 的 LLM 调用服务，支持并发处理请求并与 VLLM 交互。
# 新增功能：
#   - 对用户问题进行分类，区分“选择类问题”和“非选择类问题”。
#   - 为选择类问题（如“买A还是B”）提供全新的、基于多牌阵对比的解读逻辑和输出格式。
#   - 重构了抽牌和提示词生成逻辑，以支持两种不同的问题类型。

import asyncio
import aiohttp
import openai
import os
import random
import pandas as pd
import json
import time
import hmac
import hashlib
import logging
import ast
import re  # 导入正则表达式库
import aiomysql
import calendar
from datetime import datetime, timedelta, date
from datetime import date, datetime
from typing import Optional, List, AsyncGenerator, Dict, Any
from fastapi import FastAPI, HTTPException, Request, status
from prompt_logic import generate_non_choice_prompt, generate_choice_prompt, _draw_and_get_card_data
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from langchain_core.messages import HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware
from config import VLLM_API_BASE_URL,VLLM_MODEL_NAME,DB_CONFIG,API_KEY
from app.monitor import StepMonitor, log_step, generate_request_id


# 导入session_manager
from session_manager import initialize_session_manager, close_session_manager, get_session_history

# 导入验证规则和意图识别
from validation_rules import (
    is_gibberish,
    detect_critical_time_selection,
    detect_sensitive_political_content,
    detect_finance_or_lottery,
)
from queryIntent import (
    answer_knowledge_question_suggestion,
    answer_qimen_suggestion,
    answer_self_intro,
    answer_non_ziwei_system,
    classify_query_intent_with_llm,
    classify_qimen_with_llm,
    get_all_tags_from_db_async,
    detect_time_range_with_llm,
    is_historical_time,
)

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


VLLM_MAX_RETRIES = 5
VLLM_RETRY_DELAY_SECONDS = 5.0
VLLM_REQUEST_TIMEOUT_SECONDS = float(os.getenv("VLLM_REQUEST_TIMEOUT_SECONDS", 1800.0))

db_pool: Optional[aiomysql.Pool] = None  # 【新增】数据库连接池的全局变量


# --- 签名密钥配置 ---
APP_SECRETS: Dict[str, str] = {
    "yingshi_appid": "zhongzhoullm",
    "test_app": "test_secret_key"
}

# --- CSV 数据文件路径 ---
# 使用相对于当前文件所在目录的绝对路径，避免工作目录变化导致找不到文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARDS_CSV_PATH = os.path.join(BASE_DIR, '卡牌vs牌号.csv')
MEANINGS_CSV_PATH = os.path.join(BASE_DIR, '雷牌信息集合.csv')

renomann_cards_df: Optional[pd.DataFrame] = None
renomann_meanings_df: Optional[pd.DataFrame] = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 或明确写上 "http://192.168.1.101:5500"
    allow_methods=["*"],
    allow_headers=["*"],
)

async_aiohttp_client: Optional[aiohttp.ClientSession] = None
VLLM_CONCURRENT_LIMIT = 200
VLLM_SLOT_WAIT_TIMEOUT_SECONDS = 1500
vllm_semaphore: Optional[asyncio.Semaphore] = None
next_request_id_counter = 0


# --- Pydantic 请求模型 (保持不变) ---
class ClientRequest(BaseModel):
    appid: str = Field(..., description="应用ID")
    prompt: str = Field(..., description="用户的问题，将用于生成LLM提示词")
    format: str = Field("json", description="响应格式，默认为json")
    ftime: int = Field(..., description="时间戳 (整数)，用于签名验证")
    sign: str = Field(..., description="请求签名，用于验证请求完整性")
    session_id: Optional[str] = Field(None, description="会话ID")
    hl_ymd: Optional[str] = Field(None, description="可选的日期参数")
    skip_intent_check: int = Field(0, description="是否跳过意图识别，0=不跳过（默认），1=跳过直接进行智慧卡占卜")
    card_number_pool: Optional[List[int]] = Field(
        None,
        description="可选的卡牌编号列表，将从这个列表中随机抽取3个数字作为卡牌编号。如果未提供、列表无效，或提供了无法解析的字符串，则从所有可用卡牌中抽取。"
    )

    @validator('card_number_pool', pre=True, always=True)
    def parse_and_validate_card_number_pool(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            try:
                v = ast.literal_eval(v)
            except (ValueError, SyntaxError) as e:
                raise ValueError("card_number_pool: 输入字符串不是有效的列表字面量。")
        if not isinstance(v, list):
            raise ValueError('card_number_pool 必须是列表或表示列表的有效字符串。')
        if len(v) < 3:
            raise ValueError('card_number_pool 必须包含至少 3 个数字')
        if not all(isinstance(i, int) for i in v):
            raise ValueError('card_number_pool 必须只包含整数')
        return v


# --- 签名生成函数 (保持不变) ---
def generate_signature(params: Dict[str, Any], app_secret: str) -> str:
    sorted_params = dict(
        sorted({k: str(v) for k, v in params.items() if k not in ['sign', 'card_number_pool', 'hl_ymd', 'skip_intent_check']}.items()))
    string_to_sign = "".join(f"{k}{v}" for k, v in sorted_params.items())
    secret_bytes = app_secret.encode('utf-8')
    string_to_sign_bytes = string_to_sign.encode('utf-8')
    hmac_sha256 = hmac.new(secret_bytes, string_to_sign_bytes, hashlib.sha256)
    calculated_sign = hmac_sha256.hexdigest()
    return calculated_sign


# --- 应用启动和关闭事件 (保持不变) ---
@app.on_event("startup")
async def startup_event():
    global async_aiohttp_client, vllm_semaphore
    logger.info("FastAPI 应用启动中...")
    
    # 【功能2】初始化session_manager
    await initialize_session_manager()
    
    if not load_csv_data():
        logger.error("CSV 数据加载失败，应用可能无法正常工作。")

    connector = aiohttp.TCPConnector(limit=1000, limit_per_host=1000, enable_cleanup_closed=True, keepalive_timeout=60)
    async_aiohttp_client = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(
            total=VLLM_REQUEST_TIMEOUT_SECONDS,
            connect=VLLM_REQUEST_TIMEOUT_SECONDS,
            sock_read=None
        )
    )
    vllm_semaphore = asyncio.Semaphore(VLLM_CONCURRENT_LIMIT)

    # --- 【新增】初始化数据库连接池 ---
    try:
        # 将连接池直接赋值给 app.state 的一个属性，例如 db_pool
        app.state.db_pool = await aiomysql.create_pool(**DB_CONFIG)
        logger.info("数据库连接池创建成功并储存在 app.state 中。")
    except Exception as e:
        logger.critical(f"创建数据库连接池失败: {e}", exc_info=True)
        # 如果失败，也给 app.state.db_pool 赋一个 None 值
        app.state.db_pool = None

    # --- 【新增】启动提示词文件监控（热更新功能）---
    from prompt_logic import start_prompt_file_watcher
    prompt_watcher = start_prompt_file_watcher()
    if prompt_watcher:
        app.state.prompt_watcher = prompt_watcher
        logger.info("✓ 提示词热更新功能已启用")
    else:
        logger.info("提示词热更新功能未启用，修改提示词需要重启容器")

    logger.info("FastAPI 应用启动完成。")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI 应用关闭中...")
    
    # 【功能2】关闭session_manager
    await close_session_manager()
    
    if async_aiohttp_client:
        await async_aiohttp_client.close()

    # --- 【新增】关闭数据库连接池 ---
    db_pool = getattr(app.state, 'db_pool', None)
    if db_pool:
        db_pool.close()
        await db_pool.wait_closed()
        logger.info("数据库连接池已关闭。")

    logger.info("FastAPI 应用关闭完成。")


# --- CSV 数据加载函数 (保持不变) ---
def load_csv_data():
    global renomann_cards_df, renomann_meanings_df
    try:
        renomann_cards_df = pd.read_csv(CARDS_CSV_PATH, encoding='gbk')
        renomann_meanings_df = pd.read_csv(MEANINGS_CSV_PATH, encoding='gbk')
        logger.info("成功加载 CSV 数据。")
        return True
    except Exception as e:
        logger.error(f"加载 CSV 文件时发生错误: {e}")
        return False


# --- 【新增】数据库日志记录函数 ---
async def save_history_async(history_manager, user_question: str, ai_response: str, request_id: str):
    """异步保存历史对话到Redis"""
    with StepMonitor("会话历史写入", request_id=request_id) as monitor:
        try:
            await history_manager.add_messages([
                HumanMessage(content=user_question),
                AIMessage(content=ai_response)
            ])
            logger.info("[智慧卡] 历史对话已保存到Redis")
        except Exception as e:
            monitor.update_status("失败")
            logger.error(f"[智慧卡] 保存历史失败: {e}")


async def log_qa_to_db(db_pool: Optional[aiomysql.Pool], session_id: str, app_id: str, user_query: str, final_response: str, request_id: str):
    """
    将一次完整的问答记录异步写入数据库。
    """
    # 【改造】不再需要 global, 直接检查传入的参数
    if not db_pool:
        logger.error(f"无法记录日志到数据库：数据库连接池不可用。 Session: {session_id}")
        log_step("数据库日志写入", request_id, {"reason": "pool unavailable"}, status="跳过")
        return
    
    # ... (后面的代码不变) ...
    if not final_response or not final_response.strip():
        logger.warning(f"最终响应为空，取消数据库日志记录。 Session: {session_id}")
        log_step("数据库日志写入", request_id, {"reason": "final_response empty"}, status="跳过")
        return
    with StepMonitor("数据库日志写入", request_id=request_id, extra_data={"session": session_id}) as monitor:
        try:
            async with db_pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = """
                    INSERT INTO qa_logs (session_id, app_id, user_query, final_response)
                    VALUES (%s, %s, %s, %s)
                    """
                    await cursor.execute(sql, (session_id, app_id, user_query, final_response))
            logger.info(f"成功将日志写入数据库。 Session: {session_id}")
        except Exception as e:
            monitor.update_status("失败")
            logger.error(f"将会话 {session_id} 写入数据库时发生错误: {e}", exc_info=True)
            return




# --- 新增：问题分类和选项提取的辅助函数 ---
def is_choice_question(prompt: str) -> bool:
    """
    通过关键词判断用户提问是否为选择类问题。
    """
    choice_keywords = ["还是", "或者", "选哪个"]
    return any(keyword in prompt for keyword in choice_keywords)


async def extract_options_with_llm(user_prompt: str, request_id: str) -> List[str]:
    """
    调用 LLM 来分析用户提问，并以 JSON 格式返回其正在考虑的选项。
    这是一个非流式的、一次性的请求-响应调用。
    """
    global async_aiohttp_client
    if not async_aiohttp_client:
        logger.error(f"请求 {request_id}: LLM 选项提取失败，因为 aiohttp 客户端未初始化。")
        return []

    # 为选项提取任务专门设计的提示词
    extraction_prompt = f"""
<prompt>
    <role>
    你是一个顶级的NLU（自然语言理解）专家，你的任务是从用户的提问中，精准地识别出他/她正在纠结的、需要做出选择的几个具体选项。
    </role>

    <input_data>
        <user_question>{user_prompt}</user_question>
    </input_data>

    <output_instructions>
        <rule>你的唯一输出必须是一个严格的 JSON 对象。</rule>
        <rule>如果用户的问题是一个选择题（包含两个或以上的选项），请将所有选项作为字符串放入一个名为 "options" 的 JSON 数组中。</rule>
        <rule>如果用户的问题不是一个选择题，或者你无法识别出明确的选项，请返回一个空的 "options" 数组。</rule>
        <rule>选项应保持原意，简洁明了。例如，从“是买小米手机还是华为手机呢”中，提取出 "小米手机" 和 "华为手机"。</rule>

        <format_example>
            - 用户问题: "我应该买小米还是华为？" -> 输出: {{"options": ["小米", "华为"]}}
            - 用户问题: "我好纠结，是去北京的大公司，还是回老家考公务员呢？" -> 输出: {{"options": ["去北京的大公司", "回老家考公务员"]}}
            - 用户问题: "旅游目的地，选云南、四川还是西藏？" -> 输出: {{"options": ["云南", "四川", "西藏"]}}
            - 用户问题: "我最近财运怎么样？" -> 输出: {{"options": []}}
            - 用户问题: "他到底喜不喜欢我？" -> 输出: {{"options": []}}
        </format_example>
    </output_instructions>

    <final_command>
    现在，请分析以上用户问题，并严格按照指示输出 JSON 对象。
    </final_command>
</prompt>
/no_think
"""

    messages = [{"role": "user", "content": extraction_prompt}]
    llm_payload = {
        "model": VLLM_MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,  # 使用低温确保输出的稳定性
        "stream": False,  # 非流式
        "response_format": {"type": "json_object"}  # 请求 JSON 输出
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        'Content-Type': 'application/json'}

    with StepMonitor("选项提取", request_id=str(request_id)) as monitor:
        try:
            logger.info(f"请求 {request_id}: 开始调用 LLM 进行选项提取...")
            async with async_aiohttp_client.post(
                    f"{VLLM_API_BASE_URL}/chat/completions",
                    json=llm_payload,
                    headers=headers,
                    timeout=30  # 为此任务设置一个较短的超时
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content:
                        try:
                            # 解析LLM返回的JSON字符串
                            data = json.loads(content)
                            options = data.get("options", [])
                            monitor.update_extra(option_count=len(options))
                            if isinstance(options, list):
                                logger.info(f"请求 {request_id}: LLM 成功提取选项: {options}")
                                return options
                        except json.JSONDecodeError:
                            monitor.update_status("解析失败")
                            logger.error(f"请求 {request_id}: 无法解析LLM返回的选项JSON: {content}")
                            return []
                else:
                    response_text = await response.text()
                    monitor.update_status("调用失败")
                    logger.error(
                        f"请求 {request_id}: LLM 选项提取API调用失败，状态码: {response.status}, 响应: {response_text[:200]}")
                    return []
        except Exception as e:
            monitor.update_status("异常")
            monitor.update_extra(error=str(e))
            logger.error(f"请求 {request_id}: 在调用 LLM 进行选项提取时发生异常: {e}", exc_info=True)
            return []

    return []


def _clean_birth_info_from_prompt(prompt: str) -> str:
    """
    使用正则表达式从用户提问中移除特定格式的个人出生信息。
    
    要移除的格式示例:
    - "公历1996-08-20 12:00:00 男"
    - "农历 1990-01-15 08:30:00 女"
    - "1996-08-20 12:00:00男" (没有公历/农历和空格)
    """

    pattern = re.compile(r"(公历|农历)?\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s*(男|女)")
    
    # 使用 re.sub() 查找所有匹配项，并用空字符串替换它们
    cleaned_prompt = pattern.sub("", prompt)
    
    # 移除替换后可能留下的多余空格
    # 例如 "  今天爱情运势如何" -> "今天爱情运势如何"
    return cleaned_prompt.strip()


def _normalize_vague_time_expressions(cleaned_question: str) -> str:
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


# --- VLLM 流式响应函数 (保持不变) ---
async def stream_vllm_response_with_retry(prompt: str, request_id: str) -> AsyncGenerator[str, None]:
    global async_aiohttp_client
    if async_aiohttp_client is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="错误: 服务内部错误，客户端未准备好。")

    messages = [{"role": "user", "content": prompt}]
    llm_payload = {"model": VLLM_MODEL_NAME, "messages": messages, "temperature": 0.3, "stream": True}
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        'Content-Type': 'application/json'}
    attempt = 0
    last_error_detail = "未知错误"

    with StepMonitor("VLLM流式调用", request_id=str(request_id)) as monitor:
        while attempt <= VLLM_MAX_RETRIES:
            try:
                if attempt > 0:
                    logger.info(f"请求 {request_id}: 第 {attempt} 次重试调用 VLLM...")
                    await asyncio.sleep(VLLM_RETRY_DELAY_SECONDS)

                logger.info(f"请求 {request_id}: 开始向 VLLM 发送请求 (尝试 {attempt + 1}/{VLLM_MAX_RETRIES + 1})...")
                async with async_aiohttp_client.post(f"{VLLM_API_BASE_URL}/chat/completions", json=llm_payload,
                                                     headers=headers) as response:
                    if response.status != status.HTTP_200_OK:
                        response_text = await response.text()
                        last_error_detail = f"HTTP 状态码: {response.status}, 响应: {response_text[:200]}"
                        logger.warning(f"请求 {request_id}: 调用 VLLM 返回非 200 状态码: {last_error_detail}")
                        attempt += 1
                        continue

                    logger.info(f"请求 {request_id}: 调用 VLLM 连接成功，状态码 {response.status}。")
                    buffer = b''
                    async for chunk in response.content.iter_any():
                        buffer += chunk
                        while b'\n' in buffer:
                            line_bytes, buffer = buffer.split(b'\n', 1)
                            line = line_bytes.decode('utf-8').strip()
                            if line.startswith("data:"):
                                data_json_str = line[len("data:"):].strip()
                                if data_json_str == "[DONE]":
                                    monitor.update_extra(final_attempt=attempt + 1)
                                    logger.info(f"请求 {request_id}: 收到 VLLM 流结束信号 [DONE]。")
                                    return  # 正常结束
                                try:
                                    payload = json.loads(data_json_str)
                                    content = payload.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    if content:
                                        yield content
                                except (json.JSONDecodeError, IndexError):
                                    logger.warning(f"请求 {request_id}: 无法解析 VLLM 数据块: {data_json_str}")
                                    continue

                    logger.info(f"请求 {request_id}: VLLM 响应流结束。")
                    monitor.update_extra(final_attempt=attempt + 1)
                    return

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_error_detail = f"网络或客户端错误: {type(e).__name__} - {e}"
                monitor.update_extra(last_error=last_error_detail, attempt=attempt + 1)
                logger.error(f"请求 {request_id}: 调用 VLLM 失败 (尝试 {attempt + 1}): {last_error_detail}")
                attempt += 1
            except Exception as e:
                monitor.update_status("异常")
                monitor.update_extra(error=str(e))
                logger.critical(f"请求 {request_id}: 调用 VLLM 期间发生未知错误: {e}", exc_info=True)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    detail=f"调用 LLM 服务期间发生意外错误: {e}")

    logger.error(f"请求 {request_id}: 达到最大 VLLM 重试次数，最终失败。最后错误: {last_error_detail}")
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"错误: 调用 LLM 服务失败，已重试 {VLLM_MAX_RETRIES} 次。")


# --- FastAPI 接口端点 (核心改造) ---
@app.post("/chat_endpoints_V12_25")
async def chat_endpoint(client_request: ClientRequest, request: Request):
    global next_request_id_counter
    request_start_time = time.perf_counter()
    req_id = client_request.session_id or generate_request_id()
    log_step(
        "请求入站",
        request_id=req_id,
        extra_data={
            "appid": client_request.appid,
            "skip_intent_check": client_request.skip_intent_check,
        },
        status="开始",
    )
    next_request_id_counter += 1

    db_pool_from_state = getattr(request.app.state, 'db_pool', None)
    
    # 【功能2】获取会话历史
    history_manager = await get_session_history(req_id)
    log_step(
        "会话管理初始化",
        req_id,
        {"manager": history_manager.__class__.__name__},
    )
    
    try:
        # 1. 签名验证
        logger.info(f"请求 {req_id}: 收到新请求, AppID: '{client_request.appid}', 问题: '{client_request.prompt}', skip_intent_check: {client_request.skip_intent_check}")
        with StepMonitor("签名验证", request_id=req_id):
            app_secret = APP_SECRETS.get(client_request.appid)
            if not app_secret:
                raise HTTPException(status_code=401, detail="未授权: 无效的 AppID。")
            params_for_sig = client_request.dict(exclude={'sign', 'card_number_pool', 'hl_ymd', 'skip_intent_check'})
            if generate_signature(params_for_sig, app_secret) != client_request.sign:
                raise HTTPException(status_code=403, detail="禁止访问: 签名验证失败。")
        logger.info(f"请求 {req_id}: 签名验证成功。")

        with StepMonitor("输入清洗", request_id=req_id) as monitor:
            original_prompt = client_request.prompt
            user_question = _clean_birth_info_from_prompt(original_prompt)
            monitor.update_extra(original_length=len(original_prompt), cleaned_length=len(user_question))
        logger.info(f"请求 {req_id}: 清洗后的问题: '{user_question}'")
        
        # ==========================================================
        # 【新增】步骤 1.5: 前置内容检测和意图识别
        # ==========================================================
        
        # 政治敏感内容检测
        with StepMonitor("政治敏感检测", request_id=req_id) as monitor:
            hit_political = detect_sensitive_political_content(user_question)
            monitor.update_extra(hit=hit_political)
            if hit_political:
                monitor.update_status("命中规则")
                political_response = "抱歉，您的问题我无法回答。根据相关规定，我无法讨论或分析任何与政治、军事相关的话题。我的能力范围仅限于基于智慧卡的具体事件预测和指引。"
                
                async def political_stream():
                         
                    yield political_response
                    yield "[DONE]"
                
                log_step("输出给客户端", req_id, {"reason": "政治敏感"}, status="拒绝")
                return StreamingResponse(political_stream(), media_type="text/plain; charset=utf-8")
        
        # 乱码检测
        with StepMonitor("乱码检测", request_id=req_id) as monitor:
            gibberish = is_gibberish(user_question)
            monitor.update_extra(hit=gibberish)
            if gibberish:
                monitor.update_status("命中规则")
                guidance_response = """抱歉，您的问题我无法回答。我没能理解您的问题。

为了更好地帮助您，请尝试提出清晰的具体问题，例如：
- "我这次面试能否成功？"
- "我最近一个月的财运如何？"
- "我应该接受这个工作机会吗？"
- "这件事情该怎么办？"

请重新描述您的问题，我会为您进行专业的占卜分析。"""
            
                async def guidance_stream():
                         
                    yield guidance_response
                    yield "[DONE]"
                
                log_step("输出给客户端", req_id, {"reason": "乱码检测未通过"}, status="拒绝")
                return StreamingResponse(guidance_stream(), media_type="text/plain; charset=utf-8")
        
        # 金融/彩票/投资类固定回复（需在重大时间点拦截之前）
        with StepMonitor("金融投资检测", request_id=req_id) as monitor:
            hit_finance = detect_finance_or_lottery(user_question)
            monitor.update_extra(hit=hit_finance)
            if hit_finance:
                monitor.update_status("命中规则")
                finance_response = """您好，我的专长在于通过命理工具为您提供人生趋势的洞察与个人决策的参考，而非金融市场分析，我无法也绝不会提供任何具体的金融投资建议，包括对个股、基金或其他金融产品的买卖建议。理财有风险，投资需谨慎。"""

                async def finance_stream():
                     
                    yield finance_response
                    yield "[DONE]"

                log_step("输出给客户端", req_id, {"reason": "金融/彩票/投资限制"}, status="拒绝")
                return StreamingResponse(finance_stream(), media_type="text/plain; charset=utf-8")
        
        # 重大时间点拦截
        with StepMonitor("重大时间点检测", request_id=req_id) as monitor:
            is_critical, category = detect_critical_time_selection(user_question)
            monitor.update_extra(hit=is_critical, category=category)
            if is_critical:
                monitor.update_status("命中规则")
                refusal_response = f"""抱歉，您的问题我无法回答。关于【{category}】这类重大事项的具体时间选择，建议您：

1. 咨询相关领域的专业人士（医生、律师、财务顾问等）
2. 综合考虑实际情况和专业建议
3. 不要仅依赖占卜推算做出重大决策

占卜分析可以作为参考，但涉及健康、法律、重大财务等事项时，专业判断更为重要。

如果您想了解该时期的整体运势趋势，我可以为您分析。"""
            
                async def refusal_stream():
                         
                    yield refusal_response
                    yield "[DONE]"
                
                log_step("输出给客户端", req_id, {"reason": "重大时间点限制", "category": category}, status="拒绝")
                return StreamingResponse(refusal_stream(), media_type="text/plain; charset=utf-8")
        
        # 意图识别 + 奇门识别：判断是否更适合紫薇/奇门，或是历史事件/知识问答
        # 【新增】如果 skip_intent_check=1，则跳过意图识别和奇门识别，直接进行智慧卡占卜
        qimen_result: Dict[str, Any] = {
            "is_qimen": False,
            "qimen_type": None,
            "matched_tag": None,
            "reason": "未进行奇门识别",
        }
        if client_request.skip_intent_check == 1:
            logger.info(f"[智慧卡服务] 请求 {req_id}: skip_intent_check=1，跳过意图及奇门识别，直接进行智慧卡占卜")
            log_step("意图识别", req_id, {"skip_intent_check": True}, status="跳过")
        else:
            try:
                logger.info(f"[智慧卡服务] 请求 {req_id}: 开始意图识别与奇门识别...")

                # 先尝试从数据库获取奇门“具体事项”标签列表（带 36 小时缓存）
                qimen_tags: List[str] = []
                if db_pool_from_state:
                    try:
                        qimen_tags = await get_all_tags_from_db_async(db_pool_from_state)
                    except Exception as e:
                        logger.warning(f"[智慧卡服务] 请求 {req_id}: 获取奇门具体事项标签失败，将按无标签处理: {e}")

                # 【新增】对用户问题进行时间表达规范化，仅用于意图分类
                # 将模糊时间表达（如"未来几天"）替换为具体数字（如"未来三天"）
                # 注意：user_question 已经在前面清洗过出生信息，这里确保用于意图分类的问题也清洗过
                cleaned_for_intent = _clean_birth_info_from_prompt(user_question)
                # intent_classification_question = _normalize_vague_time_expressions(cleaned_for_intent)
                intent_classification_question = cleaned_for_intent
                logger.info(f"[智慧卡服务] 请求 {req_id}: 意图分类用问题（已规范化，已清洗出生信息）: '{intent_classification_question}'")
                
                # 第一步：先执行时间范围识别（串行）
                # 注意：所有任务都使用清洗过出生信息的问题（user_question 已在前面清洗过）
                # 为了确保万无一失，在调用前再次明确清洗
                cleaned_for_time_range = _clean_birth_info_from_prompt(user_question)
                cleaned_for_qimen = _clean_birth_info_from_prompt(user_question)
                logger.info(f"[智慧卡服务] 请求 {req_id}: 时间范围识别用问题（已清洗出生信息）: '{cleaned_for_time_range}'")
                logger.info(f"[智慧卡服务] 请求 {req_id}: 奇门识别用问题（已清洗出生信息）: '{cleaned_for_qimen}'")
                
                with StepMonitor("时间范围识别", request_id=req_id) as time_monitor:
                    try:
                        time_range_result = await detect_time_range_with_llm(
                            cleaned_for_time_range,  # 时间范围识别使用清洗后的问题
                            async_aiohttp_client,
                            vllm_semaphore,
                        )
                        logger.info(f"[智慧卡服务] 请求 {req_id}: 时间范围识别结果: {time_range_result}")
                    except Exception as e:
                        logger.warning(f"[智慧卡服务] 请求 {req_id}: 时间范围识别任务失败，将按无时间范围处理: {e}")
                        time_range_result = {"has_time_range": False, "end_date": None, "time_expression": None, "time_duration": "long", "reason": "时间范围识别任务异常，默认判断为长时间"}
                        time_monitor.update_status("失败")
                    
                    # 【新增】优先判断是否为历史时间
                    if is_historical_time(time_range_result):
                        time_monitor.update_status("命中历史时间")
                        logger.info(f"[智慧卡服务] 请求 {req_id}: 检测到历史时间范围，返回历史事件描述")
                        historical_response = "抱歉，您的问题我无法回答。我专注于未来运势的推演和指导。对于已经发生的历史事件，我无法进行回溯性分析。如果您想了解未来的运势趋势，请提出关于未来时间的问题。"
                        
                        async def historical_stream():
                            yield historical_response
                            yield "[DONE]"
                        
                        log_step("输出给客户端", req_id, {"reason": "历史时间范围"}, status="拒绝")
                        return StreamingResponse(historical_stream(), media_type="text/plain; charset=utf-8")
                
                # 第二步：提取 time_duration，然后执行意图识别和奇门识别（并行）
                # time_duration 现在总是有值（"short" 或 "long"），如果没有提取到则默认为 "long"
                time_duration = time_range_result.get("time_duration", "long")
                logger.info(f"[智慧卡服务] 请求 {req_id}: 提取的时间跨度类型: {time_duration}")
                
                intent_task = asyncio.create_task(
                    classify_query_intent_with_llm(
                        intent_classification_question,  # 使用规范化后的问题（已清洗出生信息）
                        async_aiohttp_client,
                        vllm_semaphore,
                        time_duration=time_duration,  # 传入时间跨度类型
                    )
                )
                qimen_task = asyncio.create_task(
                    classify_qimen_with_llm(
                        cleaned_for_qimen,  # 奇门识别使用清洗后的问题
                        qimen_tags,
                        async_aiohttp_client,
                        vllm_semaphore,
                    )
                )

                with StepMonitor("意图识别", request_id=req_id) as monitor:
                    # 等待意图识别和奇门识别完成
                    query_intent_result, qimen_result_task = await asyncio.gather(
                        intent_task,
                        qimen_task,
                        return_exceptions=True
                    )
                    
                    # 处理意图识别结果
                    if isinstance(query_intent_result, Exception):
                        logger.error(f"[智慧卡服务] 请求 {req_id}: 意图识别任务失败: {query_intent_result}")
                        raise query_intent_result
                    logger.info(f"[智慧卡服务] 请求 {req_id}: 意图识别结果: {query_intent_result}")
                    
                    # 处理奇门识别结果
                    if isinstance(qimen_result_task, Exception):
                        logger.warning(f"[智慧卡服务] 请求 {req_id}: 奇门识别任务失败，将按非奇门处理: {qimen_result_task}")
                        qimen_result = {
                            "is_qimen": False,
                            "qimen_type": None,
                            "matched_tag": None,
                            "reason": "奇门识别任务异常，按非奇门处理",
                        }
                    else:
                        qimen_result = qimen_result_task
                        logger.info(f"[智慧卡服务] 请求 {req_id}: 奇门识别结果: {qimen_result}")

                    intent_type = query_intent_result.get("query_type")
                    intent_reason = query_intent_result.get("reason", "")
                    monitor.update_extra(intent_type=intent_type)
                    
                    # 【新增】处理违法内容 - 最高优先级
                    if intent_type == "illegal_content":
                        monitor.update_status("命中违法内容")
                        logger.info(f"[智慧卡服务] 请求 {req_id}: 检测到违法犯罪内容，拒绝服务")
                        illegal_response = """抱歉，您的问题我无法回答。"""
                        
                        async def illegal_stream():
                                 
                            yield illegal_response
                            yield "[DONE]"
                        
                        log_step("输出给客户端", req_id, {"reason": "违法内容"}, status="拒绝")
                        return StreamingResponse(illegal_stream(), media_type="text/plain; charset=utf-8")
                    
                    # 【新增】处理常识性知识问题 - 第二优先级
                    elif intent_type == "general_knowledge":
                        monitor.update_status("命中常识问题")
                        logger.info(f"[智慧卡服务] 请求 {req_id}: 检测到常识性知识问题，拒绝服务")
                        knowledge_response = """抱歉，您的问题我无法回答。我是专注于智慧卡和运势分析的AI助手，您提出的问题超出了我的服务范围。

我的专长领域包括：
- 智慧卡分析
- 具体事件预测和指引
- 命理相关咨询

如果您有关于个人运势、占卜方面的问题，欢迎随时向我咨询。"""
                    
                        async def knowledge_stream():
                                 
                            yield knowledge_response
                            yield "[DONE]"
                        
                        log_step("输出给客户端", req_id, {"reason": "常识问题"}, status="拒绝")
                        return StreamingResponse(knowledge_stream(), media_type="text/plain; charset=utf-8")
                    
                    # 处理知识问答类：建议走紫薇
                    elif intent_type == "knowledge_question":
                        monitor.update_status("建议走紫薇")
                        logger.info(f"[智慧卡服务] 请求 {req_id}: 检测到知识问答类请求，建议使用紫薇服务")
                        knowledge_suggestion = await answer_knowledge_question_suggestion(user_question)
                        
                        async def knowledge_stream():
                                 
                            yield knowledge_suggestion
                            yield "[DONE]"
                        
                        log_step("输出给客户端", req_id, {"reason": "知识问答建议转紫薇"}, status="建议")
                        return StreamingResponse(knowledge_stream(), media_type="text/plain; charset=utf-8")

                    # 处理奇门类：需要同时满足
                    # 1）意图分类结果为 qimen
                    # 2）奇门识别结果 is_qimen 为 True
                    # 3）若 qimen_type 为 1 或 2，则 matched_tag 必须在数据库的具体事项列表中
                    # 若 qimen_type 为 3（不需要具体事项），则只要前两条满足即走奇门
                    elif intent_type == "qimen":
                        is_qimen_llm = bool(qimen_result.get("is_qimen"))
                        qimen_type_val = qimen_result.get("qimen_type")
                        matched_tag = qimen_result.get("matched_tag")

                        # 默认认为具体事项不满足（仅对类型1和2生效）
                        matched_tag_valid = False
                        if qimen_type_val in (1, 2):
                            # 类型1/2 必须匹配到数据库中的具体事项标签，否则不能归类为奇门
                            if matched_tag and matched_tag in qimen_tags:
                                matched_tag_valid = True
                                logger.info(
                                    f"[智慧卡服务] 请求 {req_id}: 奇门类型 {qimen_type_val} 且匹配到有效具体事项标签: {matched_tag}"
                                )
                            else:
                                logger.info(
                                    f"[智慧卡服务] 请求 {req_id}: 奇门类型 {qimen_type_val} 但未匹配到有效具体事项标签，按非奇门处理，继续走雷牌流程。"
                                )
                        elif qimen_type_val == 3:
                            # 类型3 不需要具体事项标签
                            matched_tag_valid = True

                        # 只有“两边都判断为奇门”且具体事项规则满足时，才真正提示走奇门
                        if is_qimen_llm and matched_tag_valid:
                            monitor.update_status("建议走奇门")
                            logger.info(
                                f"[智慧卡服务] 请求 {req_id}: 意图和奇门识别均判断为奇门（类型 {qimen_type_val}），建议使用奇门遁甲服务"
                            )
                            qimen_suggestion = await answer_qimen_suggestion(user_question)

                            async def qimen_stream():
                                 
                                yield qimen_suggestion
                                yield "[DONE]"

                            log_step("输出给客户端", req_id, {"reason": "奇门问题建议转奇门", "qimen_type": qimen_type_val}, status="建议")
                            return StreamingResponse(qimen_stream(), media_type="text/plain; charset=utf-8")
                        # 否则，视为未满足奇门条件，直接继续智慧卡（雷牌）流程
                        else:
                            logger.info(
                                f"[智慧卡服务] 请求 {req_id}: 虽然意图或奇门识别中存在奇门信号，但未同时满足条件（类型1/2缺少有效具体事项或奇门识别为非奇门），继续走智慧卡雷牌流程。"
                            )
                    
                    # 处理自我介绍
                    elif intent_type == "self_intro":
                        monitor.update_status("自我介绍")
                        logger.info(f"[智慧卡服务] 请求 {req_id}: 检测到自我介绍请求")
                        intro_text = await answer_self_intro()

                        async def intro_stream():
                             
                            yield intro_text
                            yield "[DONE]"

                        log_step("输出给客户端", req_id, {"reason": "自我介绍"}, status="完成")
                        return StreamingResponse(intro_stream(), media_type="text/plain; charset=utf-8")

                    # 处理非紫微体系
                    elif intent_type == "non_ziwei_system":
                        monitor.update_status("非紫微体系")
                        logger.info(f"[智慧卡服务] 请求 {req_id}: 检测到非紫微体系请求")
                        non_ziwei_text = await answer_non_ziwei_system()

                        async def non_ziwei_stream():
                             
                            yield non_ziwei_text
                            yield "[DONE]"

                        log_step("输出给客户端", req_id, {"reason": "非紫微体系固定回复"}, status="拒绝")
                        return StreamingResponse(non_ziwei_stream(), media_type="text/plain; charset=utf-8")
                    
                    # 判断是否更适合紫薇
                    elif intent_type == "general_long_term":
                        monitor.update_status("建议紫薇长周期")
                        logger.info(f"[智慧卡服务] 请求 {req_id}: 检测到问题更适合紫薇: {intent_type}")
                        # 清理reason末尾的标点，避免"。，"连接
                        cleaned_reason = intent_reason.rstrip('。，；')
                        ziwei_suggestion = f"""鉴于您的问题「{user_question}」{cleaned_reason}，我们建议您使用紫微斗数服务。

紫微斗数特别适合：
- 长期运势趋势分析（数月到数年）
- 个人性格和命运特质的深入解读
- 整体人生格局和发展方向的把握
- 重大人生决策的全局性参考

智慧卡则更擅长：
- 具体事件的短期预测
- 明确的行动建议和指引
- 快速决策支持

如果您仍希望从智慧卡的角度获得建议，你可以这样提问：使用智慧卡回答...

如果您希望继续询问该问题请转到紫微斗数提问：{user_question}
"""
                    
                        async def ziwei_suggestion_stream():
                                 
                            yield ziwei_suggestion
                            yield "[DONE]"
                        
                        log_step("输出给客户端", req_id, {"reason": "建议使用紫微斗数"}, status="建议")
                        return StreamingResponse(ziwei_suggestion_stream(), media_type="text/plain; charset=utf-8")
                    
            except Exception as e:
                logger.warning(f"[智慧卡服务] 请求 {req_id}: 意图识别失败，继续智慧卡分析流程: {e}")
                log_step("意图识别", req_id, {"error": str(e)}, status="失败")
                # 如果意图识别失败，继续走原有的智慧卡分析流程
        
        # 【功能2】获取历史消息作为上下文
        with StepMonitor("加载上下文", request_id=req_id) as monitor:
            full_history_messages = await history_manager.messages  # 直接调用property
            full_history_messages = full_history_messages or []
            
            # 构建上下文文本（只包含用户问题和AI回答的摘要）
            context_text = ""
            if full_history_messages:
                for i, msg in enumerate(full_history_messages):
                    if isinstance(msg, HumanMessage):
                        context_text += f"上次问题: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        # AI回答只取前100字作为摘要
                        content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                        context_text += f"上次回答摘要: {content_preview}\n"
            monitor.update_extra(history_count=len(full_history_messages), context_length=len(context_text))
        
        logger.info(f"请求 {req_id}: 加载上下文，历史消息数: {len(full_history_messages)}, 上下文长度: {len(context_text)}")

        # 2. **阶段一：调用 LLM 提取选项**
        options = await extract_options_with_llm(user_question, req_id)

        prompt_text = None

        # 3. **根据选项数量，决定处理流程**
        if options and len(options) >= 2:
            # --- 选择题处理流程 ---
            with StepMonitor("选择题抽牌与提示词", request_id=req_id) as monitor:
                logger.info(f"请求 {req_id}: 判定为选择题，选项为 {options}。开始为每个选项抽牌。")
                choices_data = {}
                success = True
                for option in options:
                    # 每个选项固定抽3张牌
                    card_data = _draw_and_get_card_data(renomann_cards_df, renomann_meanings_df, 3,
                                                        client_request.card_number_pool)
                    if not all(card_data):
                        logger.error(f"请求 {req_id}: 未能为选项 '{option}' 抽取卡牌。")
                        success = False
                        break  # 如果任何一个选项抽牌失败，则整个选择题流程中断

                    card_names, _, task_texts = card_data
                    choices_data[option] = {"names": card_names, "texts": task_texts}
                    logger.info(f"请求 {req_id}: 为选项 '{option}' 抽到的牌: {card_names}")

                if success:
                    # 【功能2】拼接上下文到问题前（历史上下文仅用于理解背景，传递给提示词时只使用当前问题）
                    full_question_for_context = f"{context_text}\n当前问题: {user_question}" if context_text else user_question
                    # 传递给提示词模板时，只使用当前问题，避免大模型在报告中提及历史对话
                    prompt_text = generate_choice_prompt(user_question, choices_data)
                    monitor.update_extra(option_count=len(options))
                else:
                    # 如果中途失败，可以决定是报错还是回退到非选择模式
                    logger.warning(f"请求 {req_id}: 因抽牌失败，选择题流程中断，将回退到非选择题模式。")
                    monitor.update_status("抽牌失败回退")
                    # 此处将 options 设为空，以触发下面的非选择题逻辑
                    options = []

        if not options or len(options) < 2:
            # --- 非选择题处理流程 ---
            with StepMonitor("常规抽牌与提示词", request_id=req_id):
                logger.info(f"请求 {req_id}: 判定为非选择题，开始常规抽牌。")
                
                # 【功能2】拼接上下文到问题前（历史上下文仅用于理解背景，传递给提示词时只使用当前问题）
                full_question_for_context = f"{context_text}\n当前问题: {user_question}" if context_text else user_question
                # 传递给提示词模板时，只使用当前问题，避免大模型在报告中提及历史对话
                
                prompt_data = generate_non_choice_prompt(
                    renomann_cards_df, renomann_meanings_df, user_question, client_request.card_number_pool
                )
                prompt_text, _, _ = prompt_data

        if not prompt_text:
            raise HTTPException(status_code=500, detail="生成最终占卜提示词失败。")

        # 4. **阶段二：流式返回占卜结果**
        logger.info(f"请求 {req_id}: 提示词生成完毕，准备进行占卜解读并流式返回。")

        async def generate_stream_and_log():
            """
            这个内部函数现在负责两件事：
            1. 向客户端流式传输数据。
            2. 在流结束后，将完整数据异步写入数据库。
            """
            response_parts = []
            final_response = ""
            try:
                log_step("VLLM流式响应", req_id, {"phase": "start"}, status="开始")
                # 使用async with来确保信号量和超时正确管理
                async with asyncio.timeout(VLLM_SLOT_WAIT_TIMEOUT_SECONDS):
                    async with vllm_semaphore:
                        # 迭代从VLLM获取的流式数据
                        async for chunk in stream_vllm_response_with_retry(prompt_text, req_id):
                            response_parts.append(chunk)
                            yield chunk
                    yield "[DONE]"
                
                # 流式传输结束后，我们在这里有完整的响应
                final_response = "".join(response_parts)
                log_step("VLLM流式响应", req_id, {"phase": "end", "response_length": len(final_response)})

            except Exception as e:
                logger.error(f"请求 {req_id}: 流式响应生成时发生错误: {e}", exc_info=True)
                yield f"错误: 流式响应时发生内部错误。\n"
            finally:
                # 无论流是否成功，finally块都会执行
                session_id = f"{req_id}"
                app_id_to_log = "leinuoman"
                
                # 【功能2】保存历史对话到Redis
                asyncio.create_task(save_history_async(
                    history_manager, original_prompt, final_response, req_id
                ))
                
                # 创建一个后台任务来执行数据库写入
                asyncio.create_task(
                    log_qa_to_db(
                        db_pool=db_pool_from_state,
                        session_id=session_id,
                        app_id=app_id_to_log,
                        user_query=user_question,
                        final_response=final_response,
                        request_id=req_id
                    )
                )
                total_time_ms = round((time.perf_counter() - request_start_time) * 1000, 2)
                log_step("输出给客户端", req_id, {"total_time_ms": total_time_ms})


        return StreamingResponse(generate_stream_and_log(), media_type="text/plain; charset=utf-8")

    except HTTPException as e:
        log_step("输出给客户端", req_id, {"error": str(e.detail)}, status="失败")
        raise e
    except Exception as e:
        logger.critical(f"请求 {req_id}: 处理请求时发生未知严重错误: {e}", exc_info=True)
        log_step("输出给客户端", req_id, {"error": str(e)}, status="失败")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}")


# --- 运行应用说明 (保持不变) ---
# 在本地运行此文件时，可以使用以下命令启动 Uvicorn 服务器：
# uvicorn your_module_name:app --host 0.0.0.0 --port 8000
#
# 其中 your_module_name 是保存此代码的文件名 (不带 .py 后缀)。```

