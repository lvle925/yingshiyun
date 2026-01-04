# -*- coding: utf-8 -*-
"""
奇门遁甲LLM服务主入口
根据用户问题判断是否为奇门问题，并进行相应的处理
"""

import asyncio
import aiohttp
import logging
import time
import re
import hmac
import hashlib
import difflib
import calendar
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, date
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from starlette.concurrency import run_in_threadpool

from config import VLLM_API_BASE_URL, VLLM_MODEL_NAME, API_KEY, DB_CONFIG
from app.monitor import StepMonitor, log_step, generate_request_id
from session_manager import initialize_session_manager, close_session_manager, get_session_history
from validation_rules import is_gibberish, detect_critical_time_selection, detect_sensitive_political_content, detect_finance_investment
from user_info_extractor import extract_user_info, get_day_stem_from_gregorian_date
from query_intent import (
    classify_query_intent_with_llm, 
    classify_second_layer_intent, 
    answer_knowledge_question_suggestion,
    extract_time_range_with_llm,
    start_prompt_file_watcher as start_intent_prompt_watcher
)
from db_query import get_all_tags_from_db_async, query_qimen_data
from llm_response import generate_final_response, start_prompt_file_watcher
import aiomysql

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VLLM_MAX_RETRIES = 5
VLLM_RETRY_DELAY_SECONDS = 5.0
VLLM_REQUEST_TIMEOUT_SECONDS = 1800.0

db_pool: Optional[aiomysql.Pool] = None

# --- 签名密钥配置 ---
APP_SECRETS: Dict[str, str] = {
    "yingshi_appid": "zhongzhoullm",
    "test_app": "test_secret_key"
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async_aiohttp_client: Optional[aiohttp.ClientSession] = None
VLLM_CONCURRENT_LIMIT = 200
VLLM_SLOT_WAIT_TIMEOUT_SECONDS = 1500
vllm_semaphore: Optional[asyncio.Semaphore] = None


# --- Pydantic 请求模型 ---
class ClientRequest(BaseModel):
    appid: str = Field(..., description="应用ID")
    prompt: str = Field(..., description="用户的问题")
    format: str = Field("json", description="响应格式，默认为json")
    ftime: int = Field(..., description="时间戳 (整数)，用于签名验证")
    sign: str = Field(..., description="请求签名，用于验证请求完整性")
    session_id: Optional[str] = Field(None, description="会话ID")
    skip_intent_check: int = Field(0, description="是否跳过意图识别，0=不跳过（默认），1=跳过直接进行奇门分析")


def normalize_fuzzy_time_expressions(cleaned_question: str) -> str:
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



# --- 签名生成函数 ---
def generate_signature(params: Dict[str, Any], app_secret: str) -> str:
    sorted_params = dict(
        sorted({k: str(v) for k, v in params.items() if k not in ['sign', 'skip_intent_check']}.items()))
    string_to_sign = "".join(f"{k}{v}" for k, v in sorted_params.items())
    secret_bytes = app_secret.encode('utf-8')
    string_to_sign_bytes = string_to_sign.encode('utf-8')
    hmac_sha256 = hmac.new(secret_bytes, string_to_sign_bytes, hashlib.sha256)
    calculated_sign = hmac_sha256.hexdigest()
    return calculated_sign


# --- 应用启动和关闭事件 ---
@app.on_event("startup")
async def startup_event():
    global async_aiohttp_client, vllm_semaphore, db_pool
    logger.info("FastAPI 应用启动中...")
    
    # 初始化session_manager
    await initialize_session_manager()
    
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
    
    # 初始化数据库连接池
    try:
        app.state.db_pool = await aiomysql.create_pool(**DB_CONFIG)
        logger.info("数据库连接池创建成功并储存在 app.state 中。")
    except Exception as e:
        logger.critical(f"创建数据库连接池失败: {e}", exc_info=True)
        app.state.db_pool = None
    
    # 启动提示词文件监控（热更新功能）
    intent_prompt_watcher = start_intent_prompt_watcher()
    llm_prompt_watcher = start_prompt_file_watcher()
    if intent_prompt_watcher:
        app.state.intent_prompt_watcher = intent_prompt_watcher
        logger.info("✓ 意图识别提示词热更新功能已启用")
    if llm_prompt_watcher:
        app.state.llm_prompt_watcher = llm_prompt_watcher
        logger.info("✓ LLM响应提示词热更新功能已启用")
    if not intent_prompt_watcher and not llm_prompt_watcher:
        logger.info("提示词热更新功能未启用，修改提示词需要重启容器")
    
    logger.info("FastAPI 应用启动完成。")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI 应用关闭中...")
    
    await close_session_manager()
    
    if async_aiohttp_client:
        await async_aiohttp_client.close()
    
    db_pool = getattr(app.state, 'db_pool', None)
    if db_pool:
        db_pool.close()
        await db_pool.wait_closed()
        logger.info("数据库连接池已关闭。")
    
    logger.info("FastAPI 应用关闭完成。")


# --- 数据库日志记录函数 ---
async def log_qa_to_db(db_pool: Optional[aiomysql.Pool], session_id: str, app_id: str, 
                       user_query: str, final_response: str, request_id: str):
    """将一次完整的问答记录异步写入数据库"""
    if not db_pool:
        logger.error(f"无法记录日志到数据库：数据库连接池不可用。 Session: {session_id}")
        log_step("数据库日志写入", request_id, {"reason": "pool unavailable"}, status="跳过")
        return
    
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


async def save_history_async(history_manager, user_question: str, ai_response: str, request_id: str):
    """异步保存历史对话到Redis"""
    with StepMonitor("会话历史写入", request_id=request_id) as monitor:
        try:
            await history_manager.add_messages([
                HumanMessage(content=user_question),
                AIMessage(content=ai_response)
            ])
            logger.info("[奇门] 历史对话已保存到Redis")
        except Exception as e:
            monitor.update_status("失败")
            logger.error(f"[奇门] 保存历史失败: {e}")


# --- FastAPI 接口端点 ---
@app.post("/chat_qimen")
async def chat_endpoint(client_request: ClientRequest, request: Request):
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
    
    db_pool_from_state = getattr(request.app.state, 'db_pool', None)
    
    # 获取会话历史
    history_manager = await get_session_history(req_id)
    log_step(
        "会话管理初始化",
        req_id,
        {"manager": history_manager.__class__.__name__},
    )
    
    try:
        # 1. 签名验证
        logger.info(f"请求 {req_id}: 收到新请求, AppID: '{client_request.appid}', 问题: '{client_request.prompt}'")
        with StepMonitor("签名验证", request_id=req_id):
            app_secret = APP_SECRETS.get(client_request.appid)
            if not app_secret:
                raise HTTPException(status_code=401, detail="未授权: 无效的 AppID。")
            params_for_sig = client_request.dict(exclude={'sign', 'skip_intent_check'})
            if generate_signature(params_for_sig, app_secret) != client_request.sign:
                raise HTTPException(status_code=403, detail="禁止访问: 签名验证失败。")
        logger.info(f"请求 {req_id}: 签名验证成功。")
        
        # 2. 提取用户信息
        with StepMonitor("用户信息提取", request_id=req_id) as monitor:
            original_prompt = client_request.prompt
            user_info, cleaned_question = extract_user_info(original_prompt)
            monitor.update_extra(has_user_info=user_info is not None)
        logger.info(f"请求 {req_id}: 清洗后的问题: '{cleaned_question}'")

        # 在意图识别与时间范围提取前，统一规范模糊时间表述为具体时间范围
        # cleaned_question = normalize_fuzzy_time_expressions(cleaned_question)


        # 政治敏感内容检测
        with StepMonitor("政治敏感检测", request_id=req_id) as monitor:
            hit_political = detect_sensitive_political_content(cleaned_question)
            monitor.update_extra(hit=hit_political)
            if hit_political:
                monitor.update_status("命中规则")
                political_response = "抱歉，您的问题我无法回答。根据相关规定，我无法讨论或分析任何与政治、军事相关的话题。"
                
                async def political_stream():
                    yield political_response
                    yield "[DONE]"
                
                log_step("输出给客户端", req_id, {"reason": "政治敏感"}, status="拒绝")
                return StreamingResponse(political_stream(), media_type="text/plain; charset=utf-8")
        
        # 乱码检测
        with StepMonitor("乱码检测", request_id=req_id) as monitor:
            gibberish = is_gibberish(cleaned_question)
            monitor.update_extra(hit=gibberish)
            if gibberish:
                monitor.update_status("命中规则")
                guidance_response = """抱歉，您的问题我无法回答。我没能理解您的问题。

为了更好地帮助您，请尝试提出清晰的具体问题，例如：
- "明天下午3点适合去逛街吗？"
- "给我几个今年适合去旅游的时间"
- "明天下午三点适合做什么？"

请重新描述您的问题，我会为您进行专业的奇门分析。"""
            
                async def guidance_stream():
                    yield guidance_response
                    yield "[DONE]"
                
                log_step("输出给客户端", req_id, {"reason": "乱码检测未通过"}, status="拒绝")
                return StreamingResponse(guidance_stream(), media_type="text/plain; charset=utf-8")
        
        # 重大时间点拦截
        # 金融/投资/彩票拦截（在重大事件前）
        with StepMonitor("金融投资拦截", request_id=req_id) as monitor:
            hit_fin = detect_finance_investment(cleaned_question)
            monitor.update_extra(hit=hit_fin)
            if hit_fin:
                monitor.update_status("命中金融投资限制")
                fin_response = """您好，我的专长在于通过命理工具为您提供人生趋势的洞察与个人决策的参考，而非金融市场分析，我无法也绝不会提供任何具体的金融投资建议，包括对个股、基金或其他金融产品的买卖建议。理财有风险，投资需谨慎。"""
                async def fin_stream():
                    yield fin_response
                    yield "[DONE]"
                log_step("输出给客户端", req_id, {"reason": "金融投资限制"}, status="拒绝")
                return StreamingResponse(fin_stream(), media_type="text/plain; charset=utf-8")

        with StepMonitor("重大时间点检测", request_id=req_id) as monitor:
            is_critical, category = detect_critical_time_selection(cleaned_question)
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
        
        # 4. 串行执行时间范围识别和意图识别
        if client_request.skip_intent_check == 1:
            logger.info(f"[奇门服务] 请求 {req_id}: skip_intent_check=1，跳过意图识别和时间范围识别，直接进行奇门分析")
            log_step("意图识别", req_id, {"skip_intent_check": True}, status="跳过")
            log_step("时间范围识别", req_id, {"skip_intent_check": True}, status="跳过")
            intent_result = {"query_type": "qimen", "reason": "跳过意图识别"}
            time_range_result = {"has_time_range": False, "end_date": None, "time_expression": None, "time_duration_type": "long"}
        else:
            try:
                logger.info(f"[奇门服务] 请求 {req_id}: 开始串行执行时间范围识别和意图识别...")
                t_serial_start = time.perf_counter()
                
                # 第一步：先执行时间范围识别
                with StepMonitor("时间范围识别", request_id=req_id) as time_monitor:
                    try:
                        time_range_result = await extract_time_range_with_llm(
                            cleaned_question,
                            async_aiohttp_client,
                            vllm_semaphore
                        )
                        logger.info(f"[奇门服务] 请求 {req_id}: 时间范围识别结果: {time_range_result}")
                    except Exception as e:
                        logger.warning(f"[奇门服务] 请求 {req_id}: 时间范围识别失败: {e}")
                        time_range_result = {
                            "has_time_range": False, 
                            "end_date": None, 
                            "time_expression": None,
                            "time_duration_type": "long"
                        }
                
                # 从时间范围结果中提取时间长短信息
                time_duration_type = time_range_result.get("time_duration_type")
                logger.info(f"[奇门服务] 请求 {req_id}: 提取到的时间长短类型: {time_duration_type}")
                
                # 第二步：执行意图识别，传入时间长短信息
                with StepMonitor("意图识别", request_id=req_id) as intent_monitor:
                    try:
                        intent_result = await classify_query_intent_with_llm(
                            cleaned_question,
                            async_aiohttp_client,
                            vllm_semaphore,
                            time_duration_type=time_duration_type
                        )
                        logger.info(f"[奇门服务] 请求 {req_id}: 意图识别结果: {intent_result}")
                    except Exception as e:
                        logger.warning(f"[奇门服务] 请求 {req_id}: 意图识别失败: {e}")
                        intent_result = {"query_type": "qimen", "reason": "识别失败，默认继续"}
                
                print(f"test LLM1 serial (time->intent) time: {time.perf_counter() - t_serial_start:.2f}s", flush=True)
                
                # 合并监控信息
                with StepMonitor("时间范围识别和意图识别", request_id=req_id) as monitor:
                    
                    # 检查是否为历史时间
                    end_date_str = time_range_result.get("end_date")
                    is_historical = False
                    
                    if end_date_str:
                        try:
                            now_dt = datetime.now()
                            end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
                            # 如果结束日期小于等于当前日期，认为是历史时间
                            if end_dt.date() <= now_dt.date():
                                is_historical = True
                                logger.info(f"[奇门服务] 请求 {req_id}: 检测到历史时间范围，结束日期: {end_date_str}")
                        except Exception as e:
                            logger.warning(f"[奇门服务] 请求 {req_id}: 时间范围解析失败: {e}")
                    
                    # 如果是历史时间，直接返回历史事件响应
                    if is_historical:
                        monitor.update_status("命中历史事件")
                        historical_response = "抱歉，您的问题我无法回答。我专注于未来运势的推演和指导。对于已经发生的历史事件，我无法进行回溯性分析。如果您想了解未来的运势趋势，请提出关于未来时间的问题。"
                        async def historical_stream():
                            yield historical_response
                            yield "[DONE]"
                        log_step("输出给客户端", req_id, {"reason": "历史事件"}, status="拒绝")
                        return StreamingResponse(historical_stream(), media_type="text/plain; charset=utf-8")
                    
                    # 继续处理意图分类结果
                    query_type = intent_result.get("query_type", "specific_short_term")
                    monitor.update_extra(query_type=query_type, time_duration_type=time_duration_type)
                    
                    # 处理各种意图类型
                    if query_type == "illegal_content":
                        monitor.update_status("命中违法内容")
                        illegal_response = "抱歉，您的问题我无法回答。"
                        async def illegal_stream():
                            yield illegal_response
                            yield "[DONE]"
                        log_step("输出给客户端", req_id, {"reason": "违法内容"}, status="拒绝")
                        return StreamingResponse(illegal_stream(), media_type="text/plain; charset=utf-8")
                    
                    elif query_type == "general_knowledge":
                        monitor.update_status("命中常识问题")
                        knowledge_response = """抱歉，您的问题我无法回答。我是专注于奇门遁甲和运势分析的AI助手，您提出的问题超出了我的服务范围。

我的专长领域包括：
- 具体时间点的择时分析
- 通过时间确定适合的事件
- 通过事件确定适合的时间

如果您有关于时间择时、运势方面的问题，欢迎随时向我咨询。"""
                        async def knowledge_stream():
                            yield knowledge_response
                            yield "[DONE]"
                        log_step("输出给客户端", req_id, {"reason": "常识问题"}, status="拒绝")
                        return StreamingResponse(knowledge_stream(), media_type="text/plain; charset=utf-8")
                    
                    elif query_type == "self_intro":
                        monitor.update_status("自我介绍")
                        intro_response = "我是您的AI生活小助手，集传统文化智慧与现代AI技术于一体，为您提供传统万年历解读、每日运势宜忌及日常养生指南。让千年智慧融入您的生活，在虚实之间揭开未来的迷雾。"
                        async def intro_stream():
                            yield intro_response
                            yield "[DONE]"
                        log_step("输出给客户端", req_id, {"reason": "自我介绍"}, status="正常")
                        return StreamingResponse(intro_stream(), media_type="text/plain; charset=utf-8")
                    
                    elif query_type == "non_ziwei_system":
                        monitor.update_status("非本系统问题")
                        nzs_response = """抱歉，您的问题我无法回答。我是专注于命理运势分析的AI助手，您提出的问题超出了我的服务范围。

我的专长领域包括：
- 具体时间点的择时分析
- 通过时间确定适合的事件
- 通过事件确定适合的时间

如果您有关于个人运势、命理方面的问题，欢迎随时向我咨询。"""
                        async def nzs_stream():
                            yield nzs_response
                            yield "[DONE]"
                        log_step("输出给客户端", req_id, {"reason": "非本系统问题"}, status="拒绝")
                        return StreamingResponse(nzs_stream(), media_type="text/plain; charset=utf-8")
                    
                    elif query_type == "knowledge_question":
                        monitor.update_status("建议走紫薇")
                        knowledge_suggestion = await answer_knowledge_question_suggestion(cleaned_question)
                        knowledge_suggestion += "\n\n奇门服务则更擅长：\n- 具体时间点的择时分析\n- 通过时间确定适合的事件\n- 通过事件确定适合的时间"
                        async def knowledge_stream():
                            yield knowledge_suggestion
                            yield "[DONE]"
                        log_step("输出给客户端", req_id, {"reason": "知识问答建议转紫薇"}, status="建议")
                        return StreamingResponse(knowledge_stream(), media_type="text/plain; charset=utf-8")
                    
                    elif query_type == "general_long_term":
                        monitor.update_status("建议紫薇长周期")
                        logger.info(f"[奇门服务] 请求 {req_id}: 检测到问题更适合紫薇: {query_type}")
                        # 清理reason末尾的标点，避免"。，"连接
                        cleaned_reason = intent_result.get("reason", "").rstrip('。，；')
                        ziwei_suggestion = f"""鉴于您的问题「{cleaned_question}」{cleaned_reason}，我们建议您使用紫微斗数服务。

紫微斗数特别适合：
- 长期运势趋势分析（数月到数年）
- 个人性格和命运特质的深入解读
- 整体人生格局和发展方向的把握
- 重大人生决策的全局性参考

奇门服务则更擅长：
- 具体时间点的择时分析
- 通过时间确定适合的事件
- 通过事件确定适合的时间

如果您希望继续询问该问题请转到紫微斗数提问：{cleaned_question}
"""
                        async def ziwei_stream():
                            yield ziwei_suggestion
                            yield "[DONE]"
                        log_step("输出给客户端", req_id, {"reason": "建议使用紫微斗数"}, status="建议")
                        return StreamingResponse(ziwei_stream(), media_type="text/plain; charset=utf-8")
                    
                    elif query_type != "qimen":
                        # 其他类型（如specific_short_term等非奇门短期事件）建议使用智慧卡服务
                        monitor.update_status("建议使用智慧卡")
                        cleaned_reason = intent_result.get("reason", "").rstrip('。，；')
                        leinuo_response = f"""鉴于您的问题「{cleaned_question}」{cleaned_reason}，我们建议您使用智慧卡服务。

智慧卡服务特别适合：
- 具体事件的短期预测
- 明确的行动建议和指引
- 快速决策支持
- 选择类问题的对比分析

奇门服务则更擅长：
- 具体时间点的择时分析
- 通过时间确定适合的事件
- 通过事件确定适合的时间

如果您希望继续询问该问题请转到智慧卡提问：{cleaned_question}
"""
                        async def leinuo_stream():
                            yield leinuo_response
                            yield "[DONE]"
                        log_step("输出给客户端", req_id, {"reason": "非奇门问题-建议智慧卡"}, status="建议")
                        return StreamingResponse(leinuo_stream(), media_type="text/plain; charset=utf-8")
                    
                    # 如果是qimen类型，继续后续处理
            except Exception as e:
                logger.warning(f"[奇门服务] 请求 {req_id}: 意图识别或时间范围识别失败，继续奇门分析流程: {e}")
                log_step("意图识别", req_id, {"error": str(e)}, status="失败")
                intent_result = {"query_type": "qimen", "reason": "识别失败，默认继续"}
                time_range_result = {"has_time_range": False, "end_date": None, "time_expression": None}
        
        # 5. 检查是否为奇门问题
        if intent_result.get("query_type") != "qimen":
            # 如果已经返回了响应，这里不应该执行到
            error_response = "抱歉，您的问题不属于奇门服务范围。"
            async def error_stream():
                yield error_response
                yield "[DONE]"
            return StreamingResponse(error_stream(), media_type="text/plain; charset=utf-8")
        
        # 6. 准备数据：查询所有标签类型
        db_pool_from_state = getattr(request.app.state, 'db_pool', None)
        with StepMonitor("查询标签类型", request_id=req_id):
            t_tags_start = time.perf_counter()
            available_tags = await get_all_tags_from_db_async(db_pool_from_state)
            print(f"test tags fetch time: {time.perf_counter() - t_tags_start:.2f}s", flush=True)
            logger.info(f"test请求-获取到的标签：{available_tags}")
            logger.info(f"请求 {req_id}: 获取到 {len(available_tags)} 个标签")
        
        # 7. 第二层意图识别：判断奇门问题的具体类型并提取时间和事件
        if not user_info:
            error_response = "抱歉，使用奇门服务需要提供您的出生年月日时和性别信息。"
            async def error_stream():
                yield error_response
                yield "[DONE]"
            return StreamingResponse(error_stream(), media_type="text/plain; charset=utf-8")
        
        try:
            logger.info(f"[奇门服务] 请求 {req_id}: 开始第二层意图识别...")
            with StepMonitor("第二层意图识别", request_id=req_id) as monitor:
                current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                t_second_total_start = time.perf_counter()

                def _validate_specific_event(result: dict) -> tuple[dict, bool]:
                    qimen_type = result.get("qimen_type")
                    if qimen_type not in ["type1", "type2"]:
                        return result, True
                    event = result.get("specific_event")
                    if event in available_tags:
                        return result, True
                    # 尝试模糊匹配（优先）
                    if event:
                        match = difflib.get_close_matches(event, available_tags, n=1, cutoff=0.6)
                        if match:
                            result["specific_event"] = match[0]
                            return result, True
                    # 如果标签列表中有词出现在原始问题中，则用该词
                    for tag in available_tags:
                        if tag and tag in cleaned_question:
                            result["specific_event"] = tag
                            return result, True
                    return result, False

                # 并行请求 5 次第二层识别，取首个匹配标签的结果
                async def _call_second_layer(idx: int):
                    t_second_start = time.perf_counter()
                    res = await classify_second_layer_intent(
                        cleaned_question,
                        available_tags,
                        current_time_str,
                        async_aiohttp_client,
                        vllm_semaphore
                    )
                    print(f"test LLM2 second-layer attempt {idx} time: {time.perf_counter() - t_second_start:.2f}s", flush=True)
                    return res

                candidates = await asyncio.gather(*[_call_second_layer(i + 1) for i in range(5)])

                second_layer_result = None
                for c in candidates:
                    c_checked, ok = _validate_specific_event(c)
                    if ok:
                        second_layer_result = c_checked
                        break

                if not second_layer_result:
                    # 尝试对第一个候选做一次原问题标签替换
                    first_cand = candidates[0]
                    first_cand, ok = _validate_specific_event(first_cand)
                    if ok:
                        second_layer_result = first_cand
                    else:
                        monitor.update_status("事件标签不匹配")
                        logger.info(f"[奇门服务] 请求 {req_id}: 第二层意图识别结果（事件无匹配标签）: {first_cand}")
                        
                        # 如果是从上层来的请求（skip_intent_check=1），输出简短回复
                        if client_request.skip_intent_check == 1:
                            no_match_response = "您所提问的时间段诸事不宜，请另外择时。"
                            async def no_match_stream():
                                yield no_match_response
                                yield "[DONE]"
                            log_step("输出给客户端", req_id, {"reason": "事件无匹配奇门标签（上层请求）"}, status="无匹配")
                            return StreamingResponse(no_match_stream(), media_type="text/plain; charset=utf-8")
                        
                        # 否则推荐智慧卡服务
                        wisdom_response = f"""抱歉，您询问的事件在奇门中没有对应的类别，无法给您准确的行动建议，因此我们推荐您使用智慧卡服务。

智慧卡服务特别适合：
- 具体事件的短期预测
- 明确的行动建议和指引
- 快速决策支持
- 选择类问题的对比分析
如果您希望继续询问该问题请转到智慧卡提问：{cleaned_question}
"""
                        async def wisdom_stream():
                            yield wisdom_response
                            yield "[DONE]"
                        log_step("输出给客户端", req_id, {"reason": "事件无匹配奇门标签，推荐智慧卡服务"}, status="建议切换服务")
                        return StreamingResponse(wisdom_stream(), media_type="text/plain; charset=utf-8")

                print(f"test LLM2 total time: {time.perf_counter() - t_second_total_start:.2f}s", flush=True)
                logger.info(f"[奇门服务] 请求 {req_id}: 第二层意图识别结果: {second_layer_result}")

                # 若模型误将“具体时间点做什么事件”(本应为 type3) 标成 type2 且未给出 specific_event，做一次类型纠偏
                # 典型示例：”这个月3号下午我适合做什么事情“ -> qimen_type: type2, specific_event: null
                if second_layer_result:
                    qt = second_layer_result.get("qimen_type")
                    se = second_layer_result.get("specific_event")
                    ts = second_layer_result.get("time_range_start")
                    te = second_layer_result.get("time_range_end")
                    if qt == "type2" and not se and ts and te:
                        logger.info(f"[奇门服务] 请求 {req_id}: 将无具体事件的 type2 纠正为 type3，用于“时间 -> 事件”场景")
                        second_layer_result["qimen_type"] = "type3"

                monitor.update_extra(qimen_type=second_layer_result.get("qimen_type") if second_layer_result else None)
        except Exception as e:
            logger.error(f"[奇门服务] 请求 {req_id}: 第二层意图识别失败: {e}", exc_info=True)
            error_response = "抱歉，无法理解您的问题，请尝试换一种说法。"
            async def error_stream():
                yield error_response
                yield "[DONE]"
            return StreamingResponse(error_stream(), media_type="text/plain; charset=utf-8")

        # 7.5 基于解析出的时间范围做“历史时间段”硬校验
        # 规则：如果第二层给出的 time_range_start 和 time_range_end 都存在，且整个时间段已经完全早于当前时间，
        # 则认为用户在询问已经过去的时间，不再做奇门分析，直接走历史事件的拒答逻辑。
        time_start_str = second_layer_result.get("time_range_start")
        time_end_str = second_layer_result.get("time_range_end")
        if time_start_str and time_end_str:
            try:
                now_dt = datetime.now()
                start_dt = datetime.strptime(time_start_str, "%Y-%m-%d %H:%M:%S")
                end_dt = datetime.strptime(time_end_str, "%Y-%m-%d %H:%M:%S")
                if end_dt < now_dt:
                    past_response = "抱歉，您的问题涉及已经完全过去的时间段，我专注于未来的运势推演与择时指导。请换一个当前或未来的时间范围再来询问。"
                    async def past_stream():
                        yield past_response
                        yield "[DONE]"
                    log_step("输出给客户端", req_id, {"reason": "时间范围已全部过去"}, status="拒绝")
                    return StreamingResponse(past_stream(), media_type="text/plain; charset=utf-8")
            except Exception as e:
                logger.warning(f"[奇门服务] 请求 {req_id}: 历史时间段硬校验失败，继续后续流程: {e}")

        # 8. 数据库查询
        with StepMonitor("数据库查询", request_id=req_id) as monitor:
            t_db_start = time.perf_counter()
            qimen_data = await query_qimen_data(
                qimen_type=second_layer_result.get("qimen_type"),
                specific_event=second_layer_result.get("specific_event"),
                time_range_start=second_layer_result.get("time_range_start"),
                time_range_end=second_layer_result.get("time_range_end"),
                jixiong_preference=second_layer_result.get("jixiong_preference"),
                user_info=user_info
            )
            print(f"test DB query time: {time.perf_counter() - t_db_start:.2f}s", flush=True)
            monitor.update_extra(data_count=len(qimen_data))
            logger.info(f"请求 {req_id}: 数据库查询返回 {len(qimen_data)} 条记录")
            logger.info(f"qimen_data{qimen_data}")
        
        if not qimen_data:
            no_data_response = "该时间段诸事不宜，请择时行动。"
            async def no_data_stream():
                yield no_data_response
                yield "[DONE]"
            return StreamingResponse(no_data_stream(), media_type="text/plain; charset=utf-8")
        
        # 9. 第三次LLM请求：生成最终结果
        logger.info(f"请求 {req_id}: 开始生成最终响应...")
        
        async def generate_stream_and_log():
            """流式返回结果并记录日志"""
            response_parts = []
            final_response = ""
            t_stream_start = time.perf_counter()
            try:
                log_step("LLM最终响应", req_id, {"phase": "start"}, status="开始")
                
                async with asyncio.timeout(VLLM_SLOT_WAIT_TIMEOUT_SECONDS):
                    async with vllm_semaphore:
                        t_llm3_start = time.perf_counter()
                        final_response = await generate_final_response(
                            cleaned_question,
                            qimen_data,
                            second_layer_result.get("qimen_type"),
                            second_layer_result.get("original_time_text"),
                            second_layer_result.get("jixiong_preference"),
                            async_aiohttp_client,
                            vllm_semaphore
                        )
                        print(f"test LLM3 final-response time: {time.perf_counter() - t_llm3_start:.2f}s", flush=True)
                        response_parts.append(final_response)
                        
                        # 流式返回
                        yield final_response
                        yield "[DONE]"
                
                log_step("LLM最终响应", req_id, {"phase": "end", "response_length": len(final_response)})
                
            except Exception as e:
                logger.error(f"请求 {req_id}: 生成最终响应时发生错误: {e}", exc_info=True)
                yield f"错误: 生成响应时发生内部错误。\n"
            finally:
                print(f"test stream+post time: {time.perf_counter() - t_stream_start:.2f}s", flush=True)
                # 保存历史对话和数据库日志
                session_id = f"{req_id}"
                app_id_to_log = "qimen"
                
                asyncio.create_task(save_history_async(
                    history_manager, original_prompt, final_response, req_id
                ))
                
                asyncio.create_task(
                    log_qa_to_db(
                        db_pool=db_pool_from_state,
                        session_id=session_id,
                        app_id=app_id_to_log,
                        user_query=cleaned_question,
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

