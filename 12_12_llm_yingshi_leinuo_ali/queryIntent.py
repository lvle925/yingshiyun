import asyncio
import logging
import json
import os
import re
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Optional

import aiohttp
import pandas as pd

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field, ValidationError, validator

from config import API_KEY, VLLM_API_BASE_URL, VLLM_MODEL_NAME

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - 说: %(message)s')
logger = logging.getLogger(__name__)


VLLM_SLOT_WAIT_TIMEOUT_SECONDS = 1500
VLLM_REQUEST_TIMEOUT_SECONDS = 500

INTENT_PROMPT_FILE_PATH = Path(__file__).resolve().parent / "prompts" / "intent_classification_prompt.xml"
QIMEN_PROMPT_FILE_PATH = Path(__file__).resolve().parent / "prompts" / "qimen_extraction_prompt.xml"
TIME_RANGE_PROMPT_FILE_PATH = Path(__file__).resolve().parent / "prompts" / "time_range_detection_prompt.xml"


def load_intent_system_prompt() -> str:
    """读取意图分类提示词模板内容（包含占位符 {current_time}）。"""
    try:
        content = INTENT_PROMPT_FILE_PATH.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.error("读取意图分类提示词文件失败: %s", exc)
        raise

    if not content:
        raise ValueError(f"提示词文件为空: {INTENT_PROMPT_FILE_PATH}")

    return content


def load_qimen_system_prompt_template() -> str:
    """读取奇门提取提示词模板内容（包含占位符 {tags_text} 和 {current_time}）。"""
    try:
        content = QIMEN_PROMPT_FILE_PATH.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.error("读取奇门提取提示词文件失败: %s", exc)
        raise

    if not content:
        raise ValueError(f"提示词文件为空: {QIMEN_PROMPT_FILE_PATH}")

    return content


def load_time_range_system_prompt_template() -> str:
    """读取时间范围识别提示词模板内容（包含占位符 {current_time}）。"""
    try:
        content = TIME_RANGE_PROMPT_FILE_PATH.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.error("读取时间范围识别提示词文件失败: %s", exc)
        raise

    if not content:
        raise ValueError(f"提示词文件为空: {TIME_RANGE_PROMPT_FILE_PATH}")

    return content


INTENT_PROMPT_TEMPLATE = load_intent_system_prompt()
QIMEN_PROMPT_TEMPLATE = load_qimen_system_prompt_template()
TIME_RANGE_PROMPT_TEMPLATE = load_time_range_system_prompt_template()


class QueryIntent(BaseModel):
    """
    用于验证 LLM 对用户问题意图分类结果的 Pydantic 模型。
    新增 'illegal_content' 用于识别违法犯罪相关内容。
    新增 'general_knowledge' 用于识别常识性知识或具体知识型问题。
    新增 'qimen' 用于识别奇门遁甲相关的择时或事件匹配类问题。
    新增 'self_intro' 用于识别自我介绍类询问。
    新增 'non_ziwei_system' 用于识别非紫微体系的请求。
    """
    query_type: Literal[
        "specific_short_term",
        "general_long_term",
        "knowledge_question",
        "illegal_content",
        "general_knowledge",
        "qimen",
        "self_intro",
        "non_ziwei_system",
    ] = Field(
        ...,
        description="问题的分类结果。'specific_short_term' 表示具体短期事件咨询；'general_long_term' 表示宏观长期解读；'knowledge_question' 表示命理专业知识问题；'illegal_content' 表示涉及违法犯罪的内容；'general_knowledge' 表示常识性知识或具体知识型问题；'qimen' 表示需要奇门遁甲择时或事件匹配的请求；'self_intro' 表示询问助手自我介绍；'non_ziwei_system' 表示非紫微体系的命理/玄学请求。"
    )
    reason: str = Field(..., description="做出分类判断的简要理由。但在输出的理由中：1. 不要提及字段名称（如'specific_short_term'、'general_long_term'等）；2. 不要提及技术术语（如'time_duration'、'标记'、'字段'、'标签'等）；3. 不要提及【因此归类为...】类似的话语。只说明分类的理由即可，使用自然语言描述。")


class QimenExtraction(BaseModel):
    """
    奇门类型识别与具体事项提取结果模型。
    - is_qimen: 是否属于奇门三种类型之一
    - qimen_type: 1/2/3，分别对应三种奇门类型；非奇门或未识别时为 None
    - matched_tag: 从数据库「具体事项」标签列表中匹配到的标签；仅当 qimen_type in (1,2) 时可能有值
    """

    is_qimen: bool = Field(..., description="是否属于奇门遁甲类型")
    qimen_type: Optional[int] = Field(
        None,
        description="奇门类型：1=具体时间点做具体事件是否合适；2=什么时间做具体事件；3=具体时间点做什么事件；非奇门时为null",
        ge=1,
        le=3,
    )
    matched_tag: Optional[str] = Field(
        None,
        description="从具体事项标签列表中匹配到的标签；仅当 is_qimen=true 且 qimen_type 为1或2时可能有值",
    )
    reason: str = Field(..., description="简要说明判断依据")


class TimeRangeDetection(BaseModel):
    """
    时间范围识别结果模型。
    - has_time_range: 是否识别到时间范围
    - end_date: 结束日期（ISO 8601格式：YYYY-MM-DD），如果没有则为null
    - time_expression: 用户问题中的原始时间表达（如果有）
    - time_duration: 时间跨度类型，"short"表示一个月内（<=30天），"long"表示一个月以上（>30天）或未提及时间
    - reason: 识别理由
    """

    has_time_range: bool = Field(..., description="是否识别到时间范围")
    end_date: Optional[str] = Field(
        None,
        description="结束日期（ISO 8601格式：YYYY-MM-DD），如果没有则为null",
    )
    time_expression: Optional[str] = Field(
        None,
        description="用户问题中的原始时间表达（如果有）",
    )
    time_duration: Literal["short", "long"] = Field(
        ...,
        description="时间跨度类型：'short'表示一个月内（<=30天），'long'表示一个月以上（>30天）或未提及时间。注意：只要不是短时间就是长时间，如果没有提及时间就代表是长时间",
    )
    reason: str = Field(..., description="简要说明提取的时间范围和判断依据")


# 注意：原有的aiohttp_vllm_invoke函数已被移除，
# 现在直接在classify_query_intent_with_llm函数中调用VLLM

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ 核心修改: 更新 Pydantic 模型和 LLM 分类函数 +++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

async def answer_knowledge_question_suggestion(cleaned_question) -> str:
    """
    【智慧卡服务专用】对于知识问答类请求，建议用户使用紫薇服务
    """
    return f"""您的问题涉及命理专业知识，建议您使用紫微斗数服务获得更专业的解答。

紫微斗数特别适合：
- 专业术语和概念的详细解释
- 命盘结构的深入理解
- 长期运势和人生格局的分析

智慧卡则更擅长：
- 具体事件的短期预测
- 实际问题的明确指引
- 决策建议和行动方向

如果您希望继续询问该问题请转到紫微斗数提问：{cleaned_question}
"""


async def answer_qimen_suggestion(cleaned_question) -> str:
    """
    【智慧卡服务专用】对于奇门类请求，建议用户使用奇门遁甲服务
    """
    return f"""您的问题涉及具体择时择事的专业判断，建议您使用奇门遁甲服务获得更专业的解答。

奇门服务则更擅长：
- 具体时间点的择时分析
- 通过时间确定适合的事件
- 通过事件确定适合的时间

智慧卡服务特别适合：
- 具体事件的短期预测
- 明确的行动建议和指引
- 快速决策支持
- 选择类问题的对比分析

如果您希望继续询问该问题请转到奇门服务提问：{cleaned_question}
"""


async def answer_self_intro() -> str:
    """
    自我介绍回复
    """
    return "我是您的AI生活小助手，集传统文化智慧与现代AI技术于一体，为您提供传统万年历解读、每日运势宜忌及日常养生指南。让千年智慧融入您的生活，在虚实之间揭开未来的迷雾。"


async def answer_non_ziwei_system() -> str:
    """
    非紫微体系固定回复
    """
    return """抱歉，您的问题我无法回答。我是专注于命理运势分析的AI助手，您提出的问题超出了我的服务范围。

我的专长领域包括：
- 具体事件的短期预测
- 明确的行动建议和指引
- 快速决策支持
- 选择类问题的对比分析

如果您有关于个人运势、命理方面的问题，欢迎随时向我咨询。"""


async def classify_query_intent_with_llm(
    user_input: str, 
    async_client: aiohttp.ClientSession, 
    semaphore: asyncio.Semaphore, 
    time_duration: str = "long",
    max_retries: int = 5
) -> Dict[str, Any]:
    """
    使用 LLM 对用户问题的意图进行分类（适配智慧卡服务）。
    - specific_short_term: 针对具体、短期事件。
    - general_long_term: 针对宏观、长期问题 (包括择色、择方位、择吉数)。
    - self_intro: 自我介绍请求。
    
    参数:
        user_input: 用户输入
        async_client: aiohttp客户端会话
        semaphore: VLLM并发控制信号量
        time_duration: 时间跨度类型，"short"表示一个月内，"long"表示一个月以上或未识别到时间范围（默认值）
        max_retries: 最大重试次数
    """

    # 为提示词注入当前时间，便于模型判断绝对时间是过去还是未来
    import datetime as _dt

    current_time_str = _dt.datetime.now().isoformat(sep=" ", timespec="seconds")
    system_prompt_content = INTENT_PROMPT_TEMPLATE.replace("{current_time}", current_time_str)

    # 将 time_duration 添加到用户输入中（time_duration 现在总是有值，默认为 "long"）
    user_prompt_content = f"<user_input>{user_input}</user_input>\n<time_duration>{time_duration}</time_duration>"

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_prompt_content},
            {"role": "assistant", "content": "{"}
        ],
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object", "schema": QueryIntent.model_json_schema()}
    }

    url = f"{VLLM_API_BASE_URL}/chat/completions"
    last_exception = None
    headers={
           "Authorization": f"Bearer {API_KEY}",
           "Content-Type": "application/json"
       }
    for attempt in range(max_retries):
        logger.info(f"正在尝试进行意图分类 (第 {attempt + 1}/{max_retries} 次)...")
        try:
            # 使用传入的async_client和semaphore
            await asyncio.wait_for(semaphore.acquire(), timeout=VLLM_SLOT_WAIT_TIMEOUT_SECONDS)
            try:
                timeout = aiohttp.ClientTimeout(total=VLLM_REQUEST_TIMEOUT_SECONDS)
                async with async_client.post(url, json=payload, headers =headers,timeout=timeout) as response:
                    response.raise_for_status()
                    json_response = await response.json()
                    content = json_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not content or not content.strip():
                        raise ValueError("VLLM返回的 content 为空。")
                    structured_response = _parse_lenient_json(content)
                validated_model = QueryIntent.model_validate(structured_response)
                logger.info("用户查询意图分类成功！")
                return validated_model.model_dump()
            finally:
                semaphore.release()
                
        except (ValidationError, ValueError) as e:
            logger.warning(f"意图分类尝试 {attempt + 1} 失败，LLM输出不符合格式。错误: {e}")
            last_exception = e
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            logger.error(f"在第 {attempt + 1} 次意图分类尝试中发生网络错误: {e}")
            last_exception = e
        except Exception as e:
            logger.error(f"在第 {attempt + 1} 次意图分类尝试中发生意外错误: {e}", exc_info=True)
            last_exception = e

        if attempt < max_retries - 1:
            await asyncio.sleep(0.5)

    logger.error(f"在 {max_retries} 次尝试后，无法有效分类用户查询。最后一次错误: {last_exception}")
    raise Exception("AI多次尝试仍无法理解您的查询意图，请尝试换一种说法。")


# ======================== 奇门识别 & 标签缓存 ========================

_TAGS_TTL_SECONDS = 36 * 60 * 60  # 36 小时
_tags_cache: List[str] = []
_tags_cache_time: float = 0.0
_tags_cache_lock = asyncio.Lock()


async def get_all_tags_from_db_async(db_pool) -> List[str]:
    """
    查询数据库所有标签的类型（异步版本，使用连接池，带 36 小时缓存）
    数据库表：qimen_interpreted_analysis，字段：具体事项
    """
    global _tags_cache_time

    if not db_pool:
        logger.error("数据库连接池不可用")
        return []

    async with _tags_cache_lock:
        now = time()
        # 命中缓存
        if _tags_cache and now - _tags_cache_time < _TAGS_TTL_SECONDS:
            return list(_tags_cache)

        try:
            async with db_pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    query = """
                        SELECT DISTINCT `具体事项`
                        FROM qimen_interpreted_analysis
                        WHERE `具体事项` IS NOT NULL
                          AND `具体事项` != ''
                        ORDER BY `具体事项`
                    """
                    await cursor.execute(query)
                    results = await cursor.fetchall()
                    tags = [row[0] for row in results if row[0]]
                    logger.info(f"从数据库获取到 {len(tags)} 个奇门具体事项标签")
                    _tags_cache.clear()
                    _tags_cache.extend(tags)
                    _tags_cache_time = now
                    return list(_tags_cache)
        except Exception as e:
            logger.error(f"查询奇门标签失败: {e}", exc_info=True)
            # 查询失败时，尽量返回已有缓存（如果有）
            return list(_tags_cache) if _tags_cache else []


async def detect_time_range_with_llm(
    user_input: str,
    async_client: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> Dict[str, Any]:
    """
    使用单独的大模型请求识别用户问题中的时间范围。
    
    返回:
        {
            "has_time_range": bool,
            "end_date": str | None,
            "time_expression": str | None,
            "time_duration": "short" | "long" | None,
            "reason": str
        }
    """
    import datetime as _dt

    # 获取当前日期（格式：YYYY-MM-DD）
    current_date = _dt.date.today().isoformat()
    system_prompt_content = TIME_RANGE_PROMPT_TEMPLATE

    # 在用户输入中添加当前日期信息
    user_prompt_content = f"<user_input>{user_input}</user_input>\n<current_date>{current_date}</current_date>"

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_prompt_content},
            {"role": "assistant", "content": "{"},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object", "schema": TimeRangeDetection.model_json_schema()},
    }

    url = f"{VLLM_API_BASE_URL}/chat/completions"
    last_exception: Optional[Exception] = None
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        logger.info(f"正在尝试进行时间范围识别 (第 {attempt + 1}/{max_retries} 次)...")
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=VLLM_SLOT_WAIT_TIMEOUT_SECONDS)
            try:
                timeout = aiohttp.ClientTimeout(total=VLLM_REQUEST_TIMEOUT_SECONDS)
                async with async_client.post(url, json=payload, headers=headers, timeout=timeout) as response:
                    response.raise_for_status()
                    json_response = await response.json()
                    content = json_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not content or not content.strip():
                        raise ValueError("VLLM返回的时间范围识别 content 为空。")
                    structured_response = _parse_lenient_json(content)
                validated_model = TimeRangeDetection.model_validate(structured_response)
                logger.info("时间范围识别成功！")
                return validated_model.model_dump()
            finally:
                semaphore.release()

        except (ValidationError, ValueError) as e:
            logger.warning(f"时间范围识别尝试 {attempt + 1} 失败，LLM输出不符合格式。错误: {e}")
            last_exception = e
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            logger.error(f"在第 {attempt + 1} 次时间范围识别尝试中发生网络错误: {e}")
            last_exception = e
        except Exception as e:
            logger.error(f"在第 {attempt + 1} 次时间范围识别尝试中发生意外错误: {e}", exc_info=True)
            last_exception = e

        if attempt < max_retries - 1:
            await asyncio.sleep(0.5)

    logger.error(f"在 {max_retries} 次尝试后，无法有效完成时间范围识别。最后一次错误: {last_exception}")
    # 时间范围识别失败时，返回无时间范围的结果，不中断主流程
    return {"has_time_range": False, "start_time": None, "end_time": None, "reason": "时间范围识别失败，按无时间范围处理"}


def is_historical_time(time_range_result: Dict[str, Any]) -> bool:
    """
    判断识别到的时间范围是否为历史时间。
    
    规则：如果时间范围的结束日期小于当前日期，则认为是历史时间。
    
    参数:
        time_range_result: detect_time_range_with_llm 返回的结果
        
    返回:
        True: 是历史时间
        False: 不是历史时间（未来时间或无法判断）
    """
    import datetime as _dt
    
    if not time_range_result.get("has_time_range"):
        return False
    
    end_date_str = time_range_result.get("end_date")
    if not end_date_str:
        return False
    
    try:
        # 解析结束日期
        end_date = _dt.datetime.strptime(end_date_str, "%Y-%m-%d").date()
        current_date = _dt.date.today()
        
        # 如果结束日期小于当前日期，认为是历史时间
        is_historical = end_date < current_date
        logger.info(f"时间范围判断: 结束日期={end_date_str}, 当前日期={current_date.isoformat()}, 是否为历史时间={is_historical}")
        return is_historical
    except ValueError as e:
        logger.warning(f"无法解析时间范围结束日期 '{end_date_str}': {e}")
        return False


async def classify_qimen_with_llm(
    user_input: str,
    available_tags: List[str],
    async_client: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> Dict[str, Any]:
    """
    使用单独的大模型请求判断用户问题是否属于奇门三种类型，并提取具体事项标签。

    仅用于：判断 is_qimen / qimen_type / matched_tag。
    """
    if not available_tags:
        logger.warning("奇门识别：可用的具体事项标签列表为空，将直接返回非奇门。")
        return {"is_qimen": False, "qimen_type": None, "matched_tag": None, "reason": "无可用具体事项标签"}

    # 为避免 format 与 JSON 花括号冲突，这里只做占位符替换
    import datetime as _dt

    tags_text = json.dumps(available_tags, ensure_ascii=False)
    current_time_str = _dt.datetime.now().isoformat(sep=" ", timespec="seconds")
    system_prompt_content = (
        QIMEN_PROMPT_TEMPLATE.replace("{tags_text}", tags_text).replace("{current_time}", current_time_str)
    )

    user_prompt_content = f"""
    <user_input>{user_input}</user_input>

    <AVAILABLE_TAGS_JSON>
    {json.dumps(available_tags, ensure_ascii=False)}
    </AVAILABLE_TAGS_JSON>
    """

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_prompt_content},
            {"role": "assistant", "content": "{"},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object", "schema": QimenExtraction.model_json_schema()},
    }

    url = f"{VLLM_API_BASE_URL}/chat/completions"
    last_exception: Optional[Exception] = None
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        logger.info(f"正在尝试进行奇门类型识别 (第 {attempt + 1}/{max_retries} 次)...")
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=VLLM_SLOT_WAIT_TIMEOUT_SECONDS)
            try:
                timeout = aiohttp.ClientTimeout(total=VLLM_REQUEST_TIMEOUT_SECONDS)
                async with async_client.post(url, json=payload, headers=headers, timeout=timeout) as response:
                    response.raise_for_status()
                    json_response = await response.json()
                    content = json_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not content or not content.strip():
                        raise ValueError("VLLM返回的奇门识别 content 为空。")
                    structured_response = _parse_lenient_json(content)
                validated_model = QimenExtraction.model_validate(structured_response)
                logger.info("奇门类型识别成功！")
                return validated_model.model_dump()
            finally:
                semaphore.release()

        except (ValidationError, ValueError) as e:
            logger.warning(f"奇门类型识别尝试 {attempt + 1} 失败，LLM输出不符合格式。错误: {e}")
            last_exception = e
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            logger.error(f"在第 {attempt + 1} 次奇门类型识别尝试中发生网络错误: {e}")
            last_exception = e
        except Exception as e:
            logger.error(f"在第 {attempt + 1} 次奇门类型识别尝试中发生意外错误: {e}", exc_info=True)
            last_exception = e

        if attempt < max_retries - 1:
            await asyncio.sleep(0.5)

    logger.error(f"在 {max_retries} 次尝试后，无法有效完成奇门识别。最后一次错误: {last_exception}")
    # 奇门识别失败时，不要中断主流程，按“非奇门”处理
    return {"is_qimen": False, "qimen_type": None, "matched_tag": None, "reason": "奇门识别失败，按非奇门处理"}


# 注意：原有的aiohttp_vllm_invoke_text函数也已被移除


def _parse_lenient_json(json_string: str) -> dict:
    """
    (最终版 v4) 尽力从一个包含任何已知错误的字符串中解析并修复JSON对象。
    这是迄今为止最健壮的版本，专门为应对LLM的各种“创意”错误而设计。
    """
    if not json_string or not json_string.strip():
        logger.warning("传入的解析内容为空。")
        return {}

    # 1. 预处理
    match = re.search(r'```(?:json)?\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```', json_string, re.DOTALL)
    if match:
        content_to_parse = match.group(1)
    else:
        content_to_parse = json_string

    first_brace = content_to_parse.find('{')
    first_bracket = content_to_parse.find('[')

    if first_brace == -1 and first_bracket == -1:
        raise json.JSONDecodeError("在字符串中未找到JSON起始符号 '{' 或 '['", json_string, 0)

    start_pos = min(first_brace if first_brace != -1 else float('inf'),
                    first_bracket if first_bracket != -1 else float('inf'))

    current_string = content_to_parse[start_pos:]

    # 2. “诊断-修复-重试”循环
    max_repairs = 5
    for attempt in range(max_repairs + 1):
        try:
            return json.loads(current_string)
        except json.JSONDecodeError as e:
            logger.info(f"解析尝试 #{attempt + 1} 失败: {e.msg} at pos {e.pos}. 尝试修复...")

            if attempt >= max_repairs:
                logger.error(f"达到最大修复次数仍无法解析。最终失败的字符串: '{current_string}'")
                raise e

            # 3. 诊断并执行修复
            error_msg = e.msg.lower()
            original_string_for_this_attempt = current_string
            stripped = current_string.rstrip()

            # --- 修复策略工具箱 (按优先级排序) ---

            # 策略A: 修复未闭合的字符串 (最明确的错误)
            if "unterminated string" in error_msg:
                logger.info("诊断: 字符串未闭合。修复: 添加 '\"' 并尝试闭合括号。")
                current_string += '"'
                # 补上引号后，可能就是一个完整的JSON了，所以直接闭合
                current_string = _close_open_brackets(current_string)

            # 策略B: 修复悬空的逗号
            elif "expecting value" in error_msg and stripped.endswith(','):
                logger.info("诊断: 悬空的逗号。修复: 移除末尾 ',' 并尝试闭合括号。")
                current_string = stripped[:-1]
                current_string = _close_open_brackets(current_string)

            # 策略C (终极手段): 暴力回溯 - 丢弃最后一个（可能已损坏的）条目
            else:
                logger.info("诊断: 泛化的语法错误或严重截断。修复: 暴力回溯，丢弃最后一个不完整的条目。")

                # 我们寻找最后一个逗号，因为它标志着上一个完整键值对的结束。
                # 但我们需要确保这个逗号不在一个字符串内部。
                last_comma_pos = -1
                in_string = False
                for i in range(len(current_string) - 1, -1, -1):
                    char = current_string[i]
                    if char == '"' and (i == 0 or current_string[i - 1] != '\\'):
                        in_string = not in_string
                    if not in_string and char == ',':
                        last_comma_pos = i
                        break

                if last_comma_pos != -1:
                    # 截断到最后一个逗号，丢弃后面的所有内容。
                    current_string = current_string[:last_comma_pos]
                    # 强制闭合剩下的、理论上是有效的部分。
                    current_string = _close_open_brackets(current_string)
                else:
                    # 如果连一个合法的逗号都找不到，说明整个JSON结构已完全损坏。
                    # 此时无法安全修复，直接放弃。
                    logger.error("无法进行回溯修复（未找到任何不在字符串内的逗号），放弃。")
                    raise e

            # 安全检查：如果修复操作没有改变任何东西，为避免死循环，直接放弃
            if current_string == original_string_for_this_attempt:
                logger.error("修复操作未能改变字符串，为避免死循环而放弃。")
                raise e
    # 理论上不会执行到这里，但作为保障
    raise json.JSONDecodeError("无法从模型响应中解析出任何有效的JSON片段", json_string, 0)

def _close_open_brackets(text: str) -> str:
    """辅助函数，计算并添加所有未闭合的括号。"""
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')

    missing_braces = open_braces - close_braces
    missing_brackets = open_brackets - close_brackets

    if missing_braces <= 0 and missing_brackets <= 0:
        return text

    # 使用一个栈来精确地确定闭合顺序
    stack = []
    for char in text:
        if char == '{' or char == '[':
            stack.append(char)
        elif char == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif char == ']' and stack and stack[-1] == '[':
            stack.pop()

    closing_sequence = ''
    for bracket in reversed(stack):
        if bracket == '{':
            closing_sequence += '}'
        elif bracket == '[':
            closing_sequence += ']'

    return text + closing_sequence
