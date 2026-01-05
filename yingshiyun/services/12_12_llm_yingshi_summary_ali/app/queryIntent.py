import os
from pathlib import Path
from functools import lru_cache
from typing import Literal
import asyncio
import logging
import pandas as pd
import aiohttp
import json
import re
import sys
from time import time
from datetime import datetime

from fastapi import FastAPI, HTTPException, status, Request
from typing import Optional, List, Dict, Any, AsyncGenerator, Literal
from pydantic import BaseModel, Field, validator, ValidationError
from dotenv import load_dotenv

load_dotenv() # 在本地开发时加载 .env 文件

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - 说: %(message)s')
logger = logging.getLogger(__name__)


VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL", "http://10.210.254.69:6002/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen3-30B-A3B-Instruct-2507")
API_KEY = os.getenv("API_KEY","sk-81d4dbe056f94030998f0639f709bff4")

VLLM_SLOT_WAIT_TIMEOUT_SECONDS = 1500
VLLM_REQUEST_TIMEOUT_SECONDS = 500

# 奇门“具体事项”标签缓存（36 小时）
_TAGS_TTL_SECONDS = 36 * 3600
_tags_cache: list[str] = []
_tags_cache_time: float = 0.0
_tags_cache_lock = asyncio.Lock()


@lru_cache(maxsize=1)
def load_intent_prompt() -> str:
    """读取意图分类的 system prompt，失败时抛出 500。"""
    prompt_path = Path(__file__).resolve().parent / "prompts" / "intent_classification_prompt.xml"
    try:
        return prompt_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("加载意图分类提示词失败: %s", exc)
        raise HTTPException(status_code=500, detail="系统提示词加载失败，请联系管理员。")


@lru_cache(maxsize=1)
def load_qimen_extraction_prompt() -> str:
    """读取奇门提取的 system prompt，失败时抛出 500。"""
    prompt_path = Path(__file__).resolve().parent / "prompts" / "qimen_extraction_prompt.xml"
    try:
        return prompt_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("加载奇门提取提示词失败: %s", exc)
        raise HTTPException(status_code=500, detail="奇门提取提示词加载失败，请联系管理员。")


@lru_cache(maxsize=1)
def load_time_range_extraction_prompt() -> str:
    """读取时间范围提取的 system prompt，失败时抛出 500。"""
    prompt_path = Path(__file__).resolve().parent / "prompts" / "time_range_extraction_prompt.xml"
    try:
        return prompt_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("加载时间范围提取提示词失败: %s", exc)
        raise HTTPException(status_code=500, detail="时间范围提取提示词加载失败，请联系管理员。")


class QueryIntent(BaseModel):
    """
    用于验证 LLM 对用户问题意图分类结果的 Pydantic 模型。
    新增 'illegal_content' 用于识别违法犯罪相关内容。
    新增 'general_knowledge' 用于识别常识性知识或具体知识型问题。
    新增 'qimen' 用于识别奇门遁甲类择时/择事问题。
    新增 'self_intro' 用于识别用户对助手功能/身份的询问。
    注意：historical_event 已从意图分类中移除，改为单独的时间范围识别。
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
        description=(
            "问题的分类结果。'specific_short_term' 表示具体短期事件咨询；"
            "'general_long_term' 表示宏观长期解读；"
            "'knowledge_question' 表示命理专业知识问题；"
            "'illegal_content' 表示涉及违法犯罪的内容；'general_knowledge' 表示常识性知识或具体知识型问题（如数学、烹饪、天气等）；"
            "'qimen' 表示奇门遁甲类择时/择事问题；"
            "'self_intro' 表示询问助手自我介绍/功能说明；"
            "'non_ziwei_system' 表示要求使用非紫微体系（如八字/风水/星座等）进行解读的问题。"
        )
    )
    reason: str = Field(..., description="AI 做出分类判断的简要理由。但在输出的理由中不要提及：【因此归类为specific_short_term。】类似这种带有字段名称的话语，也就是不希望出现字段名称。")


class TimeRangeExtraction(BaseModel):
    """
    用于验证 LLM 对用户问题中时间范围提取结果的 Pydantic 模型。
    """
    has_time_range: bool = Field(
        ...,
        description="问题中是否包含明确的时间范围"
    )
    end_date: Optional[str] = Field(
        None,
        description="时间范围的结束日期，格式为YYYY-MM-DD。如果has_time_range为false，则为null"
    )
    time_expression: Optional[str] = Field(
        None,
        description="用户问题中的原始时间表达（如果有）"
    )
    time_span_type: Literal["short_term", "long_term"] = Field(
        ...,
        description="时间跨度类型。如果has_time_range为true且有明确的开始和结束日期，则根据时间跨度判断：一个月内（≤30天）为short_term，超过一个月（>30天）为long_term。如果has_time_range为false（用户没有提及明确的时间），则默认为long_term"
    )
    reason: str = Field(..., description="简要说明提取的时间范围和判断依据")


class QimenExtraction(BaseModel):
    """
    用于验证 LLM 对用户问题进行奇门遁甲类型判断和具体事项提取的 Pydantic 模型。
    与意图分类并行调用，专门判断是否为奇门类型及提取具体事项标签。
    """
    is_qimen: bool = Field(
        ...,
        description="是否属于奇门遁甲类型"
    )
    qimen_type: Optional[int] = Field(
        None,
        description="当 is_qimen 为 true 时必须填写，取值 1、2 或 3，分别对应奇门的三种类型。非奇门时为 null。"
    )
    matched_tag: Optional[str] = Field(
        None,
        description="当 is_qimen 为 true 且 qimen_type 为 1 或 2 时，必须填写用户问题中匹配到的具体事项标签（从系统注入的标签列表中选择最匹配的一个）。qimen_type 为 3 或非奇门时为 null。"
    )
    reason: str = Field(..., description="简短说明判断依据")


# --- aiohttp 调用 VLLM 的函数 (保持不变) ---
async def aiohttp_vllm_invoke(app: FastAPI, payload: dict, max_retries: int = 5) -> Dict[str, Any]:
    
    session = getattr(app.state, 'http_session', None)
    semaphore = getattr(app.state, 'vllm_semaphore', None)

    # (此函数逻辑保持不变，它是一个通用的、健壮的VLLM调用器)
    if not session or not semaphore:
        raise HTTPException(503, "AIOHTTP客户端未初始化。")

    url = f"{VLLM_API_BASE_URL}/chat/completions"
    last_exception = None
    headers={
           "Authorization": f"Bearer {API_KEY}",
           "Content-Type": "application/json"
       }
    for attempt in range(max_retries):
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=VLLM_SLOT_WAIT_TIMEOUT_SECONDS)
            try:
                timeout = aiohttp.ClientTimeout(total=VLLM_REQUEST_TIMEOUT_SECONDS)
                async with session.post(url, json=payload, headers =headers ,timeout=timeout) as response:
                    response.raise_for_status()
                    json_response = await response.json()
                    content = json_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not content or not content.strip():
                        raise ValueError("VLLM返回的 content 为空。")
                    return _parse_lenient_json(content)
            finally:
                semaphore.release() # 释放信号量
        except (asyncio.TimeoutError, aiohttp.ClientError, json.JSONDecodeError, ValueError) as e:
            last_exception = e
            if attempt < max_retries - 1: await asyncio.sleep(0.5)
        except Exception as e:
            raise HTTPException(500, f"VLLM调用时发生未知且严重的错误: {e}")

    logger.error(f"在 {max_retries} 次尝试后，仍无法从VLLM获取有效响应。最后一次错误: {last_exception}")
    raise HTTPException(500, "调用AI服务多次失败，请联系管理员。")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ 核心修改: 更新 Pydantic 模型和 LLM 分类函数 +++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

async def answer_knowledge_question(app: FastAPI, user_question: str, max_retries: int = 3) -> str:
    """
    【问题7和10】直接在网关调用VLLM回答专业知识问题
    要求回答通俗简洁，50-100字左右
    """
    system_prompt = """<|im_start|>system
你是一位精通紫微斗数和命理学的专家。当用户询问专业术语或概念时，请用通俗易懂的语言简洁解释。

**要求：**
1. 回答要通俗易懂，避免过于晦涩的术语
2. 控制在50-100字左右
3. 直接给出解释，不要加"您好"等客套话
4. 如果涉及多个方面，只讲最核心的含义

**示例：**
用户：什么是破军化权？
回答：破军化权是紫微斗数中的一种星曜组合。破军星代表突破创新，化权则增强其力量和决断力。这个组合通常表示一个人做事果断、敢于突破常规，适合在变革性强的领域发展，但也要注意冲动行事。

用户：七杀坐守命宫是什么意思？
回答：七杀星落在命宫，代表性格独立坚强、做事果断，有领导才能和魄力。这类人通常不喜欢被束缚，适合独立创业或从事需要决断力的工作。但要注意控制急躁脾气，避免过于刚硬。
<|im_end|>"""

    user_prompt = f"<|im_start|>user\n{user_question}<|im_end|>\n<|im_start|>assistant\n"

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        "temperature": 0.5,
        "max_tokens": 300,
        "stream": False
    }

    try:
        logger.info(f"[专业知识问答] 调用VLLM回答问题: {user_question}")
        answer = await aiohttp_vllm_invoke_text(app, payload, max_retries=max_retries)
        logger.info(f"[专业知识问答] 成功获取回答，长度: {len(answer)}")
        return answer
    except Exception as e:
        logger.error(f"[专业知识问答] 调用失败: {e}", exc_info=True)
        raise


async def get_all_tags_from_db_async(db_pool) -> List[str]:
    """
    查询数据库所有"具体事项"标签（异步，带 36 小时缓存）。
    数据库表：qimen_interpreted_analysis，字段：具体事项
    """
    global _tags_cache_time
    if not db_pool:
        logger.warning(f"[标签查询] db_pool 不可用，使用已有缓存（缓存中有 {len(_tags_cache)} 个标签）")
        return _tags_cache

    async with _tags_cache_lock:
        now = time()
        cache_age = now - _tags_cache_time if _tags_cache_time > 0 else float('inf')
        if _tags_cache and cache_age < _TAGS_TTL_SECONDS:
            logger.info(f"[标签查询] 使用缓存，缓存中有 {len(_tags_cache)} 个标签，缓存年龄: {cache_age/3600:.2f} 小时")
            return _tags_cache
        
        logger.info(f"[标签查询] 缓存过期或为空，从数据库重新查询...")
        try:
            async with db_pool.acquire() as conn:
                logger.info(f"[标签查询] 数据库连接获取成功")
                async with conn.cursor() as cursor:
                    query = """
                        SELECT DISTINCT `具体事项`
                        FROM qimen_interpreted_analysis
                        WHERE `具体事项` IS NOT NULL
                        AND `具体事项` != ''
                        ORDER BY `具体事项`
                    """
                    logger.info(f"[标签查询] 执行SQL查询...")
                    await cursor.execute(query)
                    results = await cursor.fetchall()
                    tags = [row[0] for row in results if row[0]]
                    logger.info(f"[标签查询] 从数据库获取到 {len(tags)} 个标签")
                    if tags:
                        logger.info(f"[标签查询] 标签示例（前10个）: {tags[:10]}")
                    else:
                        logger.warning(f"[标签查询] ⚠️ 数据库查询返回空结果！表 qimen_interpreted_analysis 可能没有数据")
                    _tags_cache.clear()
                    _tags_cache.extend(tags)
                    _tags_cache_time = now
                    return tags
        except Exception as e:
            logger.error(f"[标签查询] 查询标签失败: {e}", exc_info=True)
            logger.warning(f"[标签查询] 返回已有缓存（{len(_tags_cache)} 个标签）或空列表")
            return _tags_cache or []


async def classify_query_intent_with_llm(app: FastAPI, user_input: str, max_retries: int = 5, db_pool=None, time_span_type: Optional[str] = None) -> Dict[str, Any]:
    """
    使用 LLM 对用户问题的意图进行分类。
    注意：奇门类型判断已拆分为独立请求 extract_qimen_with_llm，此函数不再注入具体事项标签。
    注意：历史事件判断已拆分为独立请求 extract_time_range_with_llm，此函数不再处理历史事件。
    
    参数:
        time_span_type: 时间跨度类型，来自时间范围提取的结果。可选值："short_term"（一个月内）、"long_term"（超过一个月）、None（无法判断或无时间范围）
    """

    system_prompt_content = load_intent_prompt()
    
    # 将时间跨度类型注入到 system prompt 中（time_span_type 总是会被提供，因为时间范围提取总是返回 short_term 或 long_term）
    if time_span_type:
        time_span_instruction = (
            f"\n\n【时间跨度类型信息（来自时间范围提取模块）】\n"
            f"时间跨度类型：{time_span_type}\n"
            f"- short_term：表示时间范围在一个月内（≤30天），基于实际天数差计算\n"
            f"- long_term：表示时间范围超过一个月（>30天）或用户没有提及明确时间，基于实际天数差计算\n"
            f"**重要**：请严格按照此时间跨度类型与具体事件结合进行分类判断。不要自己分析问题中的时间表达来判断长短时间，时间范围提取模块已经完成了判断。\n"
        )
        system_prompt_content = system_prompt_content + time_span_instruction
    else:
        # 如果未提供 time_span_type（不应该发生），添加默认说明
        time_span_instruction = (
            f"\n\n【时间跨度类型信息】\n"
            f"未提供时间跨度类型，默认为 long_term。请按照 long_term 的规则进行分类。\n"
        )
        system_prompt_content = system_prompt_content + time_span_instruction

    user_prompt_content = f"<user_input>{user_input}</user_input>"

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

    last_exception = None
    for attempt in range(max_retries):
        logger.info(f"正在尝试进行意图分类 (第 {attempt + 1}/{max_retries} 次)...")
        try:
            structured_response = await aiohttp_vllm_invoke(app, payload)
            validated_model = QueryIntent.model_validate(structured_response)
            logger.info("用户查询意图分类成功！")
            return validated_model.model_dump()
        except (ValidationError, ValueError) as e:
            logger.warning(f"意图分类尝试 {attempt + 1} 失败，LLM输出不符合格式。错误: {e}")
            last_exception = e
        except Exception as e:
            logger.error(f"在第 {attempt + 1} 次意图分类尝试中发生意外错误: {e}", exc_info=True)
            last_exception = e

        if attempt < max_retries - 1:
            await asyncio.sleep(0.5)

    logger.error(f"在 {max_retries} 次尝试后，无法有效分类用户查询。最后一次错误: {last_exception}")
    raise HTTPException(status_code=500, detail="AI多次尝试仍无法理解您的查询意图，请尝试换一种说法。")


async def extract_time_range_with_llm(app: FastAPI, user_input: str, max_retries: int = 5) -> Dict[str, Any]:
    """
    使用 LLM 提取用户问题中的时间范围。
    与 classify_query_intent_with_llm 并行调用，专门用于判断是否为历史事件。
    
    返回:
        {
            "has_time_range": bool,
            "end_date": str | None,  # YYYY-MM-DD格式
            "time_expression": str | None,
            "reason": str
        }
    """
    system_prompt_content = load_time_range_extraction_prompt()

    # 注入当前日期信息
    now = datetime.now()
    current_date_str = now.strftime("%Y-%m-%d")
    current_date_zh = now.strftime("%Y年%m月%d日")
    time_instruction = (
        f"\n\n【当前日期信息】\n"
        f"当前公历日期：{current_date_zh}（{current_date_str}）\n"
        f"请根据此日期计算相对时间的绝对日期。\n"
    )
    system_prompt_content = system_prompt_content + time_instruction

    # 在user_prompt中明确提供当前日期，使用<current_date>标签
    user_prompt_content = f"<user_input>{user_input}</user_input>\n<current_date>{current_date_str}</current_date>"

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_prompt_content},
            {"role": "assistant", "content": "{"}
        ],
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object", "schema": TimeRangeExtraction.model_json_schema()}
    }

    last_exception = None
    for attempt in range(max_retries):
        logger.info(f"正在尝试进行时间范围提取 (第 {attempt + 1}/{max_retries} 次)...")
        try:
            structured_response = await aiohttp_vllm_invoke(app, payload)
            validated_model = TimeRangeExtraction.model_validate(structured_response)
            logger.info(f"时间范围提取成功！has_time_range={validated_model.has_time_range}, end_date={validated_model.end_date}")
            return validated_model.model_dump()
        except (ValidationError, ValueError) as e:
            logger.warning(f"时间范围提取尝试 {attempt + 1} 失败，LLM输出不符合格式。错误: {e}")
            last_exception = e
        except Exception as e:
            logger.error(f"在第 {attempt + 1} 次时间范围提取尝试中发生意外错误: {e}", exc_info=True)
            last_exception = e

        if attempt < max_retries - 1:
            await asyncio.sleep(0.5)

    logger.error(f"在 {max_retries} 次尝试后，无法有效提取时间范围。最后一次错误: {last_exception}")
    # 如果时间范围提取失败，默认返回非历史事件（保守策略）
    # 由于用户没有提及明确的时间，time_span_type 默认为 long_term
    return {
        "has_time_range": False,
        "end_date": None,
        "time_expression": None,
        "time_span_type": "long_term",
        "reason": "时间范围提取失败，默认视为非历史事件。由于无法确定时间范围，time_span_type 默认为 long_term"
    }


async def extract_qimen_with_llm_single(app: FastAPI, user_input: str, system_prompt_content: str, attempt_num: int) -> Dict[str, Any]:
    """
    单次奇门提取请求（内部函数，用于并行调用）
    """
    user_prompt_content = f"""
    <user_input>{user_input}</user_input>
    """

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_prompt_content},
            {"role": "assistant", "content": "{"}
        ],
        "temperature": 0.0,
        "max_tokens": 150,
        "response_format": {"type": "json_object", "schema": QimenExtraction.model_json_schema()}
    }

    try:
        structured_response = await aiohttp_vllm_invoke(app, payload)
        
        # 【容错处理】如果 qimen_type 是字符串格式，转换为整数
        if "qimen_type" in structured_response:
            qimen_type = structured_response["qimen_type"]
            if isinstance(qimen_type, str):
                # 处理 "type1", "type2", "type3" 或 "1", "2", "3"
                if qimen_type.lower().startswith("type"):
                    qimen_type = qimen_type.lower().replace("type", "")
                try:
                    structured_response["qimen_type"] = int(qimen_type) if qimen_type else None
                    logger.info(f"[奇门提取-{attempt_num}] 自动转换 qimen_type: '{qimen_type}' -> {structured_response['qimen_type']}")
                except (ValueError, TypeError):
                    logger.warning(f"[奇门提取-{attempt_num}] 无法转换 qimen_type: {qimen_type}，设为 None")
                    structured_response["qimen_type"] = None
        
        validated_model = QimenExtraction.model_validate(structured_response)
        logger.info(f"[奇门提取-{attempt_num}] 成功！is_qimen={validated_model.is_qimen}, type={validated_model.qimen_type}, tag={validated_model.matched_tag}")
        return validated_model.model_dump()
    except (ValidationError, ValueError) as e:
        logger.warning(f"[奇门提取-{attempt_num}] LLM输出不符合格式: {e}")
        return {"is_qimen": False, "qimen_type": None, "matched_tag": None, "reason": f"第{attempt_num}次请求格式错误"}
    except Exception as e:
        logger.error(f"[奇门提取-{attempt_num}] 请求失败: {e}", exc_info=True)
        return {"is_qimen": False, "qimen_type": None, "matched_tag": None, "reason": f"第{attempt_num}次请求异常"}


async def extract_qimen_with_llm(app: FastAPI, user_input: str, parallel_requests: int = 5, db_pool=None) -> List[Dict[str, Any]]:
    """
    使用 LLM 判断用户问题是否属于奇门遁甲类型，并提取具体事项标签。
    并行请求多次（默认5次），返回所有结果列表。
    与 classify_query_intent_with_llm 并行调用。
    
    返回:
        List[{
            "is_qimen": bool,
            "qimen_type": int | None,  # 1, 2, 3 或 None
            "matched_tag": str | None,  # 类型1/2时的具体事项标签
            "reason": str
        }]
    """
    system_prompt_content = load_qimen_extraction_prompt()

    # 动态注入奇门"具体事项"标签，用于类型1/2的具体事项匹配
    logger.info(f"[奇门提取] 开始获取具体事项标签，db_pool={db_pool is not None}")
    try:
        tags = await get_all_tags_from_db_async(db_pool)
        logger.info(f"[奇门提取] 获取到 {len(tags)} 个标签")
        if tags:
            logger.info(f"[奇门提取] 前10个标签示例: {tags[:10]}")
    except Exception as e:
        logger.error(f"[奇门提取] 获取具体事项标签失败: {e}", exc_info=True)
        tags = []
    
    if tags:
        max_tags = 500
        shown = tags[:max_tags]
        more = f"（其余 {len(tags)-max_tags} 条省略）" if len(tags) > max_tags else ""
        tags_text = "、".join(shown)
        logger.info(f"[奇门提取] 注入标签到prompt，共 {len(tags)} 个，显示前 {min(max_tags, len(tags))} 个")
    else:
        tags_text = "（暂无标签）"
        more = ""
        logger.warning(f"[奇门提取] 标签列表为空！db_pool={db_pool is not None}")

    system_prompt_content += (
        "\n\n【具体事项标签列表（仅用于类型1和类型2的具体事项提取）】\n"
        f"{tags_text}{more}\n"
        "规则：对于类型1和类型2，必须从上述标签中选择最匹配的一个填入 matched_tag；"
        "如果用户问题表述与某个标签含义相近，选择最接近的标签。"
    )
    
    # 记录实际注入到prompt的标签部分（用于调试）
    logger.debug(f"[奇门提取] 注入的标签部分: {tags_text[:200]}...")

    # 并行发起多次请求
    logger.info(f"[奇门提取] 并行发起 {parallel_requests} 次请求...")
    tasks = [
        extract_qimen_with_llm_single(app, user_input, system_prompt_content, i+1)
        for i in range(parallel_requests)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理结果，将异常转换为默认结果
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"[奇门提取-{i+1}] 请求异常: {result}")
            processed_results.append({
                "is_qimen": False,
                "qimen_type": None,
                "matched_tag": None,
                "reason": f"第{i+1}次请求异常"
            })
        else:
            processed_results.append(result)
    
    logger.info(f"[奇门提取] 完成，共获得 {len(processed_results)} 个结果")
    return processed_results


async def aiohttp_vllm_invoke_text(app: FastAPI, payload: dict, max_retries: int = 3) -> str:
    """
    调用VLLM获取纯文本响应（不解析JSON）
    专门用于专业知识问答
    """
    session = getattr(app.state, 'http_session', None)
    semaphore = getattr(app.state, 'vllm_semaphore', None)

    if not session or not semaphore:
        raise HTTPException(503, "AIOHTTP客户端未初始化。")

    url = f"{VLLM_API_BASE_URL}/chat/completions"
    last_exception = None
    headers={
           "Authorization": f"Bearer {API_KEY}",
           "Content-Type": "application/json"
       }
    for attempt in range(max_retries):
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=VLLM_SLOT_WAIT_TIMEOUT_SECONDS)
            try:
                timeout = aiohttp.ClientTimeout(total=VLLM_REQUEST_TIMEOUT_SECONDS)
                async with session.post(url, json=payload,headers =headers ,timeout=timeout) as response:
                    response.raise_for_status()
                    json_response = await response.json()
                    content = json_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not content or not content.strip():
                        raise ValueError("VLLM返回的 content 为空。")
                    return content.strip()  # 直接返回文本，不解析JSON
            finally:
                semaphore.release() # 释放信号量
        except (asyncio.TimeoutError, aiohttp.ClientError, ValueError) as e:
            last_exception = e
            logger.warning(f"[VLLM文本调用] 第 {attempt + 1} 次尝试失败: {e}")
            if attempt < max_retries - 1: 
                await asyncio.sleep(0.5)
        except Exception as e:
            semaphore.release()
            raise HTTPException(500, f"VLLM调用时发生未知错误: {e}")

    logger.error(f"在 {max_retries} 次尝试后，仍无法从VLLM获取有效响应。最后一次错误: {last_exception}")
    raise last_exception or Exception("调用VLLM失败")


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
