import os
from typing import Literal
import asyncio
import os
import logging
import pandas as pd
import aiohttp
import json
import re
from datetime import datetime

from fastapi import FastAPI, HTTPException, status, Request
from typing import Optional, List, Dict, Any, AsyncGenerator, Literal
from pydantic import BaseModel, Field, validator, ValidationError

from dotenv import load_dotenv

# --- 配置加载 ---
load_dotenv()

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - 说: %(message)s')
logger = logging.getLogger(__name__)


VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL", "http://192.168.1.101:6002/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen3-30B-A3B-Instruct-2507")

VLLM_SLOT_WAIT_TIMEOUT_SECONDS = 1500
VLLM_REQUEST_TIMEOUT_SECONDS = 500

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENT_PROMPT_PATH = os.path.join(BASE_DIR, "..", "prompts", "intent_classification_prompt.xml")
TIME_RANGE_PROMPT_PATH = os.path.join(BASE_DIR, "..", "prompts", "time_range_detection_prompt.xml")
_INTENT_PROMPT_CACHE: Optional[str] = None
_TIME_RANGE_PROMPT_CACHE: Optional[str] = None

QIMEN_REFUSAL_MESSAGE = """您的问题涉及具体择时择事，建议您使用奇门遁甲/智慧卡服务获得更专业的解答。

奇门服务更擅长：
- 具体时间点的择时分析
- 通过时间确定适合的事件
- 通过事件确定适合的时间

智慧卡服务特别适合：
- 具体事件的短期预测
- 明确的行动建议和指引
- 快速决策支持
- 选择类问题的对比分析

紫微斗数特别适合：
- 专业术语和概念的详细解释
- 命盘结构的深入理解
- 长期运势和人生格局的分析
"""


def remove_birth_info_from_query(query: str) -> str:
    """
    从用户查询中去除出生年月日信息，只保留问题部分。
    
    支持的格式：公历 YYYY-MM-DD HH:MM:SS 性别
    例如：公历 1995-11-05 12:00:00 男
    
    Args:
        query: 用户的完整查询字符串
        
    Returns:
        str: 去除出生信息后的问题部分
    """
    if not query or not query.strip():
        return query
    
    # 匹配固定格式：公历 YYYY-MM-DD HH:MM:SS 性别
    # 例如：公历 1995-11-05 12:00:00 男 或 公历 1995-11-05 12:00:00 男，问题
    pattern = r'^公历\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[男女][\s，,]*'
    
    # 去除匹配的出生信息
    cleaned_query = re.sub(pattern, '', query.strip(), count=1).strip()
    
    # 如果去除后为空，说明整个输入都是出生信息，返回原始查询
    return cleaned_query if cleaned_query else query


class QueryIntent(BaseModel):
    """
    用于验证 LLM 对用户问题意图分类结果的 Pydantic 模型。
    新增 'illegal_content' 用于识别违法犯罪相关内容。
    新增 'general_knowledge' 用于识别常识性知识或具体知识型问题。
    新增 'qimen' 用于识别奇门遁甲相关问题。
    新增 'self_intro' 用于识别助手自我介绍/功能说明类问题。
    新增 'non_ziwei_system' 用于识别非紫微体系（八字、风水、奇门等）的专业问答/解读需求。
    注意：已移除 'historical_event'，历史事件判断由独立的时间范围识别模型处理。
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
        description="问题的分类结果。'specific_short_term' 表示具体短期事件咨询；'general_long_term' 表示宏观长期解读；'knowledge_question' 表示命理专业知识问题；'illegal_content' 表示涉及违法犯罪的内容；'general_knowledge' 表示常识性知识或具体知识型问题（如数学、烹饪、天气等）；'qimen' 表示与具体择时择事相关的问题；'self_intro' 表示助手自我介绍/功能说明类问题；'non_ziwei_system' 表示非紫微体系（如八字、风水、奇门遁甲等）的专业问答/解读需求。"
    )
    reason: str = Field(..., description="AI 做出分类判断的简要理由。但在输出的理由中不要提及：【因此归类为specific_short_term。】类似这种带有字段名称的话语，也就是不希望出现字段名称。")
    relative_time_indicator: Optional[str] = Field(None, description="提取用户查询中的时间指示器，如：'今后十年'、'未来五年'、'后年'、'明年'、'下个月'、'26 27 28年'、'2026年2027年'等。如果没有明确的时间指示器，返回None或空字符串。")


class TimeRangeDetection(BaseModel):
    """
    用于验证 LLM 对用户问题中时间范围识别结果的 Pydantic 模型。
    """
    has_time_range: bool = Field(..., description="是否识别到时间范围")
    end_date: Optional[str] = Field(None, description="结束日期，格式：YYYY-MM-DD")
    time_expression: Optional[str] = Field(None, description="用户问题中的原始时间表达（如果有）")
    reason: str = Field(..., description="简要说明提取的时间范围和判断依据")
    time_span_type: Literal["short_term", "long_term"] = Field(
        ...,
        description="时间跨度类型。判断规则：1) 如果识别到时间范围（has_time_range=true）：计算从当前日期到结束日期的天数，一个月内（≤30天）为'short_term'，超过一个月（>30天）为'long_term'。2) 如果没有识别到时间范围（has_time_range=false），返回'long_term'（默认视为长时间）。注意：只要不是短时间就是长时间，没有提及时间也代表是长时间。"
    )


# 注意：原有的aiohttp_vllm_invoke函数已被移除，
# 现在直接使用clients.vllm_client模块中的对应函数

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ 核心修改: 更新 Pydantic 模型和 LLM 分类函数 +++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

async def answer_knowledge_question(user_question: str, max_retries: int = 3) -> str:
    """
    【问题7和10】直接在紫薇/智慧卡服务调用VLLM回答专业知识问题
    要求回答通俗简洁，50-100字左右
    注意：此版本使用全局的clients模块，不需要传入app参数
    """
    from clients import vllm_client
    
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
        # 使用紫薇服务的vllm_client模块
        answer = await vllm_client.aiohttp_vllm_invoke_text(payload, max_retries=max_retries)
        logger.info(f"[专业知识问答] 成功获取回答，长度: {len(answer)}")
        return answer
    except Exception as e:
        logger.error(f"[专业知识问答] 调用失败: {e}", exc_info=True)
        raise


def _load_intent_classification_prompt() -> str:
    """
    读取意图分类的 system prompt，并使用简单缓存避免重复 IO。
    """
    global _INTENT_PROMPT_CACHE
    if _INTENT_PROMPT_CACHE is None:
        try:
            with open(INTENT_PROMPT_PATH, "r", encoding="utf-8") as f:
                _INTENT_PROMPT_CACHE = f.read()
        except Exception as exc:
            logger.error(f"加载意图分类提示词失败: {exc}")
            raise
    return _INTENT_PROMPT_CACHE


def _load_time_range_detection_prompt() -> str:
    """
    读取时间范围识别的 system prompt，并使用简单缓存避免重复 IO。
    """
    global _TIME_RANGE_PROMPT_CACHE
    if _TIME_RANGE_PROMPT_CACHE is None:
        try:
            with open(TIME_RANGE_PROMPT_PATH, "r", encoding="utf-8") as f:
                _TIME_RANGE_PROMPT_CACHE = f.read()
        except Exception as exc:
            logger.error(f"加载时间范围识别提示词失败: {exc}")
            raise
    return _TIME_RANGE_PROMPT_CACHE


async def classify_query_intent_with_llm(
    user_input: str, 
    time_span_type: Optional[str] = None,
    max_retries: int = 5
) -> Dict[str, Any]:
    """
    使用 LLM 对用户问题的意图进行分类（适配紫薇/智慧卡服务）。
    - specific_short_term: 针对具体、短期事件。
    - general_long_term: 针对宏观、长期问题 (包括择色、择方位、择吉数)。
    - self_intro: 助手自我介绍/功能说明类问题。
    注意：此版本使用全局的clients模块，不需要传入app参数
    
    Args:
        user_input: 用户输入
        time_span_type: 时间跨度类型（"short_term"或"long_term"），由时间范围识别模型提供
        max_retries: 最大重试次数
    """
    from clients import vllm_client

    # 原始意图分类 system prompt
    # 注意：历史事件判断已由独立的时间范围识别模型处理，此处不再需要时间判断逻辑
    system_prompt_content = _load_intent_classification_prompt()
    
    # 如果提供了 time_span_type，转换为 time_duration 并在 user_prompt 中添加标签
    time_duration_value = None
    if time_span_type:
        # 将 time_span_type 的值转换为 time_duration 的值
        if time_span_type == "short_term":
            time_duration_value = "short"
        elif time_span_type == "long_term":
            time_duration_value = "long"
        
        # 在 system_prompt 中注入说明信息
        time_span_info = f"\n\n**【重要：时间跨度信息】**\n系统已识别出用户问题的时间跨度为：{time_span_type}（{'短时间：一个月内' if time_span_type == 'short_term' else '长时间：超过一个月'}）。\n你必须严格按照这个时间跨度类型，结合具体事件进行意图分类：\n- 如果 time_span_type='short_term' 且涉及具体事件 → 优先考虑 `specific_short_term`\n- 如果 time_span_type='long_term' 且涉及宏观趋势 → 优先考虑 `general_long_term`\n- 如果 time_span_type='short_term' 但涉及宏观运势趋势 → 仍可考虑 `general_long_term`（如果包含紫微相关关键词）\n- 如果 time_span_type='long_term' 但涉及具体事件决策 → 仍可考虑 `specific_short_term`（如果包含智慧卡等关键词）\n"
        system_prompt_content = system_prompt_content + time_span_info

    # 构建 user_prompt，如果提供了 time_duration 则添加标签
    if time_duration_value:
        user_prompt_content = f"<user_input>{user_input}</user_input>\n<time_duration>{time_duration_value}</time_duration>"
    else:
        user_prompt_content = f"<user_input>{user_input}</user_input>"

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_prompt_content},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object", "schema": QueryIntent.model_json_schema()}
    }

    last_exception = None
    for attempt in range(max_retries):
        logger.info(f"正在尝试进行意图分类 (第 {attempt + 1}/{max_retries} 次)...")
        try:
            # 使用紫薇服务的vllm_client模块
            # aiohttp_vllm_invoke 返回 (dict, input_tokens, output_tokens)
            structured_response, input_tokens, output_tokens = await vllm_client.aiohttp_vllm_invoke(payload)
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
    raise Exception("AI多次尝试仍无法理解您的查询意图，请尝试换一种说法。")


async def detect_time_range_with_llm(
    user_input: str, 
    current_time: Optional[datetime] = None,
    max_retries: int = 5
) -> Dict[str, Any]:
    """
    使用 LLM 识别用户问题中的时间范围，并判断是否为历史时间。
    注意：此版本使用全局的clients模块，不需要传入app参数
    
    Args:
        user_input: 用户输入
        current_time: 当前时间，如果不提供则使用datetime.now()
        max_retries: 最大重试次数
    """
    from clients import vllm_client

    # 加载时间范围识别 system prompt
    system_prompt_content = _load_time_range_detection_prompt()

    # 注入当前日期
    now = current_time if current_time is not None else datetime.now()
    current_date_str = now.strftime("%Y-%m-%d")
    
    # 记录使用的时间（用于调试）
    if current_time is not None:
        logger.info(f"[时间范围识别] 使用传入的当前时间: {current_date_str}")
    else:
        logger.info(f"[时间范围识别] 使用系统当前时间: {current_date_str}")

    # 在 user_prompt 中添加当前日期标签（根据提示词要求）
    user_prompt_content = f"<user_input>{user_input}</user_input>\n<current_date>{current_date_str}</current_date>"

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_prompt_content},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object", "schema": TimeRangeDetection.model_json_schema()}
    }

    last_exception = None
    for attempt in range(max_retries):
        logger.info(f"正在尝试进行时间范围识别 (第 {attempt + 1}/{max_retries} 次)...")
        try:
            # 使用紫薇服务的vllm_client模块
            # aiohttp_vllm_invoke 返回 (dict, input_tokens, output_tokens)
            structured_response, input_tokens, output_tokens = await vllm_client.aiohttp_vllm_invoke(payload)
            validated_model = TimeRangeDetection.model_validate(structured_response)
            logger.info("用户查询时间范围识别成功！")
            return validated_model.model_dump()
        except (ValidationError, ValueError) as e:
            logger.warning(f"时间范围识别尝试 {attempt + 1} 失败，LLM输出不符合格式。错误: {e}")
            last_exception = e
        except Exception as e:
            logger.error(f"在第 {attempt + 1} 次时间范围识别尝试中发生意外错误: {e}", exc_info=True)
            last_exception = e

        if attempt < max_retries - 1:
            await asyncio.sleep(0.5)

    logger.error(f"在 {max_retries} 次尝试后，无法有效识别时间范围。最后一次错误: {last_exception}")
    # 如果识别失败，返回默认值（不抛出异常，允许继续处理）
    return {
        "has_time_range": False,
        "end_date": None,
        "time_expression": None,
        "reason": "时间范围识别失败",
        "time_span_type": "long_term"  # 识别失败时默认视为长时间
    }


# 注意：原有的aiohttp_vllm_invoke_text函数已被移除，
# 现在直接使用clients.vllm_client模块中的对应函数


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
