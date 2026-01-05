# -*- coding: utf-8 -*-
"""
意图识别模块
包含第一层意图识别（统一意图分类）和第二层意图识别（奇门问题具体类型）
"""

import asyncio
import aiohttp
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, ValidationError
from pathlib import Path
from threading import Lock
from config import VLLM_API_BASE_URL, VLLM_MODEL_NAME, API_KEY

logger = logging.getLogger(__name__)

VLLM_SLOT_WAIT_TIMEOUT_SECONDS = 1500
VLLM_REQUEST_TIMEOUT_SECONDS = 500

# 提示词模板管理
_intent_prompt_template: Optional[str] = None
_qimen_type_prompt_template: Optional[str] = None
_time_range_prompt_template: Optional[str] = None
_prompt_template_lock = Lock()
_prompts_dir = Path(__file__).parent / "prompts"


def load_prompt_template(filename: str) -> Optional[str]:
    """从XML文件加载提示词模板"""
    try:
        prompt_file = _prompts_dir / filename
        if not prompt_file.exists():
            logger.error(f"提示词文件不存在: {prompt_file}")
            return None
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"成功加载提示词文件: {filename}")
        return content
    except Exception as e:
        logger.error(f"加载提示词文件 {filename} 失败: {e}", exc_info=True)
        return None


def reload_prompt_templates():
    """重新加载提示词模板（支持热更新）"""
    global _intent_prompt_template, _qimen_type_prompt_template, _time_range_prompt_template
    
    with _prompt_template_lock:
        logger.info("开始重新加载提示词模板...")
        _intent_prompt_template = load_prompt_template("intent_classification_prompt.xml")
        _qimen_type_prompt_template = load_prompt_template("qimen_type_classification_prompt.xml")
        _time_range_prompt_template = load_prompt_template("time_range_extraction_prompt.xml")
        if _intent_prompt_template and _qimen_type_prompt_template and _time_range_prompt_template:
            logger.info("✓ 提示词模板加载成功")
        else:
            logger.error("✗ 提示词模板加载失败")


def get_intent_prompt_template() -> Optional[str]:
    """获取意图分类提示词模板"""
    with _prompt_template_lock:
        if _intent_prompt_template is None:
            reload_prompt_templates()
        return _intent_prompt_template


def get_qimen_type_prompt_template() -> Optional[str]:
    """获取奇门类型分类提示词模板"""
    with _prompt_template_lock:
        if _qimen_type_prompt_template is None:
            reload_prompt_templates()
        return _qimen_type_prompt_template


def get_time_range_prompt_template() -> Optional[str]:
    """获取时间范围识别提示词模板"""
    with _prompt_template_lock:
        if _time_range_prompt_template is None:
            reload_prompt_templates()
        return _time_range_prompt_template


# ==================== 第一层意图识别 ====================

class QueryIntent(BaseModel):
    """
    用于验证 LLM 对用户问题意图分类结果的 Pydantic 模型。
    """
    query_type: Literal["specific_short_term", "general_long_term", "knowledge_question", "illegal_content", "general_knowledge", "self_intro", "non_ziwei_system", "qimen"] = Field(
        ...,
        description="问题的分类结果。'specific_short_term' 表示具体短期事件咨询；'general_long_term' 表示宏观长期解读；'knowledge_question' 表示命理专业知识问题；'illegal_content' 表示涉及违法犯罪的内容；'general_knowledge' 表示常识性知识或具体知识型问题（如数学、烹饪、天气等）；'self_intro' 表示用户在询问助手的身份或能力（如'你是谁''你能做什么'）；'non_ziwei_system' 表示非命理体系/与本系统无关的问题；'qimen' 表示奇门遁甲相关问题（具体时间点做具体事件是否合适、什么时间做具体事件、具体时间点做什么事件）。"
    )
    reason: str = Field(..., description="AI 做出分类判断的简要理由。但在输出的理由中不要提及：【因此归类为specific_short_term。】类似这种带有字段名称的话语，也就是不希望出现字段名称。")


# ==================== 时间范围识别 ====================

class TimeRangeExtraction(BaseModel):
    """时间范围识别结果"""
    has_time_range: bool = Field(..., description="用户问题中是否包含时间相关信息")
    end_date: Optional[str] = Field(None, description="时间范围结束日期，格式：YYYY-MM-DD")
    time_expression: Optional[str] = Field(None, description="用户问题中的原始时间表达")
    time_duration_type: Literal["short", "long"] = Field(
        ..., 
        description="时间长短类型：'short'表示一个月内（短时间），'long'表示一个月以上（长时间）。如果用户没有提及明确的时间，默认返回'long'（长时间）"
    )
    reason: str = Field(..., description="简要说明提取的时间范围和判断依据")


async def extract_time_range_with_llm(
    user_input: str,
    async_client: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5
) -> Dict[str, Any]:
    """
    使用 LLM 从用户问题中提取时间范围。
    
    参数:
        user_input: 用户输入
        async_client: aiohttp客户端会话
        semaphore: VLLM并发控制信号量
        max_retries: 最大重试次数
    """
    
    # 从XML文件加载提示词
    system_prompt_content = get_time_range_prompt_template()
    if not system_prompt_content:
        raise Exception("无法加载时间范围识别提示词模板")
    
    # 为大模型提供当前日期（格式：YYYY-MM-DD）
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    
    user_prompt_content = (
        f"<current_date>{current_date_str}</current_date>\n"
        f"<user_input>{user_input}</user_input>"
    )
    
    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_prompt_content},
            {"role": "assistant", "content": "{"}
        ],
        "temperature": 0.0,
        "max_tokens": 300,
        "response_format": {"type": "json_object", "schema": TimeRangeExtraction.model_json_schema()}
    }
    
    url = f"{VLLM_API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    last_exception = None
    for attempt in range(max_retries):
        logger.info(f"正在尝试进行时间范围提取 (第 {attempt + 1}/{max_retries} 次)...")
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=VLLM_SLOT_WAIT_TIMEOUT_SECONDS)
            try:
                timeout = aiohttp.ClientTimeout(total=VLLM_REQUEST_TIMEOUT_SECONDS)
                async with async_client.post(url, json=payload, headers=headers, timeout=timeout) as response:
                    response.raise_for_status()
                    json_response = await response.json()
                    content = json_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not content or not content.strip():
                        raise ValueError("VLLM返回的 content 为空。")
                    structured_response = _parse_lenient_json(content)
                    validated_model = TimeRangeExtraction.model_validate(structured_response)
                    logger.info("时间范围提取成功！")
                    return validated_model.model_dump()
            finally:
                semaphore.release()
                
        except (ValidationError, ValueError) as e:
            logger.warning(f"时间范围提取尝试 {attempt + 1} 失败，LLM输出不符合格式。错误: {e}")
            last_exception = e
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            logger.error(f"在第 {attempt + 1} 次时间范围提取尝试中发生网络错误: {e}")
            last_exception = e
        except Exception as e:
            logger.error(f"在第 {attempt + 1} 次时间范围提取尝试中发生意外错误: {e}", exc_info=True)
            last_exception = e
        
        if attempt < max_retries - 1:
            await asyncio.sleep(0.5)
    
    logger.error(f"在 {max_retries} 次尝试后，无法有效提取时间范围。最后一次错误: {last_exception}")
    # 如果提取失败，返回默认值（表示没有时间信息，默认返回长时间）
    return {
        "has_time_range": False,
        "end_date": None,
        "time_expression": None,
        "time_duration_type": "long",
        "reason": "时间范围提取失败，默认返回长时间（long）"
    }


async def answer_knowledge_question_suggestion(cleaned_question) -> str:
    """
    【奇门服务专用】对于知识问答类请求，建议用户使用紫薇服务
    """
    return f"""您的问题涉及命理专业知识，建议您使用紫微斗数服务获得更专业的解答。

紫微斗数特别适合：
- 专业术语和概念的详细解释
- 命盘结构的深入理解
- 长期运势和人生格局的分析

奇门服务则更擅长：
- 具体时间点的择时分析
- 通过时间确定适合的事件
- 通过事件确定适合的时间

如果您希望继续询问该问题请转到紫微斗数提问：{cleaned_question}
"""


async def classify_query_intent_with_llm(
    user_input: str, 
    async_client: aiohttp.ClientSession, 
    semaphore: asyncio.Semaphore,
    time_duration_type: Optional[str] = None,
    max_retries: int = 5
) -> Dict[str, Any]:
    """
    使用 LLM 对用户问题的意图进行分类（统一意图分类，包含奇门类型）。
    
    参数:
        user_input: 用户输入
        async_client: aiohttp客户端会话
        semaphore: VLLM并发控制信号量
        time_duration_type: 时间长短类型（"short"表示短时间一个月内，"long"表示长时间一个月以上，None表示未提供或跳过时间范围识别）
        max_retries: 最大重试次数
    """
    
    # 从XML文件加载提示词
    system_prompt_content = get_intent_prompt_template()
    if not system_prompt_content:
        raise Exception("无法加载意图分类提示词模板")
    
    # 为大模型提供当前时间，辅助判断"过去/未来"
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 构建用户prompt，包含时间长短信息（如果提供）
    user_prompt_content = (
        f"<current_time>{current_time_str}</current_time>\n"
        f"<user_input>{user_input}</user_input>"
    )
    
    # 如果提供了时间长短信息，添加到prompt中
    if time_duration_type:
        user_prompt_content += f"\n<time_duration_type>{time_duration_type}</time_duration_type>"
    
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
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    last_exception = None
    for attempt in range(max_retries):
        logger.info(f"正在尝试进行意图分类 (第 {attempt + 1}/{max_retries} 次)...")
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=VLLM_SLOT_WAIT_TIMEOUT_SECONDS)
            try:
                timeout = aiohttp.ClientTimeout(total=VLLM_REQUEST_TIMEOUT_SECONDS)
                async with async_client.post(url, json=payload, headers=headers, timeout=timeout) as response:
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


# ==================== 第二层意图识别 ====================

class SecondLayerIntent(BaseModel):
    """第二层意图识别结果"""
    qimen_type: Literal["type1", "type2", "type3"] = Field(..., description="奇门问题类型")
    # type1: 具体时间点做具体事件是否合适
    # type2: 什么时间做具体事件
    # type3: 具体时间点做什么事件
    time_range_start: Optional[str] = Field(None, description="时间范围开始时间，格式：YYYY-MM-DD HH:MM:SS。对于具体时辰，与time_range_end相同")
    time_range_end: Optional[str] = Field(None, description="时间范围结束时间，格式：YYYY-MM-DD HH:MM:SS。对于具体时辰，与time_range_start相同")
    specific_event: Optional[str] = Field(None, description="具体事件标签（需匹配数据库中的标签类型）")
    jixiong_preference: Optional[Literal["吉", "凶", "吉凶"]] = Field(
        "吉凶",
        description="用户倾向的吉凶筛选：'吉' 表示仅要吉，'凶' 表示仅要凶，'吉凶' 表示不按吉凶过滤，默认吉凶"
    )
    original_time_text: Optional[str] = Field(None, description="用户原始时间表述（如“明天下午三点”“今年夏天”）")
    reason: str = Field(..., description="判断理由")


async def classify_second_layer_intent(
    user_input: str,
    available_tags: list,
    current_time: str,
    async_client: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5
) -> Dict[str, Any]:
    """
    第二层意图识别: 判断奇门问题的具体类型并提取时间和事件
    
    需要从数据库中查询所有标签类型，传递给大模型进行匹配
    """
    
    # 从XML文件加载提示词
    template = get_qimen_type_prompt_template()
    if not template:
        raise Exception("无法加载奇门类型分类提示词模板")
    
    tags_text = "\n".join([f"- {tag}" for tag in available_tags])
    system_prompt = (
        template
        .replace("{available_tags}", tags_text)
        .replace("{current_time}", current_time)
    )

    user_prompt = f"""
    <user_input>{user_input}</user_input>

    <AVAILABLE_TAGS_JSON>
    {json.dumps(available_tags, ensure_ascii=False)}
    </AVAILABLE_TAGS_JSON>
    """
    
    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "{"}
        ],
        "temperature": 0.0,
        "max_tokens": 500,
        "response_format": {"type": "json_object"}
    }
    
    url = f"{VLLM_API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    last_exception = None
    for attempt in range(max_retries):
        logger.info(f"正在尝试进行第二层意图分类 (第 {attempt + 1}/{max_retries} 次)...")
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=1500)
            try:
                timeout = aiohttp.ClientTimeout(total=500)
                async with async_client.post(url, json=payload, headers=headers, timeout=timeout) as response:
                    response.raise_for_status()
                    json_response = await response.json()
                    content = json_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not content or not content.strip():
                        raise ValueError("VLLM返回的 content 为空。")
                    structured_response = _parse_lenient_json(content)
                    validated_model = SecondLayerIntent.model_validate(structured_response)
                    logger.info("第二层用户查询意图分类成功！")
                    return validated_model.model_dump()
            finally:
                semaphore.release()
                
        except (ValueError, ValidationError, Exception) as e:
            logger.warning(f"第二层意图分类尝试 {attempt + 1} 失败: {e}")
            last_exception = e
        
        if attempt < max_retries - 1:
            await asyncio.sleep(0.5)
    
    logger.error(f"在 {max_retries} 次尝试后，无法有效分类用户查询。最后一次错误: {last_exception}")
    raise Exception("AI多次尝试仍无法理解您的查询意图，请尝试换一种说法。")


# ==================== 通用工具函数 ====================

def _parse_lenient_json(json_string: str) -> dict:
    """
    尽力从一个包含任何已知错误的字符串中解析并修复JSON对象
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
    
    # 2. "诊断-修复-重试"循环
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
            
            # 策略A: 修复未闭合的字符串
            if "unterminated string" in error_msg:
                logger.info("诊断: 字符串未闭合。修复: 添加 '\"' 并尝试闭合括号。")
                current_string += '"'
                current_string = _close_open_brackets(current_string)
            
            # 策略B: 修复悬空的逗号
            elif "expecting value" in error_msg and stripped.endswith(','):
                logger.info("诊断: 悬空的逗号。修复: 移除末尾 ',' 并尝试闭合括号。")
                current_string = stripped[:-1]
                current_string = _close_open_brackets(current_string)
            
            # 策略C: 暴力回溯
            else:
                logger.info("诊断: 泛化的语法错误或严重截断。修复: 暴力回溯，丢弃最后一个不完整的条目。")
                
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
                    current_string = current_string[:last_comma_pos]
                    current_string = _close_open_brackets(current_string)
                else:
                    logger.error("无法进行回溯修复（未找到任何不在字符串内的逗号），放弃。")
                    raise e
            
            # 安全检查
            if current_string == original_string_for_this_attempt:
                logger.error("修复操作未能改变字符串，为避免死循环而放弃。")
                raise e
    
    raise json.JSONDecodeError("无法从模型响应中解析出任何有效的JSON片段", json_string, 0)


def _close_open_brackets(text: str) -> str:
    """辅助函数，计算并添加所有未闭合的括号。"""
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


def start_prompt_file_watcher():
    """
    启动提示词文件监控（支持热更新）
    注意：需要 watchdog 库支持
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class PromptFileHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if event.src_path.endswith('.xml'):
                    logger.info(f"检测到提示词文件变化: {event.src_path}")
                    reload_prompt_templates()
        
        observer = Observer()
        event_handler = PromptFileHandler()
        observer.schedule(event_handler, path=str(_prompts_dir), recursive=False)
        observer.start()
        logger.info(f"✓ 提示词文件监控已启动，监控目录: {_prompts_dir.absolute()}")
        return observer
    except ImportError:
        logger.warning("未安装 watchdog 库，热更新功能不可用。修改提示词需要重启服务。")
        return None
    except Exception as e:
        logger.error(f"启动提示词文件监控失败: {e}", exc_info=True)
        return None


# 初始化：在模块加载时预加载提示词
reload_prompt_templates()
