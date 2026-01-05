# services/chat_processor.py
import uuid
import json
import logging
import re
import time
import traceback
from datetime import date, datetime, timedelta
from typing import AsyncGenerator, Dict, Any, Optional, List, Tuple
import aiohttp
import asyncio
from fastapi import HTTPException, Request
from langchain_core.messages import HumanMessage, AIMessage
from starlette.responses import StreamingResponse

from database import db_manager
from config import (
    MISSING_BIRTH_INFO_MESSAGE,
    UNPREDICTABLE_FUTURE_MESSAGE,
    UNANSWERABLE_PREDICTION_MESSAGE,
    SENSITIVE_BIRTH_DATES,
    MIN_LENGTH_TO_START_YIELDING,
    MAX_STREAM_RETRIES,
    YES_NO_KEYWORDS,
    ANALYSIS_LEVEL_TO_CHINESE_NAME,
    ZIWEI_API_URL,
    ZIWEI_API_TIMEOUT_SECONDS,
    TRADITIONAL_HOUR_TO_TIME_INDEX,
    VLLM_MODEL_NAME,
)
from models import SignableAPIRequest
from monitoring import REQUESTS_RECEIVED
from services.monitor import StepMonitor, log_step, generate_request_id
from .session_manager import get_session_history, get_user_analysis_data, store_user_analysis_data
from clients.vllm_client import (
    extract_birth_info_with_llm,
    check_multi_time_analysis,
    classify_query_time_type,
    extract_detailed_intent_info,
    get_analysis_chain,
    aiohttp_vllm_stream,
)
from clients.external_api_client import robust_api_call_with_retry
from .ziwei_analyzer import generate_ziwei_analysis, generate_horoscope_analysis
from utils import (
    calculate_next_decadal_start_year,
    get_current_decadal_start_year,
    get_lunar_month_range_string,
    get_solar_month_range_string,
    calculate_all_decadal_periods,
    get_decadals_in_time_span,
    parse_multiple_years,
    get_judgment_word_bank_for_score,
    parse_fixed_format_birth_info,
    get_decadal_ganzhi_by_age,
    get_decadals_by_count,
    extract_decadal_age_range_from_api_response,
    get_lunar_day_string,
)
from .validation_rules import (
    is_gibberish,
    detect_critical_time_selection,
    detect_sensitive_political_content,
    detect_age_inquiry,
    detect_finance_or_lottery,
)
from .queryIntent import (
    classify_query_intent_with_llm,
    detect_time_range_with_llm,
    answer_knowledge_question,
    QIMEN_REFUSAL_MESSAGE,
)
from .chat_time_utils import (
    convert_hour_to_time_index,
    normalize_vague_time_expressions,
    smart_normalize_punctuation,
    normalize_punctuation_simple,
    is_birth_info_complete,
    calculate_lunar_age,
)
from prompt_logic import response_string_capability

logger = logging.getLogger(__name__)


def extract_core_question(original_question: str) -> str:
    """
    从原问题中提取核心问题部分，去掉时间相关的表述
    
    Args:
        original_question: 用户的原始问题，例如 "未来三年事业运如何"
    
    Returns:
        核心问题部分，例如 "事业运如何"
    """
    if not original_question:
        return "运势如何"
    
    # 常见的时间表述模式（需要去掉的部分）
    time_patterns = [
        r'未来\s*\d+\s*年',
        r'未来\s*\d+\s*个月',
        r'未来\s*\d+\s*天',
        r'未来\s*\d+\s*周',
        r'未来\s*\d+\s*小时',
        r'未来\s*几\s*年',
        r'未来\s*几\s*个月',
        r'未来\s*几\s*天',
        r'未来\s*几\s*周',
        r'未来\s*几\s*小时',
        r'下\s*\d+\s*年',
        r'下\s*\d+\s*个月',
        r'下\s*\d+\s*天',
        r'下\s*\d+\s*周',
        r'下\s*\d+\s*小时',
        r'下\s*几\s*年',
        r'下\s*几\s*个月',
        r'下\s*几\s*天',
        r'下\s*几\s*周',
        r'下\s*几\s*小时',
        r'今后\s*\d+\s*年',
        r'今后\s*\d+\s*个月',
        r'今后\s*\d+\s*天',
        r'今后\s*\d+\s*周',
        r'今后\s*\d+\s*小时',
        r'今后\s*几\s*年',
        r'今后\s*几\s*个月',
        r'今后\s*几\s*天',
        r'今后\s*几\s*周',
        r'今后\s*几\s*小时',
        r'接下来\s*\d+\s*年',
        r'接下来\s*\d+\s*个月',
        r'接下来\s*\d+\s*天',
        r'接下来\s*\d+\s*周',
        r'接下来\s*\d+\s*小时',
        r'接下来\s*几\s*年',
        r'接下来\s*几\s*个月',
        r'接下来\s*几\s*天',
        r'接下来\s*几\s*周',
        r'接下来\s*几\s*小时',
        r'\d{4}\s*年\s*到\s*\d{4}\s*年',
        r'\d{4}\s*年\s*至\s*\d{4}\s*年',
        r'\d{4}\s*年\s*-\s*\d{4}\s*年',
        r'\d{4}\s*年\s*~\s*\d{4}\s*年',
        r'从\s*\d{4}\s*年\s*\d{1,2}\s*月\s*到\s*\d{4}\s*年\s*\d{1,2}\s*月',  # 从2026年1月到2026年3月
        r'从\s*\d{4}\s*年\s*\d{1,2}\s*月\s*至\s*\d{4}\s*年\s*\d{1,2}\s*月',  # 从2026年1月至2026年3月
        r'\d{4}\s*年\s*\d{1,2}\s*月\s*到\s*\d{4}\s*年\s*\d{1,2}\s*月',  # 2026年1月到2026年3月
        r'\d{4}\s*年\s*\d{1,2}\s*月\s*至\s*\d{4}\s*年\s*\d{1,2}\s*月',  # 2026年1月至2026年3月
        r'今年',
        r'明年',
        r'后年',
        r'当年',
        r'这一年',
        r'这年',
        r'本年',
        r'下一年',
        r'下年',
        r'后一年',
        r'下个季度',
        r'下季度',
        r'这个季度',
        r'本季度',
        r'当前大运.*?下个大运',
        r'当前大运.*?下个.*?大运',
        r'下个大运.*?当前大运',
        r'下个.*?大运.*?当前.*?大运',
        r'大运',
        r'十年大运',
        r'未来.*?大运',
        r'今后.*?大运',
        r'接下来.*?大运',
        # 模糊时间表达
        r'过\s*几\s*年',
        r'过\s*几\s*个月',
        r'过\s*几\s*天',
        r'过\s*几\s*周',
        r'过\s*几\s*小时',
        r'过\s*几\s*年',
        r'过\s*几\s*个月',
        r'过\s*几\s*天',
        r'过\s*几\s*周',
        r'过\s*几\s*小时',
        r'下一\s*阶段',
        r'下一个\s*阶段',
        r'下个\s*阶段',
        r'下一\s*时期',
        r'下一个\s*时期',
        r'下个\s*时期',
        # 年内/年后/年前表达
        r'年内',
        r'年后',
        r'年前',
        # 段时间表达
        r'过\s*一\s*段\s*时间',
        r'过\s*段\s*时间',
        r'一\s*段\s*时间',
        r'这\s*段\s*时间',
    ]
    
    core_question = original_question
    for pattern in time_patterns:
        core_question = re.sub(pattern, '', core_question, flags=re.IGNORECASE)
    
    # 清理多余的空格和标点
    core_question = re.sub(r'\s+', ' ', core_question).strip()
    core_question = re.sub(r'^[，,。.、]', '', core_question)
    core_question = re.sub(r'[，,。.、]$', '', core_question)
    
    # 如果清理后为空，返回默认值
    if not core_question or core_question.strip() == '':
        return "运势如何"
    
    return core_question.strip()


async def handle_multi_time_analysis_check(
    prompt_input: str,
    current_time_for_llm: datetime,
    birth_info_to_use: Dict[str, Any],
    persisted_data: Dict[str, Any],
    current_age_cached: Optional[int],
    session_id: str,
    monitor_request_id: str
) -> Optional[Tuple[bool, str]]:
    """
    检查是否为多流年/多流月/多大运类型，如果是则返回澄清响应
    
    Args:
        prompt_input: 用户输入
        current_time_for_llm: 当前时间
        birth_info_to_use: 出生信息
        persisted_data: 持久化数据
        current_age_cached: 缓存的当前年龄
        session_id: 会话ID
        monitor_request_id: 监控请求ID
    
    Returns:
        如果是多流年/多流月/多大运类型，返回 (True, response_text)
        否则返回 None
    """
    from models import MultiTimeAnalysisResult
    from .queryIntent import remove_birth_info_from_query
    
    # 清理用户输入，去掉出生年月日等信息，保留原始问题
    cleaned_prompt = remove_birth_info_from_query(prompt_input)
    
    # 检查是否为多流年/多大运/多流月类型
    multi_time_result = await check_multi_time_analysis(prompt_input, current_time_for_llm)
    
    # 强制单流年检查：如果包含"年内"、"上半年"、"下半年"、"年后"，强制走单流年
    force_single_yearly_keywords = ["年内", "上半年", "下半年", "年后"]
    if multi_time_result and any(keyword in prompt_input for keyword in force_single_yearly_keywords):
        logger.info(f"[会话: {session_id}] 检测到强制单流年关键词，将multi_time_result.query_type强制设置为none")
        # 创建一个新的结果对象，将query_type设置为none
        multi_time_result = MultiTimeAnalysisResult(
            query_type="none",
            multi_yearly_years=None,
            multi_decadal_span=None,
            multi_month_span=None,
            relative_time_indicator=None
        )
    
    if multi_time_result and multi_time_result.query_type != "none":
        # 判断为三种类型之一，直接返回拒绝/澄清内容
        query_type = multi_time_result.query_type
        core_question = extract_core_question(prompt_input)
        
        if query_type == "multi_yearly_analysis":
            years_to_analyze = multi_time_result.multi_yearly_years or []
            if years_to_analyze and len(years_to_analyze) >= 2:
                years_to_show = years_to_analyze[:3]
                header_lines = [
                    f"您这个问您这个问题非常好，您想要从宏观上把握人生的节奏，这是一个非常有远见的想法。但是人生每个周期都有自己的主题，为了给您更深入、更精准的分析，您目前更想要深入了解哪个问题？",
                    "",
                    "@ 是否想问"
                ]
                # 【关键修复】优先使用大模型生成的选项
                if multi_time_result.suggestion_options and len(multi_time_result.suggestion_options) > 0:
                    # 清理选项中的[DONE]标记和范围性提问
                    filtered_options = []
                    for opt in multi_time_result.suggestion_options[:3]:
                        cleaned_opt = opt.replace('[DONE]', '').strip()
                        # 过滤掉包含"哪个阶段"等范围性提问的选项
                        if not re.search(r'哪个\s*阶段', cleaned_opt, re.IGNORECASE):
                            filtered_options.append(cleaned_opt)
                    if filtered_options:
                        option_lines = [f"- {opt}" for opt in filtered_options]
                        logger.info(f"[多流年分析] 使用大模型生成的选项（已过滤范围性提问）: {option_lines}")
                    else:
                        # 如果所有选项都被过滤掉，回退到拼接方式
                        logger.warning(f"[多流年分析] 大模型生成的选项都包含范围性提问，回退到拼接方式")
                        option_lines = None
                
                if not option_lines:
                    # 回退到拼接方式
                    # 清理core_question，移除可能残留的模糊时间表达
                    clean_core_question = core_question
                    # 移除模糊时间表达（这些不应该出现在具体年份的选项中）
                    clean_core_question = re.sub(r'过\s*几\s*年', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'过\s*几\s*个月', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'今后\s*几\s*年', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'今后\s*几\s*个月', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'未来\s*几\s*年', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'未来\s*几\s*个月', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'接下来\s*几\s*年', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'接下来\s*几\s*个月', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'下\s*几\s*年', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'下\s*几\s*个月', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'下一\s*阶段', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'下一个\s*阶段', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'下个\s*阶段', '', clean_core_question, flags=re.IGNORECASE)
                    # 移除范围性提问表达（禁止在追问问题中出现范围性问题）
                    clean_core_question = re.sub(r'哪个\s*阶段', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'哪个\s*阶段\s*能', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'哪个\s*阶段\s*更', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'哪个\s*阶段\s*适合', '', clean_core_question, flags=re.IGNORECASE)
                    # 移除段时间表达
                    clean_core_question = re.sub(r'过\s*一\s*段\s*时间', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'过\s*段\s*时间', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'一\s*段\s*时间', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'这\s*段\s*时间', '', clean_core_question, flags=re.IGNORECASE)
                    # 清理多余空格和标点
                    clean_core_question = re.sub(r'\s+', ' ', clean_core_question).strip()
                    clean_core_question = re.sub(r'^[，,。.、]', '', clean_core_question)
                    clean_core_question = re.sub(r'[，,。.、]$', '', clean_core_question)
                    if not clean_core_question:
                        clean_core_question = "运势如何"
                    # 生成选项时，如果clean_core_question以"的"开头，去掉"的"
                    if clean_core_question.startswith('的'):
                        clean_core_question = clean_core_question[1:].strip()
                    # 【关键修复】清理可能残留的[DONE]标记
                    clean_core_question = clean_core_question.replace('[DONE]', '').strip()
                    option_lines = [f"- {y}年的{clean_core_question}" for y in years_to_show]
                    logger.info(f"[多流年分析] 使用拼接方式生成选项: {option_lines}")
                clarification_text = "\n".join(header_lines + option_lines)
                
                log_step(
                    "澄清：多流年问题，需要用户选择具体年份",
                    request_id=monitor_request_id,
                    status="澄清",
                    extra_data={
                        "years_to_analyze": years_to_analyze,
                        "session_id": session_id,
                        "scope": f"{years_to_analyze[0]}-{years_to_analyze[-1]}",
                        "core_question": core_question
                    }
                )
                return (True, clarification_text)
        
        elif query_type == "multi_monthly_analysis":
            span_months = multi_time_result.multi_month_span or 0
            if span_months > 1:
                # 生成月份列表
                months_to_analyze = []
                current_dt = datetime.now()
                cur_year, cur_month = current_dt.year, current_dt.month
                for _ in range(span_months):
                    months_to_analyze.append((cur_year, cur_month))
                    cur_month += 1
                    if cur_month > 12:
                        cur_month = 1
                        cur_year += 1
                
                months_sample = months_to_analyze[:3]
                header_lines = [
                    f"您这个问题非常好，您想要从宏观上把握人生的节奏，这是一个非常有远见的想法。但是人生每个周期都有自己的主题，为了给您更深入、更精准的分析，您目前更想要深入了解哪个问题？",
                    "",
                    "@ 是否想问"
                ]
                # 【关键修复】优先使用大模型生成的选项
                if multi_time_result.suggestion_options and len(multi_time_result.suggestion_options) > 0:
                    # 清理选项中的[DONE]标记和范围性提问
                    filtered_options = []
                    for opt in multi_time_result.suggestion_options[:3]:
                        cleaned_opt = opt.replace('[DONE]', '').strip()
                        # 过滤掉包含"哪个阶段"等范围性提问的选项
                        if not re.search(r'哪个\s*阶段', cleaned_opt, re.IGNORECASE):
                            filtered_options.append(cleaned_opt)
                    if filtered_options:
                        option_lines = [f"- {opt}" for opt in filtered_options]
                        logger.info(f"[多流月分析] 使用大模型生成的选项（已过滤范围性提问）: {option_lines}")
                    else:
                        # 如果所有选项都被过滤掉，回退到拼接方式
                        logger.warning(f"[多流月分析] 大模型生成的选项都包含范围性提问，回退到拼接方式")
                        option_lines = None
                
                if not option_lines:
                    # 回退到拼接方式
                    # 进一步清理core_question，移除可能残留的时间范围表述和模糊时间表达
                    # 移除时间范围表述
                    clean_core_question = re.sub(r'从\s*\d{4}\s*年\s*\d{1,2}\s*月\s*到\s*\d{4}\s*年\s*\d{1,2}\s*月', '', core_question)
                    clean_core_question = re.sub(r'从\s*\d{4}\s*年\s*\d{1,2}\s*月\s*至\s*\d{4}\s*年\s*\d{1,2}\s*月', '', clean_core_question)
                    clean_core_question = re.sub(r'\d{4}\s*年\s*\d{1,2}\s*月\s*到\s*\d{4}\s*年\s*\d{1,2}\s*月', '', clean_core_question)
                    clean_core_question = re.sub(r'\d{4}\s*年\s*\d{1,2}\s*月\s*至\s*\d{4}\s*年\s*\d{1,2}\s*月', '', clean_core_question)
                    # 移除模糊时间表达（这些不应该出现在具体月份的选项中）
                    clean_core_question = re.sub(r'过\s*几\s*年', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'过\s*几\s*个月', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'过\s*几\s*天', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'过\s*几\s*周', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'今后\s*几\s*年', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'今后\s*几\s*个月', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'未来\s*几\s*年', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'未来\s*几\s*个月', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'接下来\s*几\s*年', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'接下来\s*几\s*个月', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'下\s*几\s*年', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'下\s*几\s*个月', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'下一\s*阶段', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'下一个\s*阶段', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'下个\s*阶段', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'下一\s*时期', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'下一个\s*时期', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'下个\s*时期', '', clean_core_question, flags=re.IGNORECASE)
                    # 移除范围性提问表达（禁止在追问问题中出现范围性问题）
                    clean_core_question = re.sub(r'哪个\s*阶段', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'哪个\s*阶段\s*能', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'哪个\s*阶段\s*更', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'哪个\s*阶段\s*适合', '', clean_core_question, flags=re.IGNORECASE)
                    # 移除段时间表达
                    clean_core_question = re.sub(r'过\s*一\s*段\s*时间', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'过\s*段\s*时间', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'一\s*段\s*时间', '', clean_core_question, flags=re.IGNORECASE)
                    clean_core_question = re.sub(r'这\s*段\s*时间', '', clean_core_question, flags=re.IGNORECASE)
                    # 清理多余空格和标点
                    clean_core_question = re.sub(r'\s+', ' ', clean_core_question).strip()
                    clean_core_question = re.sub(r'^[，,。.、]', '', clean_core_question)
                    clean_core_question = re.sub(r'[，,。.、]$', '', clean_core_question)
                    if not clean_core_question:
                        clean_core_question = "运势如何"
                    # 生成选项时，如果clean_core_question以"的"开头，去掉"的"
                    if clean_core_question.startswith('的'):
                        clean_core_question = clean_core_question[1:].strip()
                    # 【关键修复】清理可能残留的[DONE]标记
                    clean_core_question = clean_core_question.replace('[DONE]', '').strip()
                    option_lines = [f"- {y}年{m}月的{clean_core_question}" for (y, m) in months_sample]
                    logger.info(f"[多流月分析] 使用拼接方式生成选项: {option_lines}")
                clarification_text = "\n".join(header_lines + option_lines)
                
                log_step(
                    "澄清：多流月问题，需要用户选择具体月份",
                    request_id=monitor_request_id,
                    status="澄清",
                    extra_data={
                        "months_to_analyze": months_to_analyze,
                        "session_id": session_id,
                        "span": span_months,
                        "core_question": core_question
                    }
                )
                return (True, clarification_text)
        
        elif query_type == "multi_decadal_analysis":
            # 多大运分析，复用现有逻辑
            logger.info(f"检测到多大运分析问题: {prompt_input}")
            
            # 获取大运信息用于生成追问
            birth_year = birth_info_to_use.get('year')
            current_age = current_age_cached if current_age_cached is not None else calculate_lunar_age(birth_info_to_use)
            all_decadal_ages = persisted_data.get('all_decadal_ages', {})
            
            # 如果没有大运信息，先获取
            if not all_decadal_ages:
                try:
                    payload_api = build_ziwei_api_payload(birth_info_to_use)
                    natal_api_response = await robust_api_call_with_retry(
                        url=ZIWEI_API_URL,
                        payload=payload_api,
                        timeout=ZIWEI_API_TIMEOUT_SECONDS
                    )
                    astrolabe_data = natal_api_response.get('data', {}).get('astrolabe', {})
                    gender = astrolabe_data.get("gender", "")
                    year_gan = astrolabe_data.get('rawDates', {}).get('chineseDate', {}).get('yearly', [None])[0]
                    wuxingju = astrolabe_data.get('fiveElementsClass', '')
                    palaces_list = astrolabe_data.get('palaces', [])
                    if gender and year_gan and wuxingju:
                        all_decadal_ages = calculate_all_decadal_periods(
                            birth_year,
                            gender,
                            year_gan,
                            wuxingju,
                            palaces_list
                        )
                        if all_decadal_ages:
                            persisted_data['all_decadal_ages'] = all_decadal_ages
                except Exception as e:
                    logger.warning(f"获取大运信息失败: {e}")
            
            # 【关键修复】优先使用大模型生成的选项
            if multi_time_result.suggestion_options and len(multi_time_result.suggestion_options) > 0:
                # 清理选项中的[DONE]标记
                followup_lines = [f"- {opt.replace('[DONE]', '').strip()}" for opt in multi_time_result.suggestion_options[:2]]
                logger.info(f"[多大运分析] 使用大模型生成的选项: {followup_lines}")
            else:
                # 回退到固定选项
                followup_lines = [
                    f"- 当前大运如何",
                    f"- 下个大运如何"
                ]
                logger.info(f"[多大运分析] 使用固定选项: {followup_lines}")
            
            comparison_response = f"""您这个问题非常好，您想要从宏观上把握人生的节奏，这是一个非常有远见的想法。但是人生每个周期都有自己的主题，为了给您更深入、更精准的分析，您目前更想要深入了解哪个问题？

@ 是否想问
{chr(10).join(followup_lines)}"""
            
            log_step(
                "多大运分析：引导用户选择",
                request_id=monitor_request_id,
                status="成功",
                extra_data={"reason": "multi_decadal_analysis", "session_id": session_id, "core_question": core_question}
            )
            return (True, comparison_response)
    
    return None


async def extract_query_intent_with_time_analysis(
    prompt_input: str,
    current_time_for_llm: datetime,
    time_range_result: Optional[Dict[str, Any]],
    suggested_intent: Optional[str],
    last_horoscope_date_used: Optional[Dict[str, Any]],
    last_relevant_palaces: Optional[List[str]],
    session_id: str,
    max_business_retries: int = 3,
    enable_intent_mapping: bool = True
) -> Dict[str, Any]:
    """
    通用的推演周期提取方法，统一处理两种模式下的意图提取逻辑
    
    Args:
        prompt_input: 用户输入
        current_time_for_llm: 当前时间
        time_range_result: 时间范围识别结果（可选）
        suggested_intent: 建议的意图（用于跳过分支，可选）
        last_horoscope_date_used: 上次使用的运势日期（可选）
        last_relevant_palaces: 上次相关的宫位（可选）
        session_id: 会话ID
        max_business_retries: 最大业务重试次数，默认3次
        enable_intent_mapping: 是否启用意图映射，默认True
    
    Returns:
        query_intent_data: 查询意图数据字典
    """
    intent_extract_start = time.time()
    
    # 第一步：查询类型分类
    query_type_result = None
    time_range_info_for_classify = time_range_result if time_range_result else None
    
    for attempt in range(max_business_retries):
        query_type_result = await classify_query_time_type(
            prompt_input,
            current_time_for_llm,
            time_range_info=time_range_info_for_classify,
            max_retries=3
        )
        if query_type_result:
            logger.info(f"✅ 查询类型分类成功：{query_type_result.query_time_type}")
            break
        logger.warning(f"第 {attempt + 1} 次查询类型分类失败，重试...")
        if attempt < max_business_retries - 1:
            await asyncio.sleep(0.5)
    
    # 兜底逻辑：查询类型分类失败
    if not query_type_result:
        logger.warning("查询类型分类失败，使用兜底逻辑")
        if suggested_intent:
            # 跳过分支：使用suggested_intent
            query_intent_data = {
                "intent_type": suggested_intent or "horoscope_analysis",
                "analysis_level": "yearly" if suggested_intent != "knowledge_question" else "knowledge_question",
                "relevant_palaces": []
            }
        else:
            # 不跳过分支：使用birth_chart_analysis
            query_intent_data = {
                "intent_type": "birth_chart_analysis",
                "analysis_level": "birth_chart",
                "relevant_palaces": []
            }
        return query_intent_data
    
    # 第二步：详细意图提取
    detailed_result = await extract_detailed_intent_info(
        prompt_input,
        query_type_result.query_time_type,
        query_type_result.time_expression,
        current_time_for_llm,
        max_retries=3
    )
    
    # 兜底逻辑：详细意图提取失败
    if not detailed_result:
        logger.warning("详细意图提取失败，使用兜底逻辑")
        if suggested_intent:
            # 跳过分支：使用suggested_intent
            query_intent_data = {
                "intent_type": suggested_intent or "horoscope_analysis",
                "analysis_level": "yearly" if suggested_intent != "knowledge_question" else "knowledge_question",
                "relevant_palaces": []
            }
        else:
            # 不跳过分支：使用birth_chart_analysis
            query_intent_data = {
                "intent_type": "birth_chart_analysis",
                "analysis_level": "birth_chart",
                "relevant_palaces": []
            }
        return query_intent_data
    
    # 根据查询类型和时间表达式解析日期和时间信息
    query_time_type = query_type_result.query_time_type
    
    # 优先使用替换后的时间表达（如果time_range_result中有替换后的值）
    if time_range_result and isinstance(time_range_result, dict):
        replaced_time_expression = time_range_result.get("time_expression")
        if replaced_time_expression and replaced_time_expression != prompt_input:
            # 检查是否包含强制替换的关键词，如果包含则使用替换后的值
            force_replace_keywords = ["年内", "上半年", "下半年", "年后"]
            original_in_prompt = any(keyword in prompt_input for keyword in force_replace_keywords)
            if original_in_prompt and any(keyword in replaced_time_expression for keyword in ["年"]):
                time_expression = replaced_time_expression
                logger.info(f"[会话: {session_id}] 使用替换后的时间表达: '{time_expression}'")
            else:
                time_expression = query_type_result.time_expression or detailed_result.relative_time_indicator or prompt_input
        else:
            time_expression = query_type_result.time_expression or detailed_result.relative_time_indicator or prompt_input
    else:
        time_expression = query_type_result.time_expression or detailed_result.relative_time_indicator or prompt_input
    
    # 构建基础意图数据
    query_intent_data = {
        "relevant_palaces": detailed_result.relevant_palaces if detailed_result.relevant_palaces else [],
        "is_query_about_other_person": detailed_result.is_about_other,
        "relationship": detailed_result.relationship,
    }
    
    # 根据查询类型处理
    if query_time_type == "lifetime":
        query_intent_data = build_intent_from_lifetime_query(detailed_result, query_intent_data)
    elif query_time_type == "single_decadal":
        query_intent_data = build_intent_from_single_decadal_query(
            detailed_result, time_expression, query_intent_data
        )
    else:
        query_intent_data = build_intent_from_temporal_query(
            query_time_type,
            time_expression,
            detailed_result,
            current_time_for_llm,
            last_horoscope_date_used,
            last_relevant_palaces,
            query_intent_data
        )
    
    # 意图映射处理（如果启用）
    if enable_intent_mapping and suggested_intent:
        problematic_intents = ["specific_short_term", "general_question", "unpredictable_future",
                               "unanswerable_question", "irrelevant_question"]
        if query_intent_data.get("intent_type") in problematic_intents:
            logger.info(
                f"[会话: {session_id}] LLM返回了问题意图: {query_intent_data.get('intent_type')}，使用映射意图: {suggested_intent}")
            query_intent_data["intent_type"] = suggested_intent
        
        # 如果analysis_level为空，根据映射后的意图设置
        if not query_intent_data.get("analysis_level"):
            if suggested_intent == "knowledge_question":
                query_intent_data["analysis_level"] = "knowledge_question"
            else:
                query_intent_data["analysis_level"] = "yearly"
    
    # 耗时统计
    intent_extract_time = time.time() - intent_extract_start
    logger.info(f"⏱️ LLM查询意图提取耗时: {intent_extract_time:.2f}秒")
    logger.info(f"[会话: {session_id}] 意图提取成功: {query_intent_data}")
    
    return query_intent_data


# ========== Query Type 处理函数 ==========

def build_intent_from_lifetime_query(
    detailed_result: Any,
    query_intent_data: Dict[str, Any]
) -> Dict[str, Any]:
    """处理一生查询（lifetime）"""
    query_intent_data["intent_type"] = "birth_chart_analysis"
    query_intent_data["analysis_level"] = "birth_chart"
    return query_intent_data


def build_intent_from_single_decadal_query(
    detailed_result: Any,
    time_expression: Optional[str],
    query_intent_data: Dict[str, Any]
) -> Dict[str, Any]:
    """处理单大运查询（single_decadal）"""
    query_intent_data["intent_type"] = "single_decadal_analysis"
    query_intent_data["analysis_level"] = "decadal"
    query_intent_data["explicit_decadal_ganzhi"] = detailed_result.explicit_decadal_ganzhi
    if detailed_result.is_next_decadal_query:
        query_intent_data["relative_time_indicator"] = "下个大运"
    elif detailed_result.is_next_decadal_query is False:
        query_intent_data["relative_time_indicator"] = "当前大运"
    else:
        query_intent_data["relative_time_indicator"] = time_expression or "下个大运"
    return query_intent_data


def build_intent_from_temporal_query(
    query_time_type: str,
    time_expression: str,
    detailed_result: Any,
    current_time_for_llm: datetime,
    last_horoscope_date_used: Optional[Dict[str, Any]],
    last_relevant_palaces: Optional[List[str]],
    query_intent_data: Dict[str, Any]
) -> Dict[str, Any]:
    """处理单流年/流月/流日查询（single_yearly/single_monthly/single_daily）"""
    # 直接使用 detailed_result 中 LLM 解析的时间字段，不再使用规则函数 resolve_horoscope_date
    
    logger.info(f"[时间解析] 使用LLM解析的时间字段: resolved_horoscope_date={getattr(detailed_result, 'resolved_horoscope_date', None) if detailed_result else None}, analysis_level={getattr(detailed_result, 'analysis_level', None) if detailed_result else None}, relative_time_indicator={getattr(detailed_result, 'relative_time_indicator', None) if detailed_result else None}")
    
    # 从 detailed_result 中提取时间字段
    resolved_horoscope_date = getattr(detailed_result, 'resolved_horoscope_date', None) if detailed_result else None
    target_year = getattr(detailed_result, 'target_year', None) if detailed_result else None
    target_month = getattr(detailed_result, 'target_month', None) if detailed_result else None
    target_day = getattr(detailed_result, 'target_day', None) if detailed_result else None
    target_hour = getattr(detailed_result, 'target_hour', None) if detailed_result else None
    target_minute = getattr(detailed_result, 'target_minute', None) if detailed_result else None
    analysis_level = getattr(detailed_result, 'analysis_level', None) if detailed_result else None
    relative_time_indicator = getattr(detailed_result, 'relative_time_indicator', None) if detailed_result else None
    
    # 上下文继承逻辑：如果 LLM 没有解析出日期，尝试从历史记录继承
    if not resolved_horoscope_date and last_horoscope_date_used and isinstance(last_horoscope_date_used, dict):
        if "resolved_horoscope_date" in last_horoscope_date_used and last_horoscope_date_used["resolved_horoscope_date"]:
            resolved_horoscope_date = last_horoscope_date_used["resolved_horoscope_date"]
        if "target_year" in last_horoscope_date_used and last_horoscope_date_used["target_year"]:
            target_year = last_horoscope_date_used["target_year"]
        if "target_month" in last_horoscope_date_used and last_horoscope_date_used["target_month"]:
            target_month = last_horoscope_date_used["target_month"]
        if "target_day" in last_horoscope_date_used and last_horoscope_date_used["target_day"]:
            target_day = last_horoscope_date_used["target_day"]
        if "target_hour" in last_horoscope_date_used and last_horoscope_date_used["target_hour"]:
            target_hour = last_horoscope_date_used["target_hour"]
        if "target_minute" in last_horoscope_date_used and last_horoscope_date_used["target_minute"]:
            target_minute = last_horoscope_date_used["target_minute"]
        if "analysis_level" in last_horoscope_date_used and last_horoscope_date_used["analysis_level"]:
            analysis_level = last_horoscope_date_used["analysis_level"]
    
    # 如果 LLM 解析成功或从历史继承成功
    if resolved_horoscope_date:
        query_intent_data["intent_type"] = "horoscope_analysis"
        analysis_level_map = {
            "single_yearly": "yearly",
            "single_monthly": "monthly",
            "single_daily": "daily"
        }
        # 优先使用 LLM 解析的 analysis_level，否则使用 query_time_type 映射
        query_intent_data["analysis_level"] = analysis_level or analysis_level_map.get(query_time_type, "yearly")
        query_intent_data["resolved_horoscope_date"] = resolved_horoscope_date
        query_intent_data["target_year"] = target_year
        query_intent_data["target_month"] = target_month
        query_intent_data["target_day"] = target_day
        # 设置默认值：如果 target_hour 为 None，默认 12；如果 target_minute 为 None，默认 0
        query_intent_data["target_hour"] = target_hour if target_hour is not None else 12
        query_intent_data["target_minute"] = target_minute if target_minute is not None else 0
        query_intent_data["relative_time_indicator"] = relative_time_indicator
        
        logger.info(f"[时间解析] 成功设置时间字段: resolved_horoscope_date={resolved_horoscope_date}, analysis_level={query_intent_data['analysis_level']}, target_year={target_year}, target_month={target_month}, target_day={target_day}")
        
        # 检查是否为后续问题（时间继承且宫位相同）
        # 注意：只有当时间是从历史继承的（而不是LLM新解析的）时，才认为是后续问题
        llm_resolved_date = getattr(detailed_result, 'resolved_horoscope_date', None) if detailed_result else None
        is_time_inherited = (not llm_resolved_date and 
                            last_horoscope_date_used and 
                            resolved_horoscope_date == last_horoscope_date_used.get("resolved_horoscope_date"))
        
        logger.info(f"[后续问题检测] LLM解析日期={llm_resolved_date}, 最终使用日期={resolved_horoscope_date}, 是否继承={is_time_inherited}, 历史日期={last_horoscope_date_used.get('resolved_horoscope_date') if last_horoscope_date_used else None}, 当前宫位={query_intent_data.get('relevant_palaces')}, 历史宫位={last_relevant_palaces}")
        
        if (query_intent_data.get("resolved_horoscope_date") and 
            is_time_inherited and
            last_relevant_palaces and
            sorted(query_intent_data.get("relevant_palaces", [])) == sorted(last_relevant_palaces)):
            logger.info(f"[后续问题检测] ✅ 检测到后续问题：时间继承自历史记录，宫位相同，将转换为general_question")
            query_intent_data["intent_type"] = "general_question"
            query_intent_data["analysis_level"] = "general_question"
            query_intent_data["resolved_horoscope_date"] = None
            query_intent_data["target_year"] = None
            query_intent_data["target_month"] = None
            query_intent_data["target_day"] = None
            query_intent_data["target_hour"] = None
            query_intent_data["target_minute"] = None
            query_intent_data["relative_time_indicator"] = None
        else:
            logger.info(f"[后续问题检测] ❌ 不是后续问题，保持horoscope_analysis")
            # 如果不是后续问题，确保 analysis_level 正确设置
            if not query_intent_data.get("analysis_level"):
                analysis_level_map = {
                    "single_yearly": "yearly",
                    "single_monthly": "monthly",
                    "single_daily": "daily"
                }
                query_intent_data["analysis_level"] = analysis_level or analysis_level_map.get(query_time_type, "yearly")
    else:
        # LLM 解析失败，使用兜底逻辑
        logger.warning(f"[时间解析] LLM解析失败，time_expression='{time_expression}', detailed_result.resolved_horoscope_date={resolved_horoscope_date}, detailed_result.analysis_level={analysis_level}")
        query_intent_data["intent_type"] = "birth_chart_analysis"
        query_intent_data["analysis_level"] = "birth_chart"
    
    return query_intent_data


# ========== Intent Type 处理函数 ==========

async def handle_single_decadal_analysis(
    query_intent_data: Dict[str, Any],
    birth_info_to_use: Dict[str, Any],
    persisted_data: Dict[str, Any],
    current_age_cached: Optional[int],
    time_indicator: str,
    monitor_request_id: str,
    session_id: str
) -> Optional[str]:
    """
    处理单个大运分析（single_decadal_analysis）
    
    Returns:
        如果返回字符串，表示需要直接返回给用户的响应；如果返回None，表示继续后续流程
    """
    logger.info(f"检测到单个大运查询: {time_indicator}")

    # 获取五行局信息
    wuxingju = persisted_data.get('wuxingju')
    if not wuxingju:
        payload_api = build_ziwei_api_payload(birth_info_to_use)
        try:
            api_call_start = time.time()
            with StepMonitor(
                "成功调用ziweiapi",
                request_id=monitor_request_id,
                extra_data={"scene": "single_decadal_base"},
            ):
                natal_api_response = await robust_api_call_with_retry(
                    url=ZIWEI_API_URL,
                    payload=payload_api,
                    timeout=ZIWEI_API_TIMEOUT_SECONDS)
            api_call_time = time.time() - api_call_start
            logger.info(f"⏱️ 外部API调用耗时: {api_call_time:.2f}秒")
            wuxingju = natal_api_response.get('data', {}).get('astrolabe', {}).get('fiveElementsClass')
            if wuxingju:
                persisted_data['wuxingju'] = wuxingju
        except aiohttp.ClientError as e:
            logger.error(f"为获取五行局而进行的本命盘排盘失败: {e}")
            return "在尝试获取您的命盘基础信息时出错，无法继续分析。请稍后再试。"

    birth_year = birth_info_to_use.get('year')
    if wuxingju and birth_year:
        current_age = current_age_cached if current_age_cached is not None else calculate_lunar_age(birth_info_to_use)

        # 童限拦截：如果当前年龄小于第一大运起运年龄，直接拒答
        all_decadal_ages = persisted_data.get('all_decadal_ages', {})
        if not all_decadal_ages:
            # 尝试获取本命盘并计算所有大运，以取得第一大运起运年龄
            try:
                payload_api = build_ziwei_api_payload(birth_info_to_use)
                natal_api_response = await robust_api_call_with_retry(
                    url=ZIWEI_API_URL,
                    payload=payload_api,
                    timeout=ZIWEI_API_TIMEOUT_SECONDS
                )
                astrolabe_data = natal_api_response.get('data', {}).get('astrolabe', {})
                gender = astrolabe_data.get("gender", "")
                year_gan = astrolabe_data.get('rawDates', {}).get('chineseDate', {}).get('yearly', [None])[0]
                wuxingju_api = astrolabe_data.get('fiveElementsClass', '')
                palaces_list = astrolabe_data.get('palaces', [])
                if gender and year_gan and wuxingju_api:
                    all_decadal_ages = calculate_all_decadal_periods(
                        birth_info_to_use.get('year'),
                        gender,
                        year_gan,
                        wuxingju_api,
                        palaces_list
                    )
                    if all_decadal_ages:
                        persisted_data['all_decadal_ages'] = all_decadal_ages
            except Exception as e:
                logger.warning(f"获取本命盘用于童限校验失败，跳过童限校验: {e}")

        if all_decadal_ages:
            try:
                first_decadal_start_age = min(age_range[0] for age_range in all_decadal_ages.values() if age_range)
                if current_age < first_decadal_start_age:
                    log_step(
                        "拒绝：童限阶段（未起运）",
                        request_id=monitor_request_id,
                        status="拒绝",
                        extra_data={
                            "reason": "child_limit_before_first_decadal",
                            "current_age": current_age,
                            "first_decadal_start_age": first_decadal_start_age,
                            "intent_type": "single_decadal_analysis",
                            "session_id": session_id
                        }
                    )
                    return """在紫微斗数中，孩子从出生到正式"起运"的这段时期，称为 "童限"。在此阶段，孩子的运势与【父母宫】及【命宫】关联极为密切，通常不独立行大运。"""
            except Exception as e:
                logger.warning(f"计算童限起运年龄失败，跳过童限校验: {e}")

        # 优先使用用户点名的大运干支（如果有）
        explicit_decadal_ganzhi = query_intent_data.get("explicit_decadal_ganzhi")
        decadal_year = None

        if explicit_decadal_ganzhi and all_decadal_ages:
            age_range = all_decadal_ages.get(explicit_decadal_ganzhi)
            if age_range:
                # 根据起运年龄反推起运年份
                decadal_year = birth_year + age_range[0] - 1
                logger.info(
                    f"✅ 用户点名大运干支【{explicit_decadal_ganzhi}】，使用对应大运起运年份: {decadal_year}"
                )
            else:
                logger.warning(
                    f"⚠️ 用户点名的大运干支 {explicit_decadal_ganzhi} 在 all_decadal_ages 中不存在，将回退到自动计算当前/下一个大运逻辑。"
                )

        if decadal_year is None:
            time_indicator_str = str(time_indicator).lower() if time_indicator else ""
            is_current_decadal = any(keyword in time_indicator_str
                                     for keyword in ['当前', '现在', '目前', '这个', '这一', '本'])

            if is_current_decadal:
                # 用户询问当前大运
                decadal_year = get_current_decadal_start_year(birth_year, wuxingju, current_age)
                logger.info(
                    f"✅ 用户询问【当前大运】，当前年龄: {current_age}岁，计算出当前大运开始年份为: {decadal_year}")
            else:
                # 用户询问下一个大运（原有逻辑）
                decadal_year = calculate_next_decadal_start_year(birth_year, wuxingju, current_age)
                logger.info(
                    f"✅ 用户询问【下一个大运】，当前年龄: {current_age}岁，计算出下一个大运开始年份为: {decadal_year}")

        horoscope_date_for_api = f"{decadal_year}-06-01 12:00:00"
        query_intent_data['resolved_horoscope_date'] = horoscope_date_for_api
        query_intent_data['perform_new_analysis'] = True
        return None  # 继续后续流程
    else:
        logger.warning("缺少五行局或出生年份信息，无法计算大运。")
        log_step(
            "拒绝：缺少五行局信息",
            request_id=monitor_request_id,
            status="拒绝",
            extra_data={"reason": "缺少五行局或出生年份信息", "intent_type": "single_decadal_analysis", "session_id": session_id}
        )
        return "无法自动计算大运，因为缺少必要的命盘信息（五行局）。请先进行一次本命盘分析。"


async def handle_special_intent_types(
    intent_type: str,
    prompt_input: str,
    monitor_request_id: str,
    session_id: str
) -> Optional[str]:
    """处理特殊意图类型，返回响应文本或None（继续后续流程）"""
    from prompt_logic import response_string_capability
    
    if intent_type == "knowledge_question":
        log_step(
            "知识问答：概念解释",
            request_id=monitor_request_id,
            status="成功",
            extra_data={"reason": "知识类问题", "intent_type": "knowledge_question", "session_id": session_id}
        )
        return f"""关于您询问的"{prompt_input}"：

这是紫微斗数中的重要概念。由于这是专业知识询问，建议您：
1. 查阅专业的紫微斗数书籍
2. 咨询专业的紫微斗数老师
3. 结合您的命盘进行实际理解

如果您想了解这些概念在您命盘中的具体表现，请提供您的出生信息，我可以为您进行个性化的运势分析。"""

    elif intent_type == "irrelevant_question":
        log_step(
            "拒绝：无关问题",
            request_id=monitor_request_id,
            status="拒绝",
            extra_data={"reason": "无关问题", "intent_type": "irrelevant_question", "session_id": session_id}
        )
        return "抱歉，您的问题我无法回答。我是一个专注于紫微斗数运势分析的助手。关于天气、设备维修、常识问答等日常问题，超出了我的能力范围。请提出与您个人运势相关的问题。"

    elif intent_type == "sensitive_topic_refusal":
        log_step(
            "拒绝：敏感话题",
            request_id=monitor_request_id,
            status="拒绝",
            extra_data={"reason": "敏感话题拒绝", "intent_type": "sensitive_topic_refusal", "session_id": session_id}
        )
        return "抱歉，您的问题我无法回答。根据相关规定，我无法讨论或分析任何与政治、军事相关的话题。我的能力范围仅限于基于紫微斗数的个人运势分析。"

    elif intent_type == "naming_question":
        log_step(
            "拒绝：命名问题",
            request_id=monitor_request_id,
            status="拒绝",
            extra_data={"reason": "功能开发中", "intent_type": "naming_question", "session_id": session_id}
        )
        return "当前功能正在开发中，敬请期待。"

    elif intent_type == "capability_inquiry":
        log_step(
            "能力询问：返回能力说明",
            request_id=monitor_request_id,
            status="成功",
            extra_data={"reason": "能力询问", "intent_type": "capability_inquiry", "session_id": session_id}
        )
        return response_string_capability

    elif intent_type == "unpredictable_future":
        log_step(
            "拒绝：不可预测未来",
            request_id=monitor_request_id,
            status="拒绝",
            extra_data={"reason": "不可预测未来", "intent_type": "unpredictable_future", "session_id": session_id}
        )
        return UNPREDICTABLE_FUTURE_MESSAGE

    elif intent_type == "unanswerable_question":
        log_step(
            "拒绝：无法回答的问题",
            request_id=monitor_request_id,
            status="拒绝",
            extra_data={"reason": "无法回答的问题", "intent_type": "unanswerable_question", "session_id": session_id}
        )
        return UNANSWERABLE_PREDICTION_MESSAGE
    
    return None  # 继续后续流程


async def call_ziwei_api_for_analysis(
    birth_info_to_use: Dict[str, Any],
    horoscope_date_for_api: str,
    determined_analysis_level: str,
    monitor_request_id: str,
    session_id: str
) -> Dict[str, Any]:
    """调用外部紫微API进行分析"""
    payload_api = build_ziwei_api_payload(birth_info_to_use, horoscope_date_for_api)

    logger.info(
        f"即将调用外部紫微API，最终Payload: {json.dumps(payload_api, indent=2, ensure_ascii=False)}")

    try:
        # ⏱️ 运势API调用时间点
        horoscope_api_start = time.time()
        with StepMonitor(
            "成功调用ziweiapi",
            request_id=monitor_request_id,
            extra_data={
                "scene": "analysis",
                "analysis_level": determined_analysis_level
            },
        ):
            api_response_data = await robust_api_call_with_retry(
                url=ZIWEI_API_URL,
                payload=payload_api,
                timeout=ZIWEI_API_TIMEOUT_SECONDS
            )
        horoscope_api_time = time.time() - horoscope_api_start
        logger.info(f"⏱️ 运势API调用耗时: {horoscope_api_time:.2f}秒")

        logger.info(
            f"外部紫微API原始响应结构: {list(api_response_data.get('data', {}).keys()) if api_response_data.get('data') else 'No data field'}")
        logger.info(f"外部紫微API响应状态: {api_response_data.get('status', 'unknown')}")
        logger.info(f"外部紫微API响应错误信息: {api_response_data.get('error', 'none')}")

        # 在调用分析函数之前，先检查API响应的基本结构
        if not api_response_data:
            logger.error(f"外部紫微API返回空响应")
            log_step(
                "错误：外部紫微API返回空响应",
                request_id=monitor_request_id,
                status="失败",
                extra_data={"reason": "外部紫微API返回空响应", "status_code": 500, "session_id": session_id}
            )
            raise HTTPException(500, "外部紫微API返回空响应，请检查API服务状态")

        if not api_response_data.get("data"):
            logger.error(
                f"外部紫微API返回缺少data字段。完整响应: {json.dumps(api_response_data, ensure_ascii=False, indent=2)}")
            log_step(
                "错误：外部紫微API返回数据格式不正确",
                request_id=monitor_request_id,
                status="失败",
                extra_data={"reason": "外部紫微API返回缺少data字段", "status_code": 500, "session_id": session_id}
            )
            raise HTTPException(500, "外部紫微API返回的数据格式不正确，缺少data字段")

        return api_response_data

    except aiohttp.ClientError as e:
        logger.error(f"外部紫微API请求失败，{payload_api}: {e}", exc_info=True)
        log_step(
            "错误：外部紫微API请求失败",
            request_id=monitor_request_id,
            status="失败",
            extra_data={"reason": "外部紫微API请求失败", "error_type": "aiohttp.ClientError", "error_message": str(e), "status_code": 503, "session_id": session_id}
        )
        raise HTTPException(503, f"外部紫微API请求失败: {e}")
    except Exception as e:
        logger.error(f"调用外部紫微API或处理其响应时发生未知错误: {e}", exc_info=True)
        log_step(
            "错误：调用外部紫微API时发生未知错误",
            request_id=monitor_request_id,
            status="失败",
            extra_data={"reason": "调用外部紫微API或处理其响应时发生未知错误", "error_type": type(e).__name__, "error_message": str(e), "status_code": 500, "session_id": session_id}
        )
        raise HTTPException(500, f"分析服务内部错误: {e}")


def build_ziwei_api_payload(birth_info: Dict[str, Any], horoscope_date: Optional[str] = None) -> Dict[str, Any]:
    """
      构建紫微API调用的payload

      Args:
          birth_info: 包含出生信息的字典，必须包含 year, month, day, hour/minute/traditional_hour_branch, gender
          horoscope_date: 可选的运势日期字符串，格式为 "YYYY-MM-DD HH:MM:SS" 或 "YYYY-MM-DD"
          type: 可选的类型字段，如果不传入则不在payload中包含此字段，让API使用默认值

      Returns:
          构建好的API payload字典
      """
    # 构建出生日期字符串
    birth_date_str = f"{birth_info['year']}-{birth_info['month']:02d}-{birth_info['day']:02d}"

    # 计算时辰索引
    time_index = convert_hour_to_time_index(birth_info)

    # 构建基础payload（使用驼峰命名，匹配API期望的格式）
    payload = {
        "dateStr": birth_date_str,
        "type": "lunar" if birth_info.get("is_lunar") else "solar",
        "timeIndex": time_index,
        "gender": birth_info['gender']
    }

    # 如果有运势日期，直接添加到payload中（不做任何格式转换）
    if horoscope_date:
        payload["horoscopeDate"] = horoscope_date

    return payload


def detect_extreme_age_from_prompt(prompt: str) -> Optional[tuple]:
    """
    从原始提问中粗粒度提取年份并估算年龄，用于在出生日期解析失败前先行拦截超高龄输入。
    返回 (估算年龄, 年份)；找不到返回None。
    """
    # 常见日期格式：YYYY-MM-DD / YYYY/MM/DD / YYYY年MM月
    year_matches = re.findall(r'(\d{4})[年/-]', prompt)
    if not year_matches:
        return None
    try:
        year = int(year_matches[0])
        current_year = datetime.now().year
        age = current_year - year + 1
        return age, year
    except Exception:
        return None


async def process_chat_request(
        request: SignableAPIRequest,
        # http_client: aiohttp.ClientSession,
    monitor_request_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """处理完整的聊天请求流程。"""

    # ⏱️ 添加时间监控
    import time
    start_time = time.time()
    request_start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    monitor_request_id = monitor_request_id or generate_request_id()

    session_id = request.session_id or str(uuid.uuid4())
    prompt_input = request.query

    # 智能标准化用户输入的标点符号（保护出生信息格式）
    normalized_prompt_input = smart_normalize_punctuation(prompt_input)
    logger.info(f"🔧 原始提问: {prompt_input}")
    if normalized_prompt_input != prompt_input:
        logger.info(f"🔧 标准化后: {normalized_prompt_input}")

    # 使用标准化后的输入进行后续处理
    prompt_input = normalized_prompt_input
    app_id = request.appid

    logger.info(f"🚀 [会话: {session_id}] 请求开始时间: {request_start_timestamp}")
    weather_info_json_str = request.weather_info
    weather_data = {}  # 初始化为空字典

    if weather_info_json_str:
        try:
            # 核心修改：仅替换不合法的换行符
            weather_info_fixed = weather_info_json_str.replace('\n', r'\n').replace('\r', r'\r')

            # 尝试将传入的字符串解析为JSON字典
            weather_data = json.loads(weather_info_fixed)
            if not isinstance(weather_data, dict):
                logger.warning(f"解析后的 weather_info 不是一个字典，将被忽略。内容: {weather_data}")
                weather_data = {}
            else:
                logger.info(f"请求中包含了结构化的天气信息: {weather_data}")
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"无法将 weather_info 解析为JSON。原始内容: '{weather_info_json_str}'. 错误: {e}")
            weather_data = {}
    else:
        logger.info("请求中未包含天气信息。")

    logger.info(f"--- [会话: {session_id}] 开始处理请求 ---")

    # 政治敏感内容检测
    if detect_sensitive_political_content(prompt_input):
        log_step(
            "拒绝：政治敏感内容",
            request_id=monitor_request_id,
            status="拒绝",
            extra_data={"reason": "政治敏感内容检测", "session_id": session_id}
        )
        political_response = "抱歉，您的问题我无法回答。根据相关规定，我无法讨论或分析任何与政治、军事相关的话题。我的能力范围仅限于基于紫微斗数的个人运势分析。[DONE]"

        yield political_response
        full_response_to_yield_and_save = political_response
        return

    # 年龄相关问题检测
    if detect_age_inquiry(prompt_input):
        log_step(
            "拒绝：年龄相关问题",
            request_id=monitor_request_id,
            status="拒绝",
            extra_data={"reason": "年龄相关问题检测", "session_id": session_id}
        )
        age_response = """抱歉，您的提问方式我无法准确回答。

目前系统暂不支持基于具体年龄（如"20岁"、"30岁"）的运势查询。

建议您可以这样提问：
- "我未来5年的运势如何？"
- "我未来10年的事业发展怎么样？"
- "我今年的财运如何？"
- "我明年的感情运势会怎样？"

这样的提问方式能让我为您提供更准确的运势分析。[DONE]"""

        yield age_response
        full_response_to_yield_and_save = age_response
        return

    # 乱码检测
    if is_gibberish(prompt_input):
        log_step(
            "拒绝：乱码检测",
            request_id=monitor_request_id,
            status="拒绝",
            extra_data={"reason": "乱码检测", "session_id": session_id}
        )
        guidance_response = """抱歉，您的问题我无法回答。我没能理解您的问题。

为了更好地帮助您，请尝试提出清晰的运势相关问题，例如：
- "我今年的事业运势如何？"
- "我最近一个月的财运怎么样？"
- "我在感情方面会有什么变化吗？"
- "我的综合运势趋势如何？"

请重新描述您的问题，我会为您进行专业的运势分析。[DONE]"""

        yield guidance_response
        full_response_to_yield_and_save = guidance_response
        return

    # 金融/投资/彩票类固定回复（规则前置拦截）
    if detect_finance_or_lottery(prompt_input):
        log_step(
            "拒绝：金融投资彩票固定回复",
            request_id=monitor_request_id,
            status="拒绝",
            extra_data={"reason": "finance_or_lottery", "session_id": session_id}
        )
        finance_response = """您好，我的专长在于通过命理工具为您提供人生趋势的洞察与个人决策的参考，而非金融市场分析，我无法也绝不会提供任何具体的金融投资建议，包括对个股、基金或其他金融产品的买卖建议。理财有风险，投资需谨慎。[DONE]"""

        yield finance_response
        full_response_to_yield_and_save = finance_response
        return

    # 粗粒度年份提取：若从原始问题估算年龄超过120岁，提前拒答
    age_and_year = detect_extreme_age_from_prompt(prompt_input)
    if age_and_year is not None:
        estimated_age, estimated_year = age_and_year
        current_year = datetime.now().year
        lower_year = current_year - 120
        # 超过120岁，或未来出生年份，或早于动态下限（120岁之外）均拒答
        if estimated_age > 120 or estimated_year > current_year or estimated_year < lower_year or estimated_age <= 0:
            log_step(
                "拒绝：粗粒度年龄超出上限",
                request_id=monitor_request_id,
                status="拒绝",
                extra_data={
                    "reason": "age_over_limit_from_prompt",
                    "estimated_age": estimated_age,
                    "estimated_year": estimated_year,
                    "session_id": session_id
                }
            )
            age_limit_response = """您好！感谢您的提问。
紫微斗数是帮助我们洞察生命轨迹的智慧，需要扎根于现存的生命历程，我无法为您提供超出人类极限的结果。
我很乐意根据更真实的情况，为您提供真正有参考价值的解读。[DONE]"""

            yield age_limit_response
            full_response_to_yield_and_save = age_limit_response
            return

    # 重大事件关键词检测（违规词检测）
    is_critical, category = detect_critical_time_selection(prompt_input)
    if is_critical:
        log_step(
            "拒绝：重大事件关键词检测",
            request_id=monitor_request_id,
            status="拒绝",
            extra_data={"reason": "重大事件关键词检测", "category": category, "session_id": session_id}
        )
        refusal_response = f"""抱歉，您的问题我无法回答。关于【{category}】这类重大事项，建议您：

1. 咨询相关领域的专业人士（医生、律师、财务顾问等）
2. 综合考虑实际情况和专业建议
3. 不要仅依赖命理推算做出重大决策

命理分析可以作为参考，但涉及健康、法律、重大财务等事项时，专业判断更为重要。

如果您想了解该时期的整体运势趋势，我可以为您分析。[DONE]"""

        yield refusal_response
        full_response_to_yield_and_save = refusal_response
        return

    # 获取会话历史及持久化数据
    with StepMonitor(
        "成功获取数据库信息",
        request_id=monitor_request_id,
        extra_data={"session_id": session_id},
    ):
        history_manager = await get_session_history(session_id)
        persisted_data = await get_user_analysis_data(session_id)
    current_time_for_llm = datetime.now()

    full_response_to_yield_and_save = ""
    # 持久化数据初始化
    analysis_data_to_persist = {}
    chart_data_full, current_birth_info = {}, {}
    query_intent_data = {}
    user_solar_date_display, user_lunar_date_display, user_chinese_date_display, user_gender_display = '', '', '', ''
    last_horoscope_date_used = None
    last_relevant_palaces = None
    is_yes_no_question = False
    direct_answer_instruction_str = ""
    astrolabe_data = {}
    api_response_data_for_decadal = None  # 保存API响应数据，用于提取大运年龄范围
    YANG_GAN = ['甲', '丙', '戊', '庚', '壬']
    YIN_GAN = ['乙', '丁', '己', '辛', '癸']

    final_score_json_str = None
    score_based_judgment_str = "无量化评分信息。"
    judgment_phrasing_options_list = []

    final_input_tokens = 0
    final_output_tokens = 0

    yes_no_keywords = [
        # 核心判断词
        "能不能", "可不可以", "行不行", "是不是", "会不会", "该不该", "有没有",
        "能", "行", "是", "会", "可以", "该", "有",

        "能...吗", "行...吗", "是...吗", "会...吗", "可以...吗", "该...吗", "有...吗",
        "值得...吗", "适合...吗", "对不对", "好不好", "有没有可能", "可能性大吗",
    ]

    for keyword in yes_no_keywords:
        # 对包含...的特殊模式使用正则匹配
        if '...' in keyword:
            pattern = keyword.replace('...', '.*')
            if re.search(pattern, prompt_input):
                is_yes_no_question = True
                break
        # 普通关键词直接检查
        elif keyword in prompt_input:
            is_yes_no_question = True
            break

    # 如果结尾是单个的“吗”，也算
    if prompt_input.strip().endswith('吗'):
        is_yes_no_question = True

    final_judgment_section_str = ""  # 默认为空
    if is_yes_no_question:
        final_judgment_section_str = """根据你的问题，推演结果如下"""

    try:
        # --- 步骤 1: 从 Redis 加载持久化数据 ---
        chart_data_full = persisted_data.get('chart_data_full', {})
        user_solar_date_display = persisted_data.get('user_solar_date_display', '')
        user_lunar_date_display = persisted_data.get('user_lunar_date_display', '')
        user_chinese_date_display = persisted_data.get('user_chinese_date_display', '')
        user_gender_display = persisted_data.get('user_gender_display', '')
        last_horoscope_date_used = persisted_data.get('last_horoscope_date_used')
        current_birth_info = persisted_data.get('user_birth_info', {})
        last_relevant_palaces = persisted_data.get('last_relevant_palaces')
        logger.info(f"从持久化存储加载数据: last_horoscope_date_used={last_horoscope_date_used}")

        # --- 步骤 2: 提取并处理出生信息 ---
        # ⏱️ 出生信息解析时间点
        parse_extract_start = time.time()

        # 首先尝试解析固定格式
        try:
            fixed_format_birth_info, actual_question = parse_fixed_format_birth_info(prompt_input)
            logger.info(f"[出生信息解析] 原始输入: {prompt_input[:100]}...")
            logger.info(f"[出生信息解析] 解析结果: {fixed_format_birth_info}")
            logger.info(f"[出生信息解析] 实际问题: {actual_question}")
            if fixed_format_birth_info:  # 如果成功解析出固定格式
                logger.info(f"✅ 成功解析固定格式出生信息: {fixed_format_birth_info}")
                logger.info(f"实际问题: {actual_question}")
                extracted_birth_info = fixed_format_birth_info
                # 更新prompt_input为实际问题部分
                prompt_input = actual_question
            else:
                # 如果不是固定格式，回退到原来的LLM解析
                logger.warning(f"⚠️ 未检测到固定格式，使用LLM解析出生信息。原始输入: {prompt_input[:100]}...")
                extracted_birth_info = await extract_birth_info_with_llm(
                    prompt_input,
                    # http_client=http_client
                )
        except Exception as e:
            logger.warning(f"固定格式解析失败: {e}，回退到LLM解析")
            extracted_birth_info = await extract_birth_info_with_llm(
                prompt_input,
                # http_client=http_client
            )

        parse_extract_time = time.time() - parse_extract_start
        logger.info(f"⏱️ 出生信息解析耗时: {parse_extract_time:.2f}秒")

        if extracted_birth_info and not extracted_birth_info.get("error"):
            is_new_complete_birth_info = (extracted_birth_info.get("year") and extracted_birth_info.get(
                "month") and extracted_birth_info.get("day") and (extracted_birth_info.get(
                "hour") is not None or extracted_birth_info.get(
                "traditional_hour_branch")) and extracted_birth_info.get("gender"))
            if is_new_complete_birth_info and extracted_birth_info != current_birth_info:
                logger.info(f"--- [会话: {session_id}] 检测到新的完整出生信息，将重置会话。 ---")
                current_birth_info = extracted_birth_info
                chart_data_full, last_horoscope_date_used = {}, None
                user_solar_date_display, user_lunar_date_display, user_chinese_date_display, user_gender_display = '', '', '', ''
                # 清理历史记录
                if hasattr(history_manager, 'clear'):
                    history_manager.clear()
                else:
                    history_manager.clear()  # for list-based history
            elif not current_birth_info:
                current_birth_info = extracted_birth_info

        birth_info_to_use = current_birth_info
        print("birth_info_to_use", birth_info_to_use)
        if not is_birth_info_complete(birth_info_to_use):
            logger.warning(f"出生信息不完整。")
            log_step(
                "拒绝：出生信息不完整",
                request_id=monitor_request_id,
                status="拒绝",
                extra_data={"reason": "出生信息不完整", "birth_info": birth_info_to_use, "session_id": session_id}
            )
            # response = "sg" + MISSING_BIRTH_INFO_MESSAGE
            response = MISSING_BIRTH_INFO_MESSAGE
            yield response + "[DONE]"
            full_response_to_yield_and_save = response
            return
            raise StopAsyncIteration

        # 【新增】在出生信息验证通过后，标准化模糊时间表述（确保使用实际问题部分）
        # prompt_input_normalized = normalize_vague_time_expressions(prompt_input)
        prompt_input_normalized = prompt_input
        if prompt_input_normalized != prompt_input:
            logger.info(f"[时间表述标准化] 原始问题: {prompt_input} → 标准化后: {prompt_input_normalized}")
            prompt_input = prompt_input_normalized  # 后续意图识别与分析都使用标准化后的问题

        try:
            current_age_cached = calculate_lunar_age(birth_info_to_use)
        except Exception as e:
            logger.warning(f"计算年龄失败，跳过年龄上限校验: {e}")
            current_age_cached = None
        
        age_for_limit = current_age_cached

        if age_for_limit is not None and age_for_limit > 120:
            log_step(
                "拒绝：年龄超出上限",
                request_id=monitor_request_id,
                status="拒绝",
                extra_data={"reason": "age_over_limit", "calculated_age": age_for_limit, "birth_info": birth_info_to_use, "session_id": session_id}
            )
            age_limit_response = """您好！感谢您的提问。
紫微斗数是帮助我们洞察生命轨迹的智慧，需要扎根于现存的生命历程，我无法为您提供超出人类极限的结果。
我很乐意根据更真实的情况，为您提供真正有参考价值的解读。[DONE]"""

            yield age_limit_response
            full_response_to_yield_and_save = age_limit_response
            return

        if not request.skip_intent_check:
            try:
                logger.info(f"[紫薇服务] 开始串行执行时间范围识别和意图识别...")
                # 去除用户输入中的出生年月日信息，只保留问题部分
                from .queryIntent import remove_birth_info_from_query
                cleaned_prompt = remove_birth_info_from_query(prompt_input)
                logger.info(f"[紫薇服务] 原始输入: {prompt_input[:100]}...")
                logger.info(f"[紫薇服务] 清理后的问题: {cleaned_prompt[:100]}...")
                
                # 第一步：先执行时间范围识别（串行）
                logger.info(f"[紫薇服务] 第一步：执行时间范围识别...")
                logger.info(f"[紫薇服务] 传入的当前时间: {current_time_for_llm.strftime('%Y-%m-%d %H:%M:%S')}")
                try:
                    time_range_result = await detect_time_range_with_llm(cleaned_prompt, current_time=current_time_for_llm)
                except Exception as e:
                    logger.error(f"[紫薇服务] 时间范围识别失败: {e}", exc_info=True)
                    time_range_result = {
                        "has_time_range": False,
                        "end_date": None,
                        "time_expression": None,
                        "reason": "时间范围识别失败",
                        "time_span_type": "long_term"  # 识别失败时默认视为长时间
                    }
                
                # 提取时间跨度类型（如果为None则默认为long_term）
                time_span_type = time_range_result.get("time_span_type") if isinstance(time_range_result, dict) else "long_term"
                if time_span_type is None:
                    time_span_type = "long_term"
                logger.info(f"[紫薇服务] 时间范围识别完成，time_span_type: {time_span_type}")
                
                # 第二步：执行意图分类，传入时间跨度类型（串行）
                logger.info(f"[紫薇服务] 第二步：执行意图分类（time_span_type={time_span_type}）...")
                try:
                    query_intent_result = await classify_query_intent_with_llm(cleaned_prompt, time_span_type=time_span_type)
                except Exception as e:
                    logger.error(f"[紫薇服务] 意图识别失败: {e}", exc_info=True)
                    raise e
                
                # 处理时间范围识别结果（时间表达替换）
                if time_range_result and isinstance(time_range_result, dict):
                    # 处理时间表达替换：如果包含"年内"、"上半年"、"下半年"、"年后"，从end_date提取年份并替换
                    if time_range_result.get("has_time_range") and time_range_result.get("end_date") and time_range_result.get("time_expression"):
                        time_expression = time_range_result.get("time_expression", "")
                        end_date = time_range_result.get("end_date")
                        force_replace_keywords = ["年内", "上半年", "下半年", "年后"]
                        
                        if any(keyword in time_expression for keyword in force_replace_keywords):
                            try:
                                # 从end_date提取年份
                                if isinstance(end_date, str):
                                    date_obj = datetime.strptime(end_date, "%Y-%m-%d")
                                else:
                                    date_obj = end_date
                                year = date_obj.year
                                
                                # 替换时间表达
                                new_time_expression = time_expression
                                if "年内" in new_time_expression:
                                    new_time_expression = f"{year}年"
                                elif "上半年" in new_time_expression:
                                    new_time_expression = f"{year}年上半年"
                                elif "下半年" in new_time_expression:
                                    new_time_expression = f"{year}年下半年"
                                elif "年后" in new_time_expression:
                                    new_time_expression = f"{year}年"
                                
                                if new_time_expression != time_expression:
                                    logger.info(f"[紫薇服务] 时间表达替换: '{time_expression}' -> '{new_time_expression}'")
                                    time_range_result["time_expression"] = new_time_expression
                            except Exception as e:
                                logger.warning(f"[紫薇服务] 时间表达替换失败: {e}")
                    # 注意：识别成功时保留原始结果，不要重置为失败状态
                
                logger.info(f"[紫薇服务] 意图识别结果: {query_intent_result}")
                logger.info(f"[紫薇服务] 时间范围识别结果: {time_range_result}")
                
                # 优先检查是否为历史时间（硬代码判断）
                if time_range_result.get("has_time_range") and time_range_result.get("end_date"):
                    try:
                        end_date_str = time_range_result.get("end_date")
                        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
                        current_date = current_time_for_llm.date()
                        
                        # 如果结束日期小于当前日期，则为历史时间
                        if end_date < current_date:
                            logger.info(f"[紫薇服务] 检测到历史时间查询，结束日期: {end_date_str}, 当前日期: {current_date}")
                            log_step(
                                "拒绝：历史时间查询",
                                request_id=monitor_request_id,
                                status="拒绝",
                                extra_data={
                                    "reason": "历史时间查询",
                                    "end_date": end_date_str,
                                    "current_date": str(current_date),
                                    "time_range": time_range_result,
                                    "session_id": session_id
                                }
                            )
                            historical_response = "抱歉，您的问题我无法回答。我专注于未来运势的推演和指导。对于已经发生的历史事件，我无法进行回溯性分析。如果您想了解未来的运势趋势，请提出关于未来时间的问题。[DONE]"
                            
                            yield historical_response
                            full_response_to_yield_and_save = historical_response
                            return
                    except (ValueError, TypeError) as e:
                        logger.warning(f"[紫薇服务] 解析结束日期失败: {e}, 继续处理")

                intent_type = query_intent_result.get("query_type")
                intent_reason = query_intent_result.get("reason", "")


                if intent_type == "illegal_content":
                    logger.info("[紫薇服务] 检测到违法犯罪内容，拒绝服务")
                    log_step(
                        "拒绝：违法内容",
                        request_id=monitor_request_id,
                        status="拒绝",
                        extra_data={"reason": "违法内容检测", "intent_type": intent_type, "session_id": session_id}
                    )
                    illegal_response = """抱歉，您的问题我无法回答。[DONE]"""

                    yield illegal_response
                    full_response_to_yield_and_save = illegal_response
                    return

                elif intent_type == "general_knowledge":
                    logger.info("[紫薇服务] 检测到常识性知识问题，拒绝服务")
                    log_step(
                        "拒绝：常识性知识问题",
                        request_id=monitor_request_id,
                        status="拒绝",
                        extra_data={"reason": "常识性知识问题", "intent_type": intent_type, "session_id": session_id}
                    )
                    knowledge_response = """抱歉，您的问题我无法回答。我是专注于紫微斗数和命理运势分析的AI助手，您提出的问题超出了我的服务范围。

我的专长领域包括：
- 紫微斗数命盘分析
- 长期运势趋势解读
- 命理专业知识解答

如果您有关于个人运势、命理方面的问题，欢迎随时向我咨询。[DONE]"""

                    yield knowledge_response
                    full_response_to_yield_and_save = knowledge_response
                    return

                elif intent_type == "qimen":
                    logger.info("[紫薇服务] 检测到奇门遁甲相关问题，建议使用奇门服务")
                    log_step(
                        "建议：奇门遁甲服务",
                        request_id=monitor_request_id,
                        status="建议",
                        extra_data={"reason": "奇门遁甲相关问题", "intent_type": intent_type, "session_id": session_id}
                    )
                    qimen_response = f"""{QIMEN_REFUSAL_MESSAGE}
如果您希望针对「{prompt_input}」进行奇门遁甲解读，请在奇门服务中提问。[DONE]"""

                    yield qimen_response
                    full_response_to_yield_and_save = qimen_response
                    return

                # 处理非紫微体系/跨体系专业问答
                elif intent_type == "non_ziwei_system":
                    logger.info("[紫薇服务] 检测到非紫微体系专业问答，固定拒答")
                    log_step(
                        "拒绝：非紫微体系专业问答",
                        request_id=monitor_request_id,
                        status="拒绝",
                        extra_data={"reason": "non_ziwei_system", "intent_type": intent_type, "session_id": session_id}
                    )
                    non_ziwei_response = """抱歉，您的问题我无法回答。我是专注于命理运势分析的AI助手，您提出的问题超出了我的服务范围。

我的专长领域包括：
- 紫微斗数命盘分析
- 长期运势趋势解读
- 命理专业知识解答

如果您有关于个人运势、命理方面的问题，欢迎随时向我咨询。[DONE]"""

                    yield non_ziwei_response
                    full_response_to_yield_and_save = non_ziwei_response
                    return

                # 处理知识问答类
                elif intent_type == "knowledge_question":
                    logger.info("[紫薇服务] 检测到知识问答类请求，直接回答")
                    try:
                        knowledge_answer = await answer_knowledge_question(prompt_input)
                        log_step(
                            "知识问答：直接回答",
                            request_id=monitor_request_id,
                            status="成功",
                            extra_data={"reason": "知识问答类请求", "intent_type": intent_type, "session_id": session_id}
                        )
                        knowledge_response = f"{knowledge_answer}\n\n如果您想了解这些概念在您命盘中的具体表现，请结合您的出生信息提出更具体的运势问题。[DONE]"

                        yield knowledge_response
                        full_response_to_yield_and_save = knowledge_response
                        return
                    except Exception as e:
                        logger.error(f"[紫薇服务] 知识问答失败: {e}", exc_info=True)
                        # 如果失败，继续走紫薇分析流程
                        logger.info("[紫薇服务] 知识问答失败，继续紫薇分析流程")

                # 助手自我介绍/功能说明类问题：直接返回固定自我介绍文案
                elif intent_type == "self_intro":
                    logger.info("[紫薇服务] 检测到自我介绍/功能说明类问题，直接返回自我介绍")
                    log_step(
                        "自我介绍：固定文案回复",
                        request_id=monitor_request_id,
                        status="成功",
                        extra_data={"reason": "self_intro", "intent_type": intent_type, "session_id": session_id}
                    )
                    intro_response = "我是您的AI生活小助手，集传统文化智慧与现代AI技术于一体，为您提供传统万年历解读、每日运势宜忌及日常养生指南。让千年智慧融入您的生活，在虚实之间揭开未来的迷雾。[DONE]"

                    yield intro_response
                    full_response_to_yield_and_save = intro_response
                    return

                # 判断是否更适合智慧卡
                elif intent_type == "specific_short_term":
                    logger.info(f"[紫薇服务] 检测到问题更适合智慧卡: {intent_type}")
                    log_step(
                        "建议：智慧卡服务",
                        request_id=monitor_request_id,
                        status="建议",
                        extra_data={"reason": "更适合智慧卡", "intent_type": intent_type, "intent_reason": intent_reason, "session_id": session_id}
                    )
                    leipai_suggestion = f"""鉴于您的问题「{prompt_input}」更侧重于{intent_reason}我们建议您使用智慧卡服务。

智慧卡特别适合：
- 了解某个时期的整体运势趋势（几周到一个月内）
- 分析事件发展的方向和结果
- 提供决策建议和行动指引

紫微斗数则更擅长：
- 长期运势趋势分析
- 个人性格和命运特质解读
- 整体人生格局的把握

如果您仍希望从紫微斗数的角度了解相关运势趋势，请继续提问。

如果您希望继续询问该问题请转到智慧卡提问：{prompt_input}
[DONE]"""

                    yield leipai_suggestion
                    full_response_to_yield_and_save = leipai_suggestion
                    return

            except Exception as e:
                logger.warning(f"[紫薇服务] 意图识别失败，继续紫薇分析流程: {e}")
                # 如果意图识别失败，继续走原有的紫薇分析流程
        else:
            logger.info(f"[紫薇服务] skip_intent_check=True，跳过智慧卡意图判断，直接进行紫薇分析")

        year = birth_info_to_use.get('year')
        month = birth_info_to_use.get('month')
        day = birth_info_to_use.get('day')
        gender = birth_info_to_use.get('gender')

        sensitive_birth_dates = [
            (1949, 10, 1, '男'),
            # (1893, 12, 26, '男'),  # 举例：其他需要屏蔽的日期
            # ...可以继续添加其他需要屏蔽的组合
        ]

        is_sensitive_birth_info = False
        for s_year, s_month, s_day, s_gender in sensitive_birth_dates:
            if (year == s_year and
                    month == s_month and
                    day == s_day and
                    (s_gender is None or gender == s_gender)):
                is_sensitive_birth_info = True
                break

        if is_sensitive_birth_info:
            logger.warning(f"检测到敏感出生信息，拒绝分析: {json.dumps(birth_info_to_use, ensure_ascii=False)}")
            log_step(
                "拒绝：敏感出生信息",
                request_id=monitor_request_id,
                status="拒绝",
                extra_data={"reason": "敏感出生信息", "birth_info": birth_info_to_use, "session_id": session_id}
            )
            response = "抱歉，您的问题我无法回答。根据相关规定，我无法对涉及特定公众人物或敏感日期的信息进行命理分析。"

            yield response + "[DONE]"
            full_response_to_yield_and_save = response
            return

        if request.skip_intent_check:
            logger.info(f"[会话: {session_id}] 检测到skip_intent_check=True")
            
            # 在 skip_intent_check 分支中，也需要进行时间范围检测以供查询类型分类使用
            if 'time_range_result' not in locals():
                try:
                    from .queryIntent import remove_birth_info_from_query
                    cleaned_prompt = remove_birth_info_from_query(prompt_input)
                    logger.info(f"[紫薇服务] skip_intent_check分支：传入的当前时间: {current_time_for_llm.strftime('%Y-%m-%d %H:%M:%S')}")
                    time_range_result = await detect_time_range_with_llm(cleaned_prompt, current_time=current_time_for_llm)
                    logger.info(f"[紫薇服务] skip_intent_check分支中时间范围识别结果: {time_range_result}")
                    
                    # 处理时间表达替换：如果包含"年内"、"上半年"、"下半年"、"年后"，从end_date提取年份并替换
                    if time_range_result and time_range_result.get("has_time_range") and time_range_result.get("end_date") and time_range_result.get("time_expression"):
                        time_expression = time_range_result.get("time_expression", "")
                        end_date = time_range_result.get("end_date")
                        force_replace_keywords = ["年内", "上半年", "下半年", "年后"]
                        
                        if any(keyword in time_expression for keyword in force_replace_keywords):
                            try:
                                # 从end_date提取年份
                                if isinstance(end_date, str):
                                    date_obj = datetime.strptime(end_date, "%Y-%m-%d")
                                else:
                                    date_obj = end_date
                                year = date_obj.year
                                
                                # 替换时间表达
                                new_time_expression = time_expression
                                if "年内" in new_time_expression:
                                    new_time_expression = f"{year}年"
                                elif "上半年" in new_time_expression:
                                    new_time_expression = f"{year}年上半年"
                                elif "下半年" in new_time_expression:
                                    new_time_expression = f"{year}年下半年"
                                elif "年后" in new_time_expression:
                                    new_time_expression = f"{year}年"
                                
                                if new_time_expression != time_expression:
                                    logger.info(f"[紫薇服务] 时间表达替换: '{time_expression}' -> '{new_time_expression}'")
                                    time_range_result["time_expression"] = new_time_expression
                            except Exception as e:
                                logger.warning(f"[紫薇服务] 时间表达替换失败: {e}")
                
                except Exception as e:
                    logger.warning(f"[紫薇服务] skip_intent_check分支中时间范围识别失败: {e}")
                    time_range_result = {
                        "has_time_range": False,
                        "end_date": None,
                        "time_expression": None,
                        "reason": "时间范围识别失败",
                        "time_span_type": "long_term"  # 识别失败时默认视为长时间
                    }

            suggested_intent = None
            if hasattr(request, 'summary_intent_type') and request.summary_intent_type:
                logger.info(
                    f"[会话: {session_id}] Summary意图: {request.summary_intent_type}, 理由: {getattr(request, 'summary_intent_reason', '未提供')}")

                # 映射Summary意图到紫薇意图
                SUMMARY_TO_ZIWEI_INTENT_MAPPING = {
                    "general_long_term": "horoscope_analysis",  # 长期解读 → 运势分析
                    "knowledge_question": "knowledge_question",  # 知识问题 → 知识问题
                    "specific_short_term": "horoscope_analysis",  # 短期事件 → 运势分析（如果到了紫薇）
                    "illegal_content": "sensitive_topic_refusal",  # 违法内容 → 敏感话题
                    "qimen": "horoscope_analysis",  # 奇门问题默认走运势分析以保持流程
                }

                suggested_intent = SUMMARY_TO_ZIWEI_INTENT_MAPPING.get(
                    request.summary_intent_type,
                    "horoscope_analysis"  # 默认值
                )
                logger.info(f"[会话: {session_id}] 映射后的紫薇意图: {suggested_intent}")
            else:
                suggested_intent = "horoscope_analysis"
                logger.info(f"[会话: {session_id}] 未收到Summary意图信息，使用默认意图: {suggested_intent}")

            # 使用通用的推演周期提取方法
            time_range_info = time_range_result if 'time_range_result' in locals() else None
            query_intent_data = await extract_query_intent_with_time_analysis(
                prompt_input,
                current_time_for_llm,
                time_range_info,
                suggested_intent,
                last_horoscope_date_used,
                last_relevant_palaces,
                session_id,
                max_business_retries=3,
                enable_intent_mapping=True
            )
        else:
            # 正常流程：使用通用的推演周期提取方法
            time_range_info = time_range_result if 'time_range_result' in locals() else None
            query_intent_data = await extract_query_intent_with_time_analysis(
                prompt_input,
                current_time_for_llm,
                time_range_info,
                None,  # 不跳过分支没有suggested_intent
                last_horoscope_date_used,
                last_relevant_palaces,
                session_id,
                max_business_retries=3,
                enable_intent_mapping=False  # 不跳过分支不需要意图映射
            )

        if not query_intent_data:
            log_step(
                "拒绝：意图数据为空",
                request_id=monitor_request_id,
                status="拒绝",
                extra_data={"reason": "无法理解意图", "session_id": session_id}
            )
            response = "抱歉，您的问题我无法回答。无法理解您的意图，请换一种方式提问。"

            yield response + "[DONE]"
            full_response_to_yield_and_save = response
            return

        resolved_date_str = query_intent_data.get('resolved_horoscope_date')
        relative_time_indicator = query_intent_data.get('relative_time_indicator')
        final_horoscope_date = None  # 初始化为 None

        # 【关键修复】历史检查已在意图分类之前完成（第1123行，基于time_range_result）
        # 如果skip_intent_check=True，不需要再进行历史检查（因为已经跳过了意图识别，说明是直接分析的）
        # 因此这里不再进行历史日期检查

        # 处理特殊意图类型
        special_response = await handle_special_intent_types(
            query_intent_data.get("intent_type"),
            prompt_input,
            monitor_request_id,
            session_id
        )
        if special_response:
            yield special_response + "[DONE]"
            full_response_to_yield_and_save = special_response
            return

        if isinstance(resolved_date_str, str) and resolved_date_str.strip():
            logger.info(f"开始处理从LLM获取的 resolved_horoscope_date: '{resolved_date_str}'")

            possible_formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M',
                '%Y-%m-%d',
                # 可以根据需要添加更多格式, 例如 '%Y/%m/%d'
            ]

            parsed_successfully = False
            for fmt in possible_formats:
                try:
                    # 尝试用当前格式解析
                    dt_obj = datetime.strptime(resolved_date_str, fmt)
                    # 如果成功，格式化为标准的 YYYY-MM-DD HH:MM:SS 格式并赋值
                    final_horoscope_date = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
                    logger.info(
                        f"成功解析日期: '{resolved_date_str}' -> '{final_horoscope_date}' (使用格式: {fmt})")
                    parsed_successfully = True
                    break  # 只要有一种格式解析成功，就跳出循环
                except ValueError:
                    # 如果当前格式不匹配，继续尝试下一种格式
                    continue

            if not parsed_successfully:

                logger.warning(
                    f"无法将 resolved_horoscope_date ('{resolved_date_str}') 解析为任何已知日期格式。将此值设置为 None。"
                )
                final_horoscope_date = None  # 确保这里是 None

        else:
            # 如果 resolved_date_str 本身就是 None, 空字符串, 或非字符串类型
            logger.info(f"resolved_horoscope_date 无效 (值: {resolved_date_str})，将此值设置为 None。")
            final_horoscope_date = None

        query_intent_data["resolved_horoscope_date"] = final_horoscope_date

        print("query_intent_data", query_intent_data, "birth_info_to_use", birth_info_to_use)

        # 检查是否为多流年/多流月/多大运类型（两种模式统一处理）
        multi_time_check_result = await handle_multi_time_analysis_check(
            prompt_input,
            current_time_for_llm,
            birth_info_to_use,
            persisted_data,
            current_age_cached,
            session_id,
            monitor_request_id
        )
        if multi_time_check_result:
            should_stop, response_text = multi_time_check_result
            if should_stop:
                yield response_text + "[DONE]"
                full_response_to_yield_and_save = response_text
                return

        # --- 4. 决策是否调用外部API进行分析 ---
        perform_new_analysis = False
        if query_intent_data.get("intent_type") in ["birth_chart_analysis", "horoscope_analysis"]:
            perform_new_analysis = True
        elif query_intent_data.get("intent_type") == "general_question" and not chart_data_full:
            # 如果是通用问题且没有历史命盘数据，则也进行一次命盘分析
            perform_new_analysis = True

        horoscope_date_for_api = None


        custom_analysis_scope_set = False
        custom_analysis_time_scope_str = None

        time_indicator = query_intent_data.get("relative_time_indicator") or ""
        current_year = datetime.now().year

  
        def is_valid_time_indicator(indicator):
            """检查时间指示器是否有效"""
            if not indicator:
                return False
            if isinstance(indicator, str):
                indicator_lower = indicator.lower().strip()
                return indicator_lower != '' and indicator_lower != 'null'
            return True

        logger.info(f"🔍 完整query_intent_data: {json.dumps(query_intent_data, ensure_ascii=False)}")
        logger.info(f"🔍 time_indicator = '{time_indicator}'")
        logger.info(f"🔍 intent_type = '{query_intent_data.get('intent_type')}'")
        logger.info(f"🔍 multi_decadal_span = '{query_intent_data.get('multi_decadal_span')}'")
        logger.info(f"🔍 multi_yearly_years = '{query_intent_data.get('multi_yearly_years')}'")


        explicit_month_range_match = re.search(
            r'(\d{4})年(\d{1,2})月(\d{1,2})日\s*到\s*(\d{4})年(\d{1,2})月(\d{1,2})日',
            prompt_input
        )
        if explicit_month_range_match:
            try:
                sy, sm, sd, ey, em, ed = map(int, explicit_month_range_match.groups())
                start_date = date(sy, sm, sd)
                end_date = date(ey, em, ed)
                if end_date >= start_date:
                    month_span = (ey - sy) * 12 + (em - sm) + 1
                    if month_span > 1:
                        logger.info(
                            f"检测到显式多月时间范围 {start_date}~{end_date}，月份跨度 {month_span}，强制使用多流月分析"
                        )
                        query_intent_data["intent_type"] = "horoscope_analysis"
                        query_intent_data["analysis_level"] = "monthly"
                        query_intent_data["multi_month_span"] = month_span
                        # 以起始月份的1号中午作为多流月的起点
                        start_month_date = date(sy, sm, 1)
                        query_intent_data["resolved_horoscope_date"] = start_month_date.strftime(
                            "%Y-%m-%d 12:00:00"
                        )
            except Exception as e:
                logger.warning(f"解析显式多月时间范围失败: {e}")


        intent_type = query_intent_data.get("intent_type")
        llm_components = get_analysis_chain()

        # ========== 单个大运分析（当前大运或下一个大运） ==========
        if intent_type == "single_decadal_analysis":
            single_decadal_response = await handle_single_decadal_analysis(
                query_intent_data,
                birth_info_to_use,
                persisted_data,
                current_age_cached,
                time_indicator,
                monitor_request_id,
                session_id
            )
            if single_decadal_response:
                yield single_decadal_response + "[DONE]"
                full_response_to_yield_and_save = single_decadal_response
                return
            # 【关键修复】handle_single_decadal_analysis可能设置了perform_new_analysis，需要同步到局部变量
            if query_intent_data.get('perform_new_analysis'):
                perform_new_analysis = True
            # 【关键修复】handle_single_decadal_analysis已经设置了resolved_horoscope_date，需要获取
            horoscope_date_for_api = query_intent_data.get("resolved_horoscope_date")
        else:
            horoscope_date_for_api = query_intent_data.get("resolved_horoscope_date")

        determined_analysis_level = query_intent_data.get("analysis_level", "general_question")

        if determined_analysis_level == "decadal" and not horoscope_date_for_api and not query_intent_data.get(
                "relative_time_indicator"):
            logger.info(f"检测到大运级别但无时间信息，降级为yearly处理当前流年")
            determined_analysis_level = "yearly"
            query_intent_data["analysis_level"] = "yearly"
            # 使用当前年份
            current_year = datetime.now().year
            horoscope_date_for_api = f"{current_year}-{datetime.now().month:02d}-15 12:00:00"
            query_intent_data["resolved_horoscope_date"] = horoscope_date_for_api

        if perform_new_analysis:
            # 构建外部紫微 API 请求 payload
            horoscope_date_for_api = query_intent_data.get("resolved_horoscope_date")
            if not horoscope_date_for_api:
                current_year = datetime.now().year
                horoscope_date_for_api = f"{current_year}-06-01 12:00:00"
                logger.info(f"LLM未解析出具体日期，使用当前年份作为fallback: {horoscope_date_for_api}")

            try:
                # 调用外部紫微API
                api_response_data = await call_ziwei_api_for_analysis(
                    birth_info_to_use,
                    horoscope_date_for_api,
                    determined_analysis_level,
                    monitor_request_id,
                    session_id
                )

                # 根据分析级别生成分析数据
                if determined_analysis_level == "birth_chart" or query_intent_data.get(
                        "intent_type") == "birth_chart_analysis":
                    analysis_data_result = await generate_ziwei_analysis(birth_info_to_use, api_response_data)
                    last_horoscope_date_used = None
                else:
                    # 对于运势分析，先检查是否有horoscope数据
                    if not api_response_data["data"].get("horoscope"):
                        logger.error(
                            f"外部紫微API返回的响应中缺少horoscope数据。完整响应: {json.dumps(api_response_data, ensure_ascii=False, indent=2)}")
                        log_step(
                            "错误：外部紫微API返回运势数据不完整",
                            request_id=monitor_request_id,
                            status="失败",
                            extra_data={"reason": "外部紫微API返回的响应中缺少horoscope数据", "status_code": 500, "session_id": session_id}
                        )
                        raise HTTPException(500, "外部紫微API返回的运势数据不完整，请检查API服务状态")

                    analysis_data_result, _ = await  generate_horoscope_analysis(birth_info_to_use,
                                                                                 horoscope_date_for_api,
                                                                                 determined_analysis_level,
                                                                                 api_response_data,
                                                                                 query_intent_data,
                                                                                 prompt_input)
                    final_score_json_str = None
                    last_horoscope_date_used = {k: v for k, v in query_intent_data.items() if
                                                k in ["target_year", "target_month", "target_day", "target_hour",
                                                      "target_minute", "relative_time_indicator",
                                                      "resolved_horoscope_date", "analysis_level"]}

                    relevant_palaces = query_intent_data.get("relevant_palaces", [])

                    if final_score_json_str:
                        try:
                            scores = json.loads(final_score_json_str)
                            # 优先使用与意图最相关的宫位的分数
                            target_score = None
                            if relevant_palaces:
                                for palace in relevant_palaces:
                                    if palace in scores:
                                        target_score = scores[palace]
                                        break  # 找到第一个就用

                            # 如果找不到相关宫位分数，就用命宫分数
                            if target_score is None and "命宫" in scores:
                                target_score = scores["命宫"]

                            # 如果连命宫分数都没有，就取平均分
                            if target_score is None and scores:
                                numeric_scores = [s for s in scores.values() if isinstance(s, (int, float))]
                                if numeric_scores:
                                    target_score = sum(numeric_scores) / len(numeric_scores)

                            # 根据最终确定的分数，获取对应的词库
                            judgment_phrasing_options_list = get_judgment_word_bank_for_score(target_score)
                            logger.debug(f"【判断词库】已根据分数 {target_score:.2f} 选择对应层级。")

                        except (json.JSONDecodeError, TypeError) as e:
                            logger.error(f"解析final_score时出错: {e}, 将使用默认中性词库。")
                            judgment_phrasing_options_list = get_judgment_word_bank_for_score(None)
                    else:
                        judgment_phrasing_options_list = get_judgment_word_bank_for_score(None)

                if "error" in analysis_data_result:
                    log_step(
                        "错误：分析数据结果包含错误",
                        request_id=monitor_request_id,
                        status="失败",
                        extra_data={"reason": "分析数据结果包含错误", "error_detail": analysis_data_result["error"], "status_code": 500, "session_id": session_id}
                    )
                    raise HTTPException(status_code=500, detail=analysis_data_result["error"])

                chart_data_full = analysis_data_result
                
                # 记录chart_data_full的结构，用于调试命宫信息提取问题
                logger.info(f"[会话: {session_id}] API返回的chart_data_full键: {list(chart_data_full.keys())[:20]}...")
                if "命宫" in chart_data_full:
                    logger.info(f"[会话: {session_id}] ✅ chart_data_full中包含命宫数据")
                    logger.info(f"[会话: {session_id}] 命宫数据类型: {type(chart_data_full['命宫'])}")
                else:
                    logger.warning(f"[会话: {session_id}] ⚠️ chart_data_full中不包含命宫数据！")
          
                if query_intent_data.get('analysis_level') == 'decadal':
                    api_response_data_for_decadal = api_response_data
                astrolabe_data = api_response_data.get('data', {}).get('astrolabe', {})
                if astrolabe_data:
       
                    gender = astrolabe_data.get("gender", "")
                    # 从 chineseDate.yearly[0] 获取年干
                    year_gan = astrolabe_data.get('rawDates', {}).get('chineseDate', {}).get('yearly', [None])[0]
                    # API响应中的 fiveElementsClass 似乎是五行局
                    wuxingju = astrolabe_data.get('fiveElementsClass', '')
                    palaces_list = astrolabe_data.get('palaces', [])

                    # 确保所有信息都获取到了
                    if gender and year_gan and wuxingju:
                        # 调用修正后的函数
                        all_decadal_ages = calculate_all_decadal_periods(
                            birth_info_to_use.get('year'),
                            gender,
                            year_gan,
                            wuxingju,
                            palaces_list
                        )
                        if all_decadal_ages:
                            persisted_data['all_decadal_ages'] = all_decadal_ages

                # 从API返回的数据中获取显示值（参考代码逻辑）
                user_solar_date_display = astrolabe_data.get('solarDate', '未知')
                user_lunar_date_display = astrolabe_data.get('lunarDate', '未知')
                user_chinese_date_display = astrolabe_data.get('chineseDate', '未知')
                user_gender_display = astrolabe_data.get('gender', '未知')
            except HTTPException:
                # HTTPException 已经在 call_ziwei_api_for_analysis 中处理，直接重新抛出
                raise
            except Exception as e:
                # 处理分析数据生成时的错误
                logger.error(f"生成分析数据时发生错误: {e}", exc_info=True)
                log_step(
                    "错误：生成分析数据时发生错误",
                    request_id=monitor_request_id,
                    status="失败",
                    extra_data={"reason": "生成分析数据时发生错误", "error_type": type(e).__name__, "error_message": str(e), "status_code": 500, "session_id": session_id}
                )
                raise HTTPException(500, f"分析服务内部错误: {e}")

        # --- 5. 准备LLM输入并生成回复 (流式) ---

        # --- 决策逻辑结束 ---
        relevant_palaces = query_intent_data.get("relevant_palaces", [])

        print("relevant_palaces", relevant_palaces)

        is_query_about_other = query_intent_data.get("is_query_about_other_person", False)
        if relevant_palaces:
            seen_palaces = set()
            palaces_ordered = []
            for palace in relevant_palaces:
                if palace in chart_data_full and palace not in seen_palaces:
                    palaces_ordered.append(palace)
                    seen_palaces.add(palace)

            if not is_query_about_other and "命宫" not in seen_palaces and "命宫" in chart_data_full:
                palaces_ordered.insert(0, "命宫")
                seen_palaces.add("命宫")
            elif is_query_about_other and "命宫" in palaces_ordered:
                palaces_ordered = [p for p in palaces_ordered if p != "命宫"]

            if not palaces_ordered:
                fallback_palaces = (
                    ["事业宫", "财帛宫", "夫妻宫", "兄弟宫", "父母宫", "子女宫"]
                    if is_query_about_other else
                    ["命宫", "事业宫", "财帛宫", "夫妻宫", "兄弟宫", "父母宫", "子女宫"]
                )
                palaces_ordered = [p for p in fallback_palaces if p in chart_data_full]

            relevant_analysis_data = {p: chart_data_full.get(p, {}) for p in palaces_ordered}
            logger.info(f"[会话: {session_id}] 具体宫位查询，将分析: {palaces_ordered}")
        else:
            if is_query_about_other:
                default_palaces = ["事业宫", "财帛宫", "夫妻宫"]  # 询问他人时不包含命宫
            else:
                default_palaces = ["命宫", "事业宫", "财帛宫", "夫妻宫", "兄弟宫", "父母宫", "子女宫"]
            relevant_analysis_data = {p: chart_data_full.get(p, {}) for p in default_palaces if p in chart_data_full}
            logger.info(f"[会话: {session_id}] 通用查询，将分析默认宫位: {list(relevant_analysis_data.keys())}")

        # 安全获取命宫信息
        ming_gong_data = chart_data_full.get("命宫", {})
        items_list = list(ming_gong_data.items()) if ming_gong_data else []

        # 2. 通过索引取最后一个（添加安全检查）
        last_key_value_pair = ""
        if items_list and len(items_list) > 0:
            # 尝试多种方式获取命宫信息
            for item in items_list:
                if isinstance(item, tuple) and len(item) > 1:
                    gong_wei_info = item[1]
                    if isinstance(gong_wei_info, dict):
                        # 优先检查本命盘结构：{"原局盘": {"宫位信息": "..."}}
                        if "原局盘" in gong_wei_info and isinstance(gong_wei_info["原局盘"], dict):
                            temp_info = gong_wei_info["原局盘"].get("宫位信息", "")
                            if temp_info:
                                last_key_value_pair = temp_info
                                break
                        # 兼容流年/流月结构：直接包含"宫位信息"
                        elif "宫位信息" in gong_wei_info:
                            temp_info = gong_wei_info["宫位信息"]
                            if temp_info:
                                last_key_value_pair = temp_info
                                break
                        # 兼容其他可能的流年盘结构：如{"2025年流年盘": {"宫位信息": "..."}}
                        else:
                            for key, value in gong_wei_info.items():
                                if isinstance(value, dict) and "宫位信息" in value:
                                    temp_info = value["宫位信息"]
                                    if temp_info:
                                        last_key_value_pair = temp_info
                                        break
                                if last_key_value_pair:
                                    break
                    elif isinstance(gong_wei_info, str) and gong_wei_info.strip():
                        # 如果直接是字符串，使用它
                        last_key_value_pair = gong_wei_info
                        break
                    if last_key_value_pair:
                        break

        # 如果获取失败，记录警告但不设置默认值（空字符串表示没有命宫信息）
        if not last_key_value_pair:
            logger.warning(f"[会话: {session_id}] ⚠️ 无法获取命宫宫位信息！")
            logger.warning(f"[会话: {session_id}] 命宫数据结构详情: {json.dumps(ming_gong_data, ensure_ascii=False, indent=2)}")
            logger.warning(f"[会话: {session_id}] items_list长度: {len(items_list)}")
            logger.warning(f"[会话: {session_id}] chart_data_full的键: {list(chart_data_full.keys())[:10]}...")  # 只显示前10个键
            # 不设置默认值，保持为空字符串，这样在模板中可以通过条件判断是否显示
            last_key_value_pair = ""
        else:
            logger.info(f"[会话: {session_id}] ✅ 成功提取命宫信息，长度: {len(last_key_value_pair)}")

        separator1 = "，这可能意味着"
        result1 = last_key_value_pair.split(separator1, 1)[
            0] if separator1 in last_key_value_pair else last_key_value_pair

        separator2 = "，这些星曜"
        last_key_value_pair = result1.split(separator2, 1)[0] if separator2 in result1 else result1

        separator3 = "。 命宫的三方四正为"
        last_key_value_pair = last_key_value_pair.split(separator3, 1)[
            0] if separator3 in last_key_value_pair else last_key_value_pair

        ANALYSIS_LEVEL_TO_CHINESE_NAME = {
            "hourly": "流时",
            "daily": "流日",
            "monthly": "流月",
            "yearly": "流年",
            "decadal": "大运",  # 或者 "大限"，根据您的术语偏好
            "birth_chart": "原局",  # 或者 "本命盘"

            # 以下为非分析性的级别，可以映射为通用描述
            "general_question": "通用问答",
            "missing_birth_info": "信息不全",
            "unpredictable_future": "无法预测",
            "unanswerable_question": "无法回答",
            "sensitive_topic_refusal": "敏感话题",
            "irrelevant_question": "无关问题"
        }

        chinese_level_name = ANALYSIS_LEVEL_TO_CHINESE_NAME.get(
            query_intent_data.get('analysis_level'),
            '未知分析级别'
        )
        # 注意：analysis_level 只用于内部分析/日志，不再修改命宫文案本身，
        # 避免出现“您的通用问答命宫”这类奇怪表述

        if query_intent_data.get("is_query_about_other_person"):
            # 确保命宫信息明确标注为"您本人"的，避免大模型误以为是询问对象的命宫
            if "您本人" not in last_key_value_pair and "您自己的" not in last_key_value_pair:
                # 如果命宫信息开头是"您的"，替换为"您本人的"
                if last_key_value_pair.startswith("您的"):
                    last_key_value_pair = last_key_value_pair.replace("您的", "您本人的", 1)
                elif not last_key_value_pair.startswith("您本人"):
                    # 如果开头不是"您本人"或"您的"，则添加"您本人的"前缀
                    last_key_value_pair = f"您本人的{last_key_value_pair}"

        if custom_analysis_scope_set and custom_analysis_time_scope_str:
            analysis_time_scope_str = custom_analysis_time_scope_str
            logger.info(f"✅ 使用自定义推演周期: {analysis_time_scope_str}")
        else:
            # 默认推演周期
            analysis_time_scope_str = "针对命主一生的本命盘特征，不涉及具体流年。"

            relative_time_indicator_raw = query_intent_data.get('relative_time_indicator')
            relative_time_indicator = relative_time_indicator_raw if is_valid_time_indicator(
                relative_time_indicator_raw) else None

            if relative_time_indicator:
                # 【修复】只有在非多时段分析且analysis_level为decadal时才使用大运描述
                # 【关键修复】使用API返回的decadal干支和年龄范围，保持与API计算的一致性
                if intent_type not in ('multi_yearly_analysis', 'multi_decadal_analysis') and query_intent_data.get('analysis_level') == 'decadal':
                    # 从chart_data_full中获取API返回的decadal干支
                    decadal_info_raw = chart_data_full.get('_decadal_info', '')
                    decadal_info_gan_zhi = decadal_info_raw.replace('大运', '') if isinstance(decadal_info_raw, str) else ''
                    
                    # 【关键修复】从API响应中提取年龄范围，确保与API返回的decadal干支一致
                    age_range = None
                    if decadal_info_gan_zhi and api_response_data_for_decadal:
                        age_range = extract_decadal_age_range_from_api_response(
                            api_response_data_for_decadal, 
                            decadal_info_gan_zhi
                        )
                        if age_range:
                            logger.info(f"✅ 从API响应中提取到年龄范围: {age_range}, 大运干支: {decadal_info_gan_zhi}")
                    
                    # 如果从API响应中找不到，尝试从all_decadal_ages中查找（作为fallback）
                    if not age_range and decadal_info_gan_zhi:
                        all_decadal_ages = persisted_data.get('all_decadal_ages', {})
                        age_range = all_decadal_ages.get(decadal_info_gan_zhi)
                        if age_range:
                            logger.warning(f"⚠️ 从API响应中未找到年龄范围，使用all_decadal_ages中的值: {age_range}")

                    age_range_str = ""
                    if age_range:
                        age_range_str = f"（{age_range[0]}岁至{age_range[1]}岁）"

                    # 【修复】如果relative_time_indicator无效，在大运场景下使用更自然的描述
                    if relative_time_indicator and decadal_info_gan_zhi:
                        analysis_time_scope_str = f"针对【{relative_time_indicator}】{age_range_str}的{decadal_info_gan_zhi}大运。"
                    elif decadal_info_gan_zhi:
                        # 没有有效时间指示器时，直接使用年龄范围和大运干支
                        analysis_time_scope_str = f"针对{age_range_str}的{decadal_info_gan_zhi}大运。"
                    else:
                        # 如果找不到大运干支，使用默认描述
                        analysis_time_scope_str = f"针对【{relative_time_indicator}】。"

                else:
                    # 如果 decadal_info 为空，则进入更精细的判断
                    level = query_intent_data.get('analysis_level')
                    date_str = query_intent_data.get('resolved_horoscope_date')

                    if date_str:
                        try:
                            dt_obj = datetime.fromisoformat(date_str.replace(' ', 'T'))

                            if level == 'yearly':
                                # 检查原始输入中是否包含"年内"、"上半年"、"下半年"、"年后"
                                original_prompt_has_keywords = any(keyword in prompt_input for keyword in ["年内", "上半年", "下半年", "年后"])
                                
                                # 如果是"年内"、"上半年"、"下半年"、"年后"，使用更明确的描述
                                if relative_time_indicator and "年内" in relative_time_indicator:
                                    current_date = datetime.now()
                                    analysis_time_scope_str = f"针对'年内'，即 {dt_obj.year} 年（从当前日期到年底）的流年紫微命盘进行推演解析。"
                                elif relative_time_indicator and "上半年" in relative_time_indicator:
                                    # 检查是否有年份信息（如"今年上半年"、"明年上半年"）
                                    time_desc = relative_time_indicator if relative_time_indicator else "上半年"
                                    analysis_time_scope_str = f"针对'{time_desc}'，即 {dt_obj.year} 年（1月到6月）的流年紫微命盘进行推演解析。"
                                elif relative_time_indicator and "下半年" in relative_time_indicator:
                                    # 检查是否有年份信息（如"今年下半年"、"明年下半年"）
                                    time_desc = relative_time_indicator if relative_time_indicator else "下半年"
                                    analysis_time_scope_str = f"针对'{time_desc}'，即 {dt_obj.year} 年（7月到12月）的流年紫微命盘进行推演解析。"
                                elif relative_time_indicator and "年后" in relative_time_indicator:
                                    # 【关键修复】"年后"应该显示为具体年份，与其他流年保持一致
                                    analysis_time_scope_str = f"针对'{dt_obj.year}年'的流年紫微命盘进行推演解析。"
                                elif original_prompt_has_keywords and relative_time_indicator and "年" in relative_time_indicator:
                                    # 如果原始输入包含这些关键词，但relative_time_indicator已经被替换为"2025年"这样的格式
                                    # 使用替换后的时间表达
                                    analysis_time_scope_str = f"针对'{relative_time_indicator}'的流年紫微命盘进行推演解析。"
                                else:
                                    analysis_time_scope_str = f"针对'流年'，即 {dt_obj.year} 年的流年紫微命盘进行推演解析。"
                            elif level == 'monthly':
                                # 使用公历月份范围描述，保持与API调用一致
                                analysis_time_scope_str = get_solar_month_range_string(dt_obj) + "的流月紫微命盘进行推演解析"
                            elif level == 'daily':
                                if relative_time_indicator:
                                    analysis_time_scope_str = f"针对'{relative_time_indicator}'，即 {dt_obj.strftime('%Y年%m月%d日')} 的流日紫微命盘进行推演解析。"
                                else:
                                    analysis_time_scope_str = f"针对 {dt_obj.strftime('%Y年%m月%d日')} 的流日紫微命盘进行推演解析。"
                            else:
                                time_indicator_text = relative_time_indicator if relative_time_indicator else "该时段"
                                analysis_time_scope_str = f"针对【{time_indicator_text}】这段时间。"
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"无法解析 resolved_horoscope_date '{date_str}' 来生成时间范围描述: {e}，将使用相对时间。")
                            time_indicator_text = relative_time_indicator if relative_time_indicator else "该时段"
                            analysis_time_scope_str = f"针对【{time_indicator_text}】这段时间。"
                    else:
                        time_indicator_text = relative_time_indicator if relative_time_indicator else "该时段"
                        analysis_time_scope_str = f"针对【{time_indicator_text}】这段时间。"

            # 如果没有相对时间指示器，但有具体的日期
            elif query_intent_data.get('resolved_horoscope_date'):
                level = query_intent_data.get('analysis_level')
                date_str = query_intent_data.get('resolved_horoscope_date')
                
                # 对于大运分析，即使没有relative_time_indicator，也尝试显示大运信息
                if level == 'decadal' and intent_type != 'multi_yearly_analysis':
                    decadal_info_raw = chart_data_full.get('_decadal_info', '')
                    decadal_info_gan_zhi = decadal_info_raw.replace('大运', '') if isinstance(decadal_info_raw, str) else ''
                    
                    age_range = None
                    if decadal_info_gan_zhi and api_response_data_for_decadal:
                        age_range = extract_decadal_age_range_from_api_response(
                            api_response_data_for_decadal, 
                            decadal_info_gan_zhi
                        )
                        if age_range:
                            logger.info(f"✅ 从API响应中提取到年龄范围: {age_range}, 大运干支: {decadal_info_gan_zhi}")
                    
                    # 如果从API响应中找不到，尝试从all_decadal_ages中查找（作为fallback）
                    if not age_range and decadal_info_gan_zhi:
                        all_decadal_ages = persisted_data.get('all_decadal_ages', {})
                        age_range = all_decadal_ages.get(decadal_info_gan_zhi)
                        if age_range:
                            logger.warning(f"⚠️ 从API响应中未找到年龄范围，使用all_decadal_ages中的值: {age_range}")
                    
                    age_range_str = ""
                    if age_range:
                        age_range_str = f"（{age_range[0]}岁至{age_range[1]}岁）"
                    
                    if decadal_info_gan_zhi:
                        analysis_time_scope_str = f"针对{age_range_str}的{decadal_info_gan_zhi}大运。"
                    else:
                        try:
                            dt_obj = datetime.fromisoformat(date_str.replace(' ', 'T'))
                            analysis_time_scope_str = f"针对 {dt_obj.year} 年的大运。"
                        except (ValueError, TypeError):
                            analysis_time_scope_str = "针对大运期间。"
                else:
                    try:
                        dt_obj = datetime.fromisoformat(date_str.replace(' ', 'T'))

                        if level == 'yearly':
                            analysis_time_scope_str = f"针对公历 {dt_obj.year} 年。"
                        elif level == 'monthly':
                            analysis_time_scope_str = get_solar_month_range_string(dt_obj) + "的流月紫微命盘进行推演解析"
                        elif level == 'daily':
                            analysis_time_scope_str = f"针对公历 {dt_obj.strftime('%Y年%m月%d日')}。"
                        # 其他情况（如 hourly）可以继续添加
                        else:
                            analysis_time_scope_str = f"针对公历 {dt_obj.strftime('%Y年%m月%d日 %H时')}。"
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"无法解析 resolved_horoscope_date '{date_str}' 来生成时间范围描述: {e}，将使用默认值。")

        analysis_section_content = ""

        # 定义一个更有逻辑的宫位分析顺序
        palace_analysis_order = [
            "命宫", "福德宫", "事业宫", "财帛宫", "迁移宫",
            "夫妻宫", "子女宫", "田宅宫", "父母宫", "兄弟宫",
            "交友宫", "疾厄宫"
        ]

        available_palaces_in_order = [
            p for p in palace_analysis_order
            if p in relevant_analysis_data and relevant_analysis_data[p]
        ]

        title_map = {
            "命宫": "命宫",
            "福德宫": "福德宫",
            "事业宫": "事业宫",
            "财帛宫": "财帛宫",
            "迁移宫": "迁移宫",
            "夫妻宫": "夫妻宫",
            "子女宫": "子女宫",
            "田宅宫": "田宅宫",
            "父母宫": "父母宫",
            "兄弟宫": "兄弟宫",
            "交友宫": "交友宫",
            "疾厄宫": "疾厄宫",
        }

        # 用于构建星宫推演的摘要列表
        palace_summaries_for_deduction = []
        gongwei_info_list = []
        # 根据真实可用宫位生成模板内容
        section_counter = 1
        for palace in available_palaces_in_order:
            palace_data = relevant_analysis_data[palace]

            first_horoscope_key = next(iter(palace_data), None)

            if first_horoscope_key and isinstance(palace_data[first_horoscope_key], dict):
                horoscope_details = palace_data[first_horoscope_key]

                # 提取数据，并提供默认值以防万一
                gongwei_info = horoscope_details.get("宫位信息", "无宫位信息。").strip()
                key_info = horoscope_details.get("关键信息汇总", "无关键信息。").strip()
                gongwei_info_list.append(palace)
                # 构建标题，确保编号连续
                title_prefix_full = title_map.get(palace, f"{section_counter}. 核心影响：({palace})")
                # 智能地处理编号和标题，避免重复
                if title_prefix_full.startswith(f"{section_counter}."):
                    final_title = f"**{title_prefix_full}**"
                else:
                    title_prefix = title_prefix_full.split(".")[
                        1].strip() if "." in title_prefix_full else title_prefix_full
                    final_title = f"**{section_counter}. {title_prefix}**"

                # 拼接成一个带有上下文的指令块
                analysis_section_content += f"{final_title}\n"
                analysis_section_content += "<!-- 指令：请基于以下【原始数据】，用你自己的语言进行深度解读，提炼核心论点，并提供一个具体的【应验事件】例子。 -->\n"
                analysis_section_content += "<原始数据>\n"
                analysis_section_content += f"  <宫位信息>{gongwei_info}</宫位信息>\n"
                analysis_section_content += f"  <关键信息汇总>{key_info}</关键信息汇总>\n"
                analysis_section_content += "</原始数据>\n\n"

                # 为“星宫推演”收集材料，只收集有实质内容的
                if key_info and key_info != "无关键信息。":
                    palace_summaries_for_deduction.append(f"【{palace}】揭示的趋势: {key_info}")

                section_counter += 1

        # 仅在有多个宫位可供分析时，才添加“星宫推演”部分
        if len(available_palaces_in_order) > 1:
            deduction_context = "\n".join(f"- {s}" for s in palace_summaries_for_deduction)
            analysis_section_content += "**星宫推演：最终显现的事件 (逻辑推演结论)**\n"
            analysis_section_content += "<!-- 指令：请基于以下各宫位的核心趋势，综合推演出一个最终可能显现的事件。 -->\n"
            analysis_section_content += "<各宫核心趋势>\n"
            analysis_section_content += f"{deduction_context}\n"
            analysis_section_content += "</各宫核心趋势>\n\n"

        travel_keywords = ["出行", "出门", "旅游", "出差", "去", "到", "逛"]

        # 判断是否为出行意图
        is_travel_query = (
                bool(weather_data) and  # 条件1: 必须成功解析出天气数据
                any(keyword in prompt_input for keyword in travel_keywords) and
                query_intent_data.get("analysis_level") in ["daily", "hourly"]
        )

        selected_prompt_template = None
        use_multi_period_prompt = False  

        # print("is_travel_query",is_travel_query)

        if is_travel_query:
            logger.info("检测到出行建议意图，使用专属模板。")
            selected_prompt_template = llm_components["travel_advice"]["prompt"]
           
            relevant_palaces = ["迁移宫", "命宫"]
        else:
            # 走原来的通用分析逻辑
            intent_type = query_intent_data.get("intent_type")

            if intent_type == "multi_yearly_analysis":
                logger.info(f"检测到多流年分析意图，使用普通分析模板。")
                use_multi_period_prompt = False
                selected_prompt_template = llm_components["overall_summary"]["prompt"]
                # 获取原有的 relevant_palaces
                relevant_palaces = query_intent_data.get("relevant_palaces", [])
            elif intent_type == "single_decadal_analysis":
                logger.info(f"检测到单个大运分析意图，使用普通分析模板。")
                # 单个大运使用普通分析模板
                selected_prompt_template = llm_components["overall_summary"]["prompt"]
                relevant_palaces = query_intent_data.get("relevant_palaces", [])
            elif intent_type == "missing_birth_info":
                selected_prompt_template = llm_components["missing_birth_info"]["prompt"]
            elif query_intent_data.get("is_query_about_other_person"):
                selected_prompt_template = llm_components["missing_other_person_birth_info"]["prompt"]
            elif intent_type in ["birth_chart_analysis", "horoscope_analysis"]:
                selected_prompt_template = llm_components["overall_summary"]["prompt"]
            else:
                selected_prompt_template = llm_components["general_question"]["prompt"]

            # 获取原有的 relevant_palaces（对于非multi_period的情况）
            if not use_multi_period_prompt:
                relevant_palaces = query_intent_data.get("relevant_palaces", [])


        full_history_messages = []
        try:
            # 统一使用 .messages 属性（支持 async property）
            messages_attr = history_manager.messages
            # 如果是 async property，需要 await
            if hasattr(messages_attr, '__await__'):
                full_history_messages = await messages_attr
            else:
                full_history_messages = messages_attr
        except AttributeError:
            # 兼容旧的列表模式
            if isinstance(history_manager, list):
                full_history_messages = history_manager
            else:
                logger.warning(f"未知的 history_manager 类型: {type(history_manager)}，历史记录将为空。")
                full_history_messages = []

        recent_history = full_history_messages[-6:] if len(full_history_messages) > 6 else full_history_messages
        logger.info(f"[会话: {session_id}] 使用历史对话: {len(recent_history)}条消息")
        # print("recent_history", recent_history)

        question = prompt_input

        # 提供默认值以防 weather_data 为空或某些键不存在
        destination_city = f"{weather_data.get('provinceCn', '')}{weather_data.get('dname', '未知目的地')}" if weather_data else "未知目的地"
        day_condition = weather_data.get('conditionDay', '未知')
        day_temp = weather_data.get('tempDay', '未知')
        day_wind_dir = weather_data.get('windDirDay', '未知')
        day_wind_level = weather_data.get('windLevelDay', '未知')
        night_condition = weather_data.get('conditionNight', '未知')
        night_temp = weather_data.get('tempNight', '未知')
        night_wind_dir = weather_data.get('windDirNight', '未知')
        night_wind_level = weather_data.get('windLevelNight', '未知')


        prompt_kwargs = {
            "question": prompt_input,
            "history": recent_history,
            "full_structured_analysis_data_json": json.dumps(relevant_analysis_data, ensure_ascii=False,
                                                             indent=2),
            "full_chart_data_json": json.dumps(chart_data_full, ensure_ascii=False, indent=2),
            "analysis_time_scope_str": analysis_time_scope_str,
            "user_solar_date_display": user_solar_date_display,
            "user_lunar_date_display": user_lunar_date_display,
            "user_chinese_date_display": user_chinese_date_display,
            "user_gender_display": user_gender_display,

            "score_based_judgment_str": score_based_judgment_str,
            "judgment_phrasing_options_json": json.dumps(judgment_phrasing_options_list, ensure_ascii=False),

            "analysis_section_title": "### 二 命盘推断与解析",
            "analysis_section_content": analysis_section_content,

            "final_judgment_section": final_judgment_section_str,
            "last_key_value_pair": last_key_value_pair if not query_intent_data.get(
                "is_query_about_other_person") else "",  
            # 根据命宫信息是否为空，设置命宫信息行
            "ming_gong_info_line": f"    > ###### 命宫信息：{last_key_value_pair}" if last_key_value_pair and not query_intent_data.get("is_query_about_other_person") else "",
            "destination_city": destination_city,
            "day_condition": day_condition,
            "day_temp": day_temp,
            "day_wind_dir": day_wind_dir,
            "day_wind_level": day_wind_level,
            "night_condition": night_condition,
            "night_temp": night_temp,
            "night_wind_dir": night_wind_dir,
            "night_wind_level": night_wind_level,
            "gongwei_info_list": gongwei_info_list,
            "decadal_followup_info": ""
        }

        log_step(
            "成功构建提示词",
            request_id=monitor_request_id,
            extra_data={
                "intent_type": query_intent_data.get("intent_type"),
                "analysis_level": query_intent_data.get("analysis_level")
            }
        )

        MAX_STREAM_RETRIES = 3
        MIN_LENGTH_TO_START_YIELDING = 30
        llm_output = ""
        generation_succeeded = False

        final_input_tokens = 0
        final_output_tokens = 0
        llm_analysis_start = time.time()  

        for attempt in range(MAX_STREAM_RETRIES):
            logger.info(
                f"--- [会话: {session_id}] 准备生成并校验流式回复 (第 {attempt + 1}/{MAX_STREAM_RETRIES} 次尝试) ---")

            try:
                async with StepMonitor(
                    "成功请求llm",
                    request_id=monitor_request_id,
                    extra_data={"attempt": attempt + 1},
                ):
                    # --- 每次尝试都重新准备 ---
                    current_attempt_response_content = ""
                    buffer = []
                    gate_opened = False

                    if use_multi_period_prompt and not chart_data_full:
                        logger.warning("[多时段分析] 检测到chart_data为空，自动降级为普通模板。")
                        use_multi_period_prompt = False
                        if not selected_prompt_template:
                            selected_prompt_template = llm_components["overall_summary"]["prompt"]
                        relevant_palaces = query_intent_data.get("relevant_palaces", [])

                    # 不再使用多时段专用prompt，统一使用原有的模板方式
                    if use_multi_period_prompt:
                        logger.warning("[多时段分析] use_multi_period_prompt 已废弃，自动降级为普通模板。")
                        use_multi_period_prompt = False
                        if not selected_prompt_template:
                            selected_prompt_template = llm_components["overall_summary"]["prompt"]
                    
                    if False:  # 保留代码结构，但不再执行
                        # 使用多时段专用prompt（已废弃）
                        pass
                    else:
                        # 使用原有的模板方式
                        # 🔧 兜底检查：确保 selected_prompt_template 不为 None
                        if selected_prompt_template is None:
                            logger.warning(f"[会话: {session_id}] selected_prompt_template 为 None，使用默认模板。intent_type={intent_type}")
                            selected_prompt_template = llm_components["overall_summary"]["prompt"]
                        
                        # 填充 Prompt 模板
                        # 模板诊断和格式化
                        template_vars = selected_prompt_template.input_variables
                        filtered_kwargs = {k: v for k, v in prompt_kwargs.items() if k in template_vars}
                        missing_keys = set(template_vars) - set(filtered_kwargs.keys())
                        if missing_keys:
                            raise ValueError(f"模板缺少变量: {missing_keys}")

                        final_messages = selected_prompt_template.format_messages(**filtered_kwargs)
                        # 【修复】LangChain 的 HumanMessage.type 是 "human"，但 OpenAI/DashScope API 需要 "user"
                        def _normalize_role(role: str) -> str:
                            role_map = {"human": "user", "ai": "assistant"}
                            return role_map.get(role, role)
                        vllm_messages = [{"role": _normalize_role(m.type), "content": m.content} for m in final_messages]

                        # 构建 VLLM 请求 payload
                        vllm_payload = {
                            "model": VLLM_MODEL_NAME,
                            "messages": vllm_messages,
                            "temperature": 0.7,
                            "stream": True
                        }

                    logger.info(f"--- [会话: {session_id}] 正在排队等待VLLM资源... ---")
                    # ⏱️ LLM分析生成时间点
                    llm_analysis_start = time.time()
                    stream = aiohttp_vllm_stream(vllm_payload)

                    # --- 免责声明的准备 ---
                    disclaimer_to_yield = ""
                    if query_intent_data.get("is_query_about_other_person"):
                        relationship_label = query_intent_data.get("relationship")
                        search_terms = ["公司", "田", "房", "家", "地", "宅", "单位"]
                        if relationship_label and (any(term in relationship_label for term in search_terms) or "田宅宫" in query_intent_data.get("relevant_palaces") ):
                            disclaimer_to_yield = f"**重要提示：** 本次分析是基于**您的命盘**对您和您**{relationship_label}**进行的运势分析。\n\n"
                        elif relationship_label:
                            disclaimer_to_yield = f"**重要提示：** 本次分析是基于**您的命盘**对您和您**{relationship_label}**进行的运势分析。如需对该关系人进行更准确、深入的独立运势分析，请您更新/添加该关系人的档案，并重新提问。\n\n"
                        else:
                            disclaimer_to_yield = "**重要提示：** 本次分析是基于**您的命盘**对您所提及之人/物的间接解读。如需对该关系人进行更准确、深入的独立运势分析，请您更新/添加该关系人的档案，并重新提问。\n\n"

                    async for chunk_data in stream:

                        # 健壮性检查，确保收到的数据是符合预期的元组
                        if not (isinstance(chunk_data, tuple) and len(chunk_data) == 3):
                            logger.warning(f"Received malformed data from stream: {chunk_data}")
                            continue

                        # 正确解包
                        chunk, in_tokens, out_tokens = chunk_data

                 
                        final_input_tokens = in_tokens
                        final_output_tokens = out_tokens

                        # 检查流是否返回错误信息
                        if chunk.startswith("[错误:"):
                            logger.error(f"Stream returned an error on attempt {attempt + 1}: {chunk}")
                            # 抛出一个异常来触发外层的 except 和重试逻辑
                            raise RuntimeError(f"Stream Error: {chunk}")

                        if not chunk: continue

                        if not gate_opened:
                            buffer.append(chunk)

                            # 累积当前缓冲区的所有内容
                            current_buffered_text = "".join(buffer)
                            # 条件1: 累积的字符数必须达到阈值
                            length_condition_met = len(current_buffered_text) >= MIN_LENGTH_TO_START_YIELDING

                            if len(current_buffered_text) > 200:
                                header_condition_met = True  # 内容足够长，强制打开闸门
                            else:
                                header_condition_met = "### 二" not in current_buffered_text

                            if length_condition_met and header_condition_met:
                                logger.info(f"内容长度达到阈值，打开流式输出闸门。")
                                gate_opened = True

                                # 闸门打开，先发送免责声明
                                if disclaimer_to_yield:
                                    yield disclaimer_to_yield
                                    full_response_to_yield_and_save += disclaimer_to_yield

                                # 再发送缓冲区内容
                                yield current_buffered_text

                                current_attempt_response_content += current_buffered_text
                        else:
                            yield chunk
                            current_attempt_response_content += chunk

                    # --- 单次流结束后的判断 ---
                    if gate_opened:
                        llm_output = current_attempt_response_content
                        generation_succeeded = True
                        logger.info(f"--- 第 {attempt + 1} 次尝试成功，共生成 {len(llm_output)} 字符 ---")
                        logger.info(
                            f"--- 报告生成成功 --- \n"
                            f"输入Tokens: {final_input_tokens}\n"
                            f"输出Tokens: {final_output_tokens}"
                        )

                        break 
                    else:
                        buffered_content = "".join(buffer)
                        # ⚙️ 放宽闸门策略：只要有非空内容，就直接作为本次结果返回，
                        # 避免因为回答过短而被误判为“生成失败”，从而触发多次重试和系统报错。
                        if buffered_content.strip():
                            logger.info(
                                f"第 {attempt + 1} 次尝试：闸门未正式打开，但已收到有效内容 ({len(buffered_content)} 字符)，直接输出作为结果。"
                            )
                            gate_opened = True
                            if disclaimer_to_yield:
                                yield disclaimer_to_yield
                                full_response_to_yield_and_save += disclaimer_to_yield
                            yield buffered_content
                            current_attempt_response_content = buffered_content
                            llm_output = current_attempt_response_content
                            generation_succeeded = True
                            break
                        else:
                            logger.warning(
                                f"第 {attempt + 1} 次尝试：流式内容为空或仅空白字符，准备重试。"
                            )
                            # 不做任何事，循环会自动进入下一次重试

            except Exception as e:
                logger.error(f"--- 在流式生成的第 {attempt + 1} 次尝试中发生错误 ---")
                logger.error(f"错误类型: {type(e).__name__}, 错误信息: {e}")
                # 如果是最后一次尝试，打印更详细的traceback
                if attempt == MAX_STREAM_RETRIES - 1:
                    logger.error(f"详细追溯: \n{traceback.format_exc()}")
                # 让循环继续，进行下一次重试
                continue

        if generation_succeeded:
            log_step(
                "成功收到llm返回信息",
                request_id=monitor_request_id,
                extra_data={"length": len(llm_output)},
            )

        # --- 所有重试结束后的最终处理 ---
        if not generation_succeeded:
            log_step(
                "错误：LLM生成失败",
                request_id=monitor_request_id,
                status="失败",
                extra_data={"reason": "AI多次尝试后仍无法生成有效回复", "max_retries": MAX_STREAM_RETRIES, "session_id": session_id}
            )
            llm_output = "\n[系统提示: AI多次尝试后仍无法生成有效回复，请您调整问题后重试。]"
            # 如果生成失败，也要发送错误消息给客户端
            yield llm_output

        # 累积LLM的输出
        full_response_to_yield_and_save += llm_output

        llm_analysis_time = time.time() - llm_analysis_start
        logger.info(f"⏱️ LLM分析生成耗时: {llm_analysis_time:.2f}秒")

        final_disclaimer = "[DONE]"
        yield final_disclaimer

        full_response_to_yield_and_save += final_disclaimer

        # 调试信息仅用于日志，不添加到响应中（避免泄露给客户端）
        logger.debug(f"完整响应内容长度: {len(full_response_to_yield_and_save)} 字符")
        logger.debug(f"意图信息: {query_intent_data}")
        logger.debug(f"用户问题: {prompt_input}")
        logger.debug(f"出生信息: {birth_info_to_use}")
        debug_info = f"Intent={query_intent_data.get('intent_type')}, BirthInfo Used={birth_info_to_use is not None}"
        logger.debug(f"Debug Info: {debug_info}")


        try:
            await db_manager.upsert_api_usage_stats(session_id, app_id)
            logger.info(f"用量统计已更新: session_id={session_id}, app_id={app_id}")
            logger.debug(f"用量统计已更新: session_id={session_id}, app_id={app_id}")
        except Exception as e:
            # 即使日志记录失败，也不应该中断主流程
            logger.error(f"调用用量统计更新时发生顶层错误: {e}")

    except (HTTPException, aiohttp.ClientError) as e:
        error_message = f"\n[系统提示: 服务暂时不可用，请稍后再试。({e.detail if isinstance(e, HTTPException) else str(e)})]"
        logger.error(f"主流程中捕获到HTTP或网络错误: {e}", exc_info=True)
        # 记录监控日志
        status_code = e.status_code if isinstance(e, HTTPException) else 503
        log_step(
            "错误：HTTP或网络错误",
            request_id=monitor_request_id,
            status="失败",
            extra_data={
                "reason": "HTTP或网络错误",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "status_code": status_code,
                "session_id": session_id
            }
        )
        # yield error_message
        full_response_to_yield_and_save = error_message
    except Exception as e:
        error_message = "\n[系统提示: 发生了一个内部错误，我们正在处理，请您稍后再试。]"
        logger.error(f"在主流程中捕获到未知异常: {e}", exc_info=True)
        logger.error(f"详细追溯: \n{traceback.format_exc()}")
        # 记录监控日志
        log_step(
            "错误：内部异常",
            request_id=monitor_request_id,
            status="失败",
            extra_data={
                "reason": "内部异常",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "session_id": session_id
            }
        )
        # yield error_message
        full_response_to_yield_and_save = error_message



    finally:
        logger.info(f"--- [会话: {session_id}] 函数执行完毕，执行 finally 块 ---")

        try:
            # 准备要持久化的数据
            final_palaces_for_this_round = query_intent_data.get('relevant_palaces')
            wuxingju = astrolabe_data.get('fiveElements', '')

            if not isinstance(final_palaces_for_this_round, list):
                # 如果从意图提取中得到的不是列表，记录一个警告并将其置为None或空列表
                logger.warning(
                    f"意图提取返回的 relevant_palaces 类型不正确: {type(final_palaces_for_this_round)}，将不会持久化。")
                final_palaces_for_this_round = None

            analysis_data_to_persist = {

                "user_birth_info": current_birth_info,

                "chart_data_full": chart_data_full,

                "last_horoscope_date_used": last_horoscope_date_used,

                "user_solar_date_display": user_solar_date_display,

                "user_lunar_date_display": user_lunar_date_display,

                "user_chinese_date_display": user_chinese_date_display,

                "user_gender_display": user_gender_display,
                "last_relevant_palaces": final_palaces_for_this_round,
                "wuxingju": wuxingju,

            }

            logger.info(f"--- [会话: {session_id}] 正在持久化分析数据... ---")

            await store_user_analysis_data(session_id, analysis_data_to_persist)

            logger.info(f"--- [会话: {session_id}] 分析数据持久化完成。 ---")


            if full_response_to_yield_and_save:
                logger.info(f"--- [会话: {session_id}] 正在更新聊天历史... ---")
                try:
                    messages_to_add = [
                        HumanMessage(content=prompt_input),
                        AIMessage(content=full_response_to_yield_and_save)
                    ]

                    if hasattr(history_manager, 'add_messages') and callable(history_manager.add_messages):
                        logger.info(
                            f"...检测到兼容的 History Manager (类型: {type(history_manager).__name__})，正在异步更新。")
                        await history_manager.add_messages(messages_to_add)
                        logger.info("[紫微] 历史对话已保存到Redis")

                    elif isinstance(history_manager, list):
                        logger.warning("...检测到原始内存历史记录 (list)，正在同步更新。(注意: 这是一个兼容模式)")
                        history_manager.extend(messages_to_add)
                        logger.info("[紫微] 历史对话已保存到内存")

                    else:
                        # 如果是其他任何我们不认识的类型，就报错
                        logger.error(f"...未知的 history_manager 类型: {type(history_manager)}，无法更新历史记录。")

                    logger.info(f"--- [会话: {session_id}] 聊天历史更新完成。 ---")

                except Exception as e:
                    # 这个 except 块现在只捕获在更新过程中发生的真实错误
                    logger.error(f"--- [会话: {session_id}] 在 finally 块中更新聊天历史时发生严重错误: {e} ---",
                                 exc_info=True)

                # 记录QA日志到数据库
                if full_response_to_yield_and_save:
                    await db_manager.log_qa_record(
                        session_id=session_id,
                        app_id=app_id,
                        user_query=prompt_input,
                        final_response=full_response_to_yield_and_save
                    )
                    logger.info(f"[紫微] QA日志已记录: session_id={session_id}")

                logger.info(f"--- [会话: {session_id}] 所有持久化流程执行完毕。 ---")

        except Exception as e:
            # 捕获在 finally 块内部可能发生的任何错误
            logger.error(f"--- [会话: {session_id}] 在 finally 块中持久化或更新历史时发生严重错误: {e} ---",
                         exc_info=True)

        finally:
            # ⏱️ 总请求时间统计
            total_time = time.time() - start_time
            end_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            logger.info(f"✅ [会话: {session_id}] 请求处理完成")
            logger.info(f"📊 请求时间统计:")
            logger.info(f"   开始时间: {request_start_timestamp}")
            logger.info(f"   结束时间: {end_timestamp}")
            logger.info(f"   总耗时: {total_time:.2f}秒")
            try:
                logger.info(f"   信息解析: {parse_extract_time:.2f}秒")
                logger.info(f"   API调用: {api_call_time:.2f}秒")
                logger.info(f"   LLM分析: {llm_analysis_time:.2f}秒")
            except:
                logger.info(f"   部分时间统计数据不可用")
            logger.info(f"--- [会话: {session_id}] 请求处理结束 ---")
            log_step(
                "成功输出给客户端",
                request_id=monitor_request_id,
                extra_data={
                    "total_time_sec": round(total_time, 3),
                    "session_id": session_id
                }
            )

