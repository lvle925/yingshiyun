# clients/vllm_client.py

import asyncio
import json
import logging
import random
from typing import AsyncGenerator, Dict, Any, Tuple, Optional, Literal, List, Set
from pathlib import Path
import aiohttp
from langchain_openai import ChatOpenAI
import re
from datetime import datetime, timedelta
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tokenizer import count_tokens_for_messages, count_tokens_for_string
from fastapi import FastAPI, HTTPException, Depends, Body
from pydantic import BaseModel, Field, ValidationError, field_validator
# from .shared_client import async_aiohttp_client

from config import (
    VLLM_API_BASE_URL, VLLM_MODEL_NAME, VLLM_CONCURRENT_LIMIT,
    VLLM_SLOT_WAIT_TIMEOUT_SECONDS, VLLM_REQUEST_TIMEOUT_SECONDS, ALL_PALACES,
    MAX_API_CALL_RETRIES, MAX_STREAM_RETRIES,API_KEY
)
from prompt_logic import (
    get_ossp_xml_template,
    get_travel_advice_prompt,
    get_query_intent_prompt,
    BIRTH_INFO_EXTRACTION_PROMPT_TEMPLATE,
    # 保留这些用于向后兼容（虽然实际使用的是动态获取的函数）
    OSSP_XML_TEMPLATE_STR,
    GENERAL_QNA_XML_TEMPLATE_STR,
    TRAVEL_ADVICE_PROMPT_TEMPLATE,
    QUERY_INTENT_EXTRACTION_PROMPT_TEMPLATE,
    reload_all_prompts,
)
from models import BirthInfo, LLMExtractionResult, MultiTimeAnalysisResult, QueryTypeClassificationResult, DetailedIntentExtractionResult
from monitoring import VLLM_REQUESTS_SENT_ATTEMPTS, VLLM_RESPONSES_SUCCESS, VLLM_RESPONSES_FAILED

from utils import _map_topics_to_palaces, _parse_lenient_json, simple_clean_birth_info, \
    validate_branch_in_prompt, validate_birth_info_logic


# import db_manager
from clients import shared_client

logger = logging.getLogger(__name__)


# 并发控制器
vllm_semaphore: Optional[asyncio.Semaphore] = None


def initialize_vllm_semaphore():
    """初始化VLLM信号量"""
    global vllm_semaphore
    if vllm_semaphore is None:
        vllm_semaphore = asyncio.Semaphore(VLLM_CONCURRENT_LIMIT)
        logger.info(f"VLLM信号量已初始化，并发限制: {VLLM_CONCURRENT_LIMIT}")
    else:
        logger.warning("VLLM信号量已存在，跳过重复初始化")


UNPREDICTABLE_FUTURE_MESSAGE = "很抱歉，我无法预测“下辈子”或“来世”的运势。紫微斗数侧重于对您今生命运和运势的分析。"
MISSING_BIRTH_INFO_MESSAGE = "很抱歉，我需要完整的出生信息（公历或农历日期、时辰、性别）才能为您排盘分析。请您补充完整。\n\n例如：我的出生日期是1990年1月1日早上8点，我是女性。"

# 注意：QueryIntent 类已移除，现在使用 QueryTypeClassificationResult 和 DetailedIntentExtractionResult 两步法


# 辅助函数，来自您原代码的 ziwei_ai_function.py


# LangChain LLM 实例，用于模板格式化
llm = ChatOpenAI(
    base_url=VLLM_API_BASE_URL,
    model_name=VLLM_MODEL_NAME,
    api_key=API_KEY,
    temperature=0.7
)


def get_analysis_chain():
    """
    获取不同分析类型的、基于XML的、语法纯净的LangChain提示模板。
    """

    # --- 模板 A: 详细报告生成器 (overall_summary) ---
    # 【修复】动态获取最新的提示词模板，避免模块导入时序问题
    try:
        # 每次调用前先热更新一次提示词，避免缓存旧版本模板
        try:
            reload_all_prompts(use_lock=False)
        except Exception as reload_err:
            logger.warning(f"重新加载提示词模板时出现非致命错误，将尝试使用已有缓存: {reload_err}")

        ossp_template_str = get_ossp_xml_template()
        if not ossp_template_str or len(ossp_template_str.strip()) == 0:
            logger.error("❌ OSSP 提示词模板为空！请检查 prompts/ossp_xml_template_str.xml 文件是否已正确加载")
            logger.error(f"提示词目录: {Path('prompts').absolute()} (存在: {Path('prompts').exists()})")
            # 再尝试重新加载一次（带锁）
            reload_all_prompts(use_lock=True)
            ossp_template_str = get_ossp_xml_template()
            if not ossp_template_str or len(ossp_template_str.strip()) == 0:
                raise ValueError("OSSP_XML_TEMPLATE_STR 未正确加载，无法创建提示模板")
            else:
                logger.info("✅ 重新加载后成功获取 OSSP 提示词模板")
    except Exception as e:
        logger.error(f"❌ 获取 OSSP 提示词模板时出错: {e}", exc_info=True)
        raise ValueError(f"OSSP_XML_TEMPLATE_STR 未正确加载: {e}")

    overall_summary_prompt_template = ChatPromptTemplate.from_messages([
        ("system", ossp_template_str)
    ])

    # --- 模板 B: 通用问答处理器 (general_question & other_person) ---
    # 【已修正】: 将所有的 {{...}} 修改为 {...}
    general_qna_xml_str = """
    <prompt>
    <internal_analysis_framework>
        <global_context>
            <role>你是一位顶级的紫微斗数命理分析宗师，擅长对具体问题进行追问解答。</role>
            <user_question>{question}</user_question>
            <analysis_time_scope>{analysis_time_scope_str}</analysis_time_scope>
            <data_source type="紫微斗数命盘数据" name="EVIDENCE_BASE">
                {full_structured_analysis_data_json}
            </data_source>
            <supplementary_data type="完整十二宫原始数据" name="FULL_CHART_DATA">
                {full_chart_data_json}
            </supplementary_data>
        </global_context>
        <core_methodology>
            **核心分析方法**: 你的回答必须深度结合 <data_source> 中的数据，直接、清晰地回答用户的追问。所有分析都必须基于 <data_source> 中的事实。

            **【询问他人时的分析规则】**
            - 如果用户询问的是他人（如子女、配偶、父母等）的运势，请注意：
              - <data_source> 中的数据是**用户本人的命盘数据**，不是询问对象的命盘。
              - 如果 <data_source> 中包含"子女宫"、"夫妻宫"、"父母宫"等宫位，这些宫位代表用户本人与询问对象的关系和互动，可以用来间接分析询问对象的情况，另外身份识别不要混乱，如用户提问的是“弟”，“妹”，“哥”，“姐”，这都是跟用户的“兄弟宫”相关。
              - **绝对禁止**使用"命宫"信息来分析他人，命宫只代表用户本人的性格和特质。
              - 在输出中必须明确说明这是"根据您本人的命盘推演"或"从您命盘中的XX宫位来看"等表述。
              - 如果用户没有提供询问对象的出生信息，这是基于用户本人命盘的间接分析，不是精确的独立分析。
              - 对于分析关系和互动，必须参考以下宫位和对应的限制范围：
                    - **命宫**: 用以描述**核心人格、行为模式、做事风格**。
                    - **福德宫**: 用以描述**精神世界、心态、思想、深层价值观**。
                    - **事业宫**: 用以描述**工作、事业、学业、职场**方面的状况，**以及在各项行动、计划推进中的顺遂程度和进展**。同时揭示相应的**吉凶**。
                    - **财帛宫**: 用以描述**财富来源、资产状况、消费偏好**。
                    - **迁移宫**: 用以描述**在外发展、出行、人际互动、给外界的印象**以及状态的**吉凶**。
                    - **田宅宫**: 用以描述**家庭环境、居住状况、公司状况、学校状况、办公室环境、不动产**。
                    - **夫妻宫**: 用以描述**婚姻、恋爱关系、择偶标准**以及相应关系的**吉凶**。
                    - **子女宫**: 用以描述与**子女、晚辈、下属、宠物、合伙项目**相关的事宜以及相应关系的**吉凶**。
                    - **兄弟宫**: 用以描述与**同辈手足、核心好友、合作搭档**的关系以及相应关系的**吉凶**。
                    - **父母宫**: 用以描述与**父母、长辈、直属领导、权威人士**的关系以及相应关系的**吉凶**。
                    - **仆役宫(交友宫)**: 用以描述与**普通朋友、团队成员、更广泛的人际关系**以及相应关系的**吉凶**。
                    - **疾厄宫**: 用以描述**健康状况、身体隐患、潜在的障碍或麻烦**以及状态的**吉凶**。
            **【补充数据使用规则】**
            - 优先使用 <data_source> 中已筛选好的宫位信息。
            - 只有当 <data_source> 缺乏所需细节时，才可查阅 <supplementary_data>。
            - 从 <supplementary_data> 引用信息时，必须明确指出该信息属于补充材料，并说明其与用户问题的关联。

            **【时间一致性强制规则】**
            - `<analysis_time_scope>` 中已经给出本次推演的准确时间范围，回答时必须逐字引用该描述，或在首次出现时使用“针对【<analysis_time_scope>】……”的结构。
            - 所有结论与建议的时间指代必须与 `<analysis_time_scope>` 完全一致，严禁出现与之冲突的词语（如默认写“今年”“最近”“当前”等）。
            - 若 `<analysis_time_scope>` 同时包含相对时间与具体年份（例如“明年 · 2026 年”），首句必须同时呈现相对称呼与年份（示例：“针对明年（2026年）的健康走势……”）。
        </core_methodology>
    </internal_analysis_framework>

    <final_output_instructions>
        **【最终输出渲染指令】**
        现在，请忘记所有XML标签，将你的分析结果转化成一篇对用户友好的、结构清晰的追问解答。

        - **【强制】输出结构**: 你的回答必须严格遵循以下两个部分的 Markdown 结构。每个部分都必须使用二级标题 (`###`)。
            1.  `### 核心结论`
            2.  `### 调整建议`
            @ 是否想问

        - **【强制】内容要求**:
            - **核心结论**: 用一到两句极其精炼的话，直接给出问题的核心答案。
            - **调整建议**: 基于命盘解读，提供2-3条具体的、可操作的行动建议。
            - **是否想问**：基于用户的提问题和输出的回答，给出拓展性的提问建议。
            - 核心结论部分第一句话必须明确引用 `<analysis_time_scope>` 描述（例如：“针对【明年（2026年）】的健康走势……”），并严禁出现任何与该时间范围不符的指代词。

        - **【绝对禁令】**: 严禁输出XML标签。严禁分析紫微斗数以外的数据。严格遵守时间范围。
    </final_output_instructions>

    <!-- ==================== 输出格式样例 (金标准) ==================== -->
    <example_output>
## 核心结论
本日在工作中，与同事的合作确实存在不小的挑战，过程可能不会太顺利，需要更多耐心和沟通技巧来应对。

## 调整建议
为应对本日的挑战，可以尝试以下调整：
- **提前沟通，明确分工**: 在开始合作任务前，主动与同事进行一次简短沟通，明确各自的职责和期望，避免后续因理解偏差产生矛盾。
- **保持情绪稳定**: 当遇到不同意见时，先深呼吸，避免立即反驳。尝试换位思考，理解对方的出发点，再提出自己的看法。

@ 是否想问
    (指令：此为报告的最后一部分。你的任务是基于对【用户当前问题的主题】和你给出的【分析结论】，为用户生成3个他们可能立即想追问的、具体的、且你可以基于`<evidence_base>`数据回答的后续问题。问题必须直接、精炼，并以项目符号(-)呈现。)

    **【问题生成思维链（实用视角）】**
    你必须严格遵循以下两种模式来构思问题，确保生成的问题与用户当前关心的话题高度相关：

    1.  **【时间扩展模式 (Time-Frame Expansion)】**:
        -   **逻辑**: 锁定用户当前问题的核心【主题】（例如：财运、事业、感情），然后将这个问题应用到其他重要且可分析的时间尺度上（如：流年、大运）。
        -   **核心任务**: 识别`<analysis_scope>`中的当前时间尺度，然后生成关于**其他时间尺度**的问题。
        -   **示例 (如果当前分析的是“今年财运”)**:
            - `- 我当前十年大运的整体财运格局是怎样的？`
            - `- 下个月我的财运会有什么具体变化吗？`
        -   **示例 (如果当前分析的是“本命事业格局”)**:
            - `- 明年我的事业上会有好的发展机会吗？`

    2.  **【判断性深化模式 (Judgmental Deep-Dive)】**:
        -   **逻辑**: 将用户较为宽泛的问题，转化为一个更具体的、“是/否”类型的判断性问题。这能帮助用户获得更明确的预期。
        -   **核心任务**: 从你给出的分析中，找到一个关键的机遇点或风险点，并将其构造成一个判断题。
        -   **示例 (如果当前问题是“财运如何”，且分析显示有投资机会)**:
            - `- 我明年会有加薪的可能吗？`
            - `- 如果进行一项新的投资，会不会有不错的收益？`
        -   **示例 (如果当前问题是“感情怎么样”，且分析显示有桃花)**:
            - `- 我今年能遇到合适的结婚对象吗？`

    **【输出要求】**
    -   生成的问题必须是用户视角的第一人称提问。
    -   问题必须具体、实用，是普通用户会关心的实际问题。
    -   **绝对禁止**使用“格局”、“星曜组合”、“三方四正”等过于专业的命理术语。
    -   **绝对禁止**提出无法基于`<evidence_base>`回答的问题。

    <!-- **下方为占位符，你必须生成真实的问题来替换它们** -->
    - [基于时间扩展模式生成的建议问题]
    - [基于判断性深化模式生成的建议问题]
    - [另一个基于以上两种模式生成的建议问题]

    </example_output>
    <!-- ============================================================== -->

</prompt>
/no_think
    """
    general_question_prompt_template = ChatPromptTemplate.from_messages([
        ("system", general_qna_xml_str),
        MessagesPlaceholder(variable_name="history"),
        # 【已修正】: 这里也要用单大括号
        ("human", "<user_question>{question}</user_question>")
    ])

    # --- 模板 C: 简单固定回复 ---
    # 【注意】: 确保这里的 f-string 能正确工作
    # 如果 MISSING_BIRTH_INFO_MESSAGE 自身包含大括号，需要用双大括号转义
    # 例如：f"{{'key': '{value}'}}" -> {'key': '...'}
    # 但如果它只是普通字符串，f-string 是多余的，可以直接用
    missing_birth_info_prompt_template = ChatPromptTemplate.from_template(f"gac,{MISSING_BIRTH_INFO_MESSAGE}")
    unpredictable_future_prompt_template = ChatPromptTemplate.from_template(f"{UNPREDICTABLE_FUTURE_MESSAGE}")

    # 【修复】动态获取出行建议提示词模板，避免模块导入时序问题
    travel_template_str = get_travel_advice_prompt()
    if not travel_template_str or len(travel_template_str.strip()) == 0:
        logger.warning("⚠️ TRAVEL_ADVICE_PROMPT_TEMPLATE 为空，使用默认模板")
        travel_template_str = "你是一位出行顾问，请根据用户的命盘信息提供出行建议。"

    travel_advice_prompt_template = ChatPromptTemplate.from_messages([("system", travel_template_str)])
    return {
        "overall_summary": {"prompt": overall_summary_prompt_template},
        "general_question": {"prompt": general_question_prompt_template},
        "missing_birth_info": {"prompt": missing_birth_info_prompt_template},
        "missing_other_person_birth_info": {"prompt": general_question_prompt_template},  # 复用
        "unpredictable_future": {"prompt": unpredictable_future_prompt_template},
        "travel_advice": {"prompt": travel_advice_prompt_template},
    }


# def get_analysis_chain():
#    """获取不同分析类型的LangChain提示模板。"""
#    overall_summary_prompt_template = ChatPromptTemplate.from_messages([("system", OSSP_XML_TEMPLATE_STR)])
#    general_question_prompt_template = ChatPromptTemplate.from_messages([
#        ("system", GENERAL_QNA_XML_TEMPLATE_STR),
#        MessagesPlaceholder(variable_name="history"),
#        ("human", "<user_question>{question}</user_question>")
#    ])
#    travel_advice_prompt_template = ChatPromptTemplate.from_messages([("system", TRAVEL_ADVICE_PROMPT_TEMPLATE)])
#    return {
#       "overall_summary": {"prompt": overall_summary_prompt_template},
#        "general_question": {"prompt": general_question_prompt_template},
#        "travel_advice": {"prompt": travel_advice_prompt_template},
#    }

async def aiohttp_vllm_invoke(
        payload: dict,
        # http_client: aiohttp.ClientSession, # <--- 新增
        max_retries: int = MAX_API_CALL_RETRIES
) -> Tuple[Dict[str, Any], int, int]:
    """
    使用全局 aiohttp 客户端调用 VLLM，并带有重试机制以确保获取到有效的JSON响应。
    """
    input_tokens = count_tokens_for_messages(payload.get("messages", []))

    if not shared_client.async_aiohttp_client:
        logger.error("AIOHTTP客户端未初始化。")
        raise HTTPException(503, "AIOHTTP客户端未初始化。")

    if vllm_semaphore is None:
        logger.warning("VLLM信号量未初始化，正在初始化...")
        initialize_vllm_semaphore()

    url = f"{VLLM_API_BASE_URL}/chat/completions"
    logger.info(f"准备调用VLLM API: {url}")
    logger.debug(f"VLLM_API_BASE_URL配置值: {VLLM_API_BASE_URL}")
    last_exception = None
    headers={
           "Authorization": f"Bearer {API_KEY}",
           "Content-Type": "application/json"
       }
    for attempt in range(max_retries):
        logger.info(f"正在调用VLLM (第 {attempt + 1}/{max_retries} 次)...")
        raw_content_for_logging = ""

        try:
            await asyncio.wait_for(vllm_semaphore.acquire(), timeout=VLLM_SLOT_WAIT_TIMEOUT_SECONDS)
            try:
                timeout = aiohttp.ClientTimeout(total=VLLM_REQUEST_TIMEOUT_SECONDS)
                logger.debug(f"获取到VLLM许可，准备发送请求到 {url} (超时: {timeout.total}s)")
                async with shared_client.async_aiohttp_client.post(url, json=payload,headers =headers, timeout=timeout) as response:
                    # 即使是4xx/5xx错误，也可能值得重试（例如503 Service Unavailable）
                    # 但为了简单起见，我们先只处理200 OK的情况，对于非200直接重试
                    if response.status != 200:
                        # 抛出一个可捕获的异常，触发重试
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"VLLM返回非200状态码: {response.status}",
                            headers=response.headers,
                        )

                    json_response = await response.json()
                    content = json_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    print("content", content)
                    raw_content_for_logging = content

                    output_tokens = count_tokens_for_string(content)

                    if not content or not content.strip():
                        # 返回空内容也视为一次失败的尝试
                        raise ValueError("VLLM返回的 content 为空。")

                    # 使用健壮的解析函数，如果解析失败会抛出 JSONDecodeError
                    parsed_json = _parse_lenient_json(content)

                    # 成功获取并解析了JSON，直接返回
                    logger.info(f"VLLM调用成功 (第 {attempt + 1} 次尝试)。")
                    # return parsed_json
                    return parsed_json, input_tokens, output_tokens  # 新的返回

            finally:
                vllm_semaphore.release()
                logger.debug("VLLM许可已释放。")

        # --- START: 捕获可重试的异常 ---
        except (asyncio.TimeoutError, aiohttp.ClientError, json.JSONDecodeError, ValueError) as e:
            # TimeoutError: 等待许可超时
            # ClientError: 网络问题或非200状态码
            # JSONDecodeError / ValueError: 返回的不是有效的JSON或内容为空

            logger.warning(
                f"VLLM调用在第 {attempt + 1} 次尝试时失败。错误类型: {type(e).__name__}, 错误: {e}. "
                f"原始Content(如果可用): '{raw_content_for_logging}'"
            )
            last_exception = e

            # 如果不是最后一次尝试，则等待后重试
            if attempt < max_retries - 1:
                # 对不同错误类型可以设置不同等待时间
                wait_time = 1.0 if isinstance(e, aiohttp.ClientError) else 0.5
                await asyncio.sleep(wait_time)
            else:
                # 这是最后一次尝试失败，跳出循环，在下面统一处理
                break
        # --- END: 捕获可重试的异常 ---

        except Exception as e:
            # 捕获其他所有未预料的、不应重试的严重错误
            logger.error(f"【FATAL】aiohttp_vllm_invoke: 发生不可重试的未知错误: {e}", exc_info=True)
            # 直接向上抛出，让上层应用知道发生了严重问题
            # raise HTTPException(500, f"VLLM调用时发生未知且严重的错误: {e}")
            return None, input_tokens, 0

    # --- 所有重试都失败后的最终处理 ---
    logger.error(f"在 {max_retries} 次尝试后，仍无法从VLLM获取有效响应。最后一次错误: {last_exception}")

    # 根据最后一次的异常类型，返回一个更具体的HTTPException
    if isinstance(last_exception, asyncio.TimeoutError):
        # raise HTTPException(503, "AI服务持续繁忙，请稍后再试。")
        return None, input_tokens, 0
    if isinstance(last_exception, aiohttp.ClientError):
        # raise HTTPException(503, f"连接AI服务失败: {last_exception}")
        return None, input_tokens, 0
    if isinstance(last_exception, (json.JSONDecodeError, ValueError)):
        # raise HTTPException(500, "AI服务持续返回无效格式的数据。")
        return None, input_tokens, 0

    # 默认的最终错误
    raise HTTPException(500, "调用AI服务多次失败，请联系管理员。")


async def aiohttp_vllm_stream(
        payload: dict,
        # http_client: aiohttp.ClientSession,
        max_retries: int = 5,
        initial_delay: float = 0.5,
        max_delay: float = 4.0,

) -> AsyncGenerator[Tuple[str, int, int], None]:
    """
    (已正确改造) 使用全局 aiohttp 客户端调用 VLLM 并流式处理响应。
    集成了在连接阶段的指数退避重试机制。
    这是一个异步生成器，使用 `yield` 返回数据块。
    """
    input_tokens = count_tokens_for_messages(payload.get("messages", []))
    output_tokens_accumulator = 0

    if not shared_client.async_aiohttp_client:
        logger.error("AIOHTTP客户端未初始化。")
        yield "[错误: AIOHTTP客户端未初始化。][DONE]"
        return  # 使用 return 结束生成器

    if vllm_semaphore is None:
        logger.warning("VLLM信号量未初始化，正在初始化...")
        initialize_vllm_semaphore()

    # 确保 payload 中 stream=True
    payload['stream'] = True

    url = f"{VLLM_API_BASE_URL}/chat/completions"
    delay = initial_delay
    last_exception = None

    # 获取信号量，这部分在所有重试之外，只获取一次
    try:
        await asyncio.wait_for(vllm_semaphore.acquire(), timeout=VLLM_SLOT_WAIT_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.error("等待VLLM许可超时 (aiohttp_vllm_stream)。")
        VLLM_RESPONSES_FAILED.labels(reason="semaphore_timeout").inc()
        yield "[错误: AI服务正忙，请稍后再试。][DONE]"
        return

    try:
        headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
                    }
        # 重试循环只用于建立连接和获取响应头
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    wait_time = delay + random.uniform(0, delay * 0.5)
                    logger.warning(f"VLLM(stream)连接失败，将在 {wait_time:.2f} 秒后进行第 {attempt} 次重试...")
                    await asyncio.sleep(wait_time)
                    delay = min(delay * 2, max_delay)

                VLLM_REQUESTS_SENT_ATTEMPTS.inc()
                logger.info(f"计数: 尝试发送到VLLM的请求数 +1。当前总数: {VLLM_REQUESTS_SENT_ATTEMPTS._value.get()}")
                logger.debug(f"尝试第 {attempt + 1} 次连接VLLM(stream)到 {url}")
                timeout = aiohttp.ClientTimeout(total=VLLM_REQUEST_TIMEOUT_SECONDS)
                async with shared_client.async_aiohttp_client.post(url, json=payload,headers = headers,timeout=timeout) as response:

                    # 检查初始响应状态码，如果失败则重试
                    response.raise_for_status()

                    # --- 连接成功，开始流式处理数据 ---
                    # 一旦进入这个循环，就不再重试了
                    raw_lines: List[str] = []
                    yielded_any = False

                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if not line:
                            continue

                        raw_lines.append(line)

                        if line.startswith("data:"):
                            line_data = line[5:].strip()
                            if line_data == "[DONE]":
                                break  # 流结束
                            try:
                                chunk = json.loads(line_data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content_piece = delta.get("content")
                                if content_piece:
                                    output_tokens_accumulator += count_tokens_for_string(content_piece)
                                    yielded_any = True
                                    # 使用 yield 返回数据块
                                    yield content_piece, input_tokens, output_tokens_accumulator
                            except json.JSONDecodeError:
                                logger.warning(f"无法解析VLLM流中的JSON行: {line_data}")
                                continue

                    # 如果整个流期间一次内容都没有产出，尝试将其视为非流式JSON进行兜底解析
                    if not yielded_any:
                        raw_body = "\n".join(raw_lines)
                        logger.warning(
                            f"VLLM(stream)连接成功但未返回任何内容，尝试按非流式JSON解析。"
                            f" 原始内容前500字符: {raw_body[:500]}"
                        )

                        parsed = None
                        # 优先尝试整体解析
                        if raw_body:
                            try:
                                parsed = json.loads(raw_body)
                            except json.JSONDecodeError:
                                parsed = None

                        # 如果整体解析失败，再尝试逐行解析（有些服务可能一行一个JSON）
                        if parsed is None:
                            for candidate in reversed(raw_lines):
                                try:
                                    parsed = json.loads(candidate)
                                    break
                                except json.JSONDecodeError:
                                    continue

                        if parsed is not None:
                            try:
                                choices = parsed.get("choices", [])
                                if choices:
                                    first_choice = choices[0] or {}
                                    message = first_choice.get("message") or {}
                                    content_piece = message.get("content")

                                    # 兼容某些返回只提供 delta 的情况
                                    if not content_piece:
                                        delta = first_choice.get("delta", {})
                                        content_piece = delta.get("content")

                                    if content_piece:
                                        output_tokens_accumulator += count_tokens_for_string(content_piece)
                                        logger.info("VLLM(stream)无SSE数据行，但成功从整体JSON响应中解析到内容。")
                                        VLLM_RESPONSES_SUCCESS.inc()
                                        logger.info(f"计数: VLLM成功响应数 +1。当前总数: {VLLM_RESPONSES_SUCCESS._value.get()}")
                                        yield content_piece, input_tokens, output_tokens_accumulator
                                        return
                            except Exception as e:
                                logger.error(f"兜底解析VLLM整体JSON响应时发生错误: {e}", exc_info=True)

                        # 如果仍然无法解析任何内容，则明确返回错误，避免上层误以为空内容
                        logger.error("VLLM(stream)连接成功但未能从响应中解析出任何内容，将返回错误提示。")
                        VLLM_RESPONSES_FAILED.labels(reason="empty_stream").inc()
                        error_message = "[错误: AI服务繁忙...]"
                        yield error_message, input_tokens, 0
                        return

                    # 正常流式情况下，至少产生过一次内容
                    VLLM_RESPONSES_SUCCESS.inc()
                    logger.info(f"计数: VLLM成功响应数 +1。当前总数: {VLLM_RESPONSES_SUCCESS._value.get()}")
                    # 流处理成功，直接返回以结束函数
                    return

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                # 捕获所有适合重试的连接错误
                logger.warning(f"VLLM(stream)连接尝试 #{attempt + 1} 遇到错误: {type(e).__name__}")
                last_exception = e
                # 如果是最后一次尝试，则在循环外处理
                if attempt == max_retries:
                    logger.error(f"VLLM(stream)在 {max_retries} 次重试后连接彻底失败。", exc_info=True)
                    VLLM_RESPONSES_FAILED.labels(reason="connection_error").inc()
                    # yield f"\n[错误: 连接AI服务失败: {last_exception}][DONE]"
                    error_message = "[错误: AI服务繁忙...]"
                    yield error_message, input_tokens, 0  # 新的方式

                continue  # 继续下一次重试

    except Exception as e:
        logger.error(f"处理VLLM(stream)时发生未知错误: {e}", exc_info=True)
        VLLM_RESPONSES_FAILED.labels(reason="unknown_stream_error").inc()
        # yield "\n[错误: 处理AI响应时发生未知错误][DONE]"
        error_message = "[错误: AI服务繁忙...]"
        yield error_message, input_tokens, 0  # 新的方式


    finally:
        # 确保信号量被释放
        vllm_semaphore.release()
        logger.debug("VLLM(stream)许可已释放。")


async def extract_birth_info_with_llm(user_input: str,
                                      # http_client: aiohttp.ClientSession,
                                      max_retries: int = 5) -> Dict[str, Any]:
    """
    使用 LLM 提取用户输入的出生信息。
    """
    # 原始的系统Prompt内容

    print("user_input", user_input)

    system_prompt_content = (
        """<|im_start|>system
你是一个专门用于解析用户出生信息的AI助手，角色是【信息提取器】。
你的唯一任务是分析下面<user_input>标签中的文本，并严格按照<output_format>中定义的JSON结构和规则，提取并生成一个JSON对象。

**【核心指令】**
1.  **绝对专注**: 只关注明确的、与个人出生相关的年月日时和性别信息。完全忽略任何与“运势查询”、“未来时间”等相关的词语，如“明天”、“2025年运势”。
2.  **严格格式**: 你的输出必须且只能是一个JSON对象。在JSON对象生成完毕后，立即停止，不得添加任何解释、注释或额外文本。
3.  **数值转换**: `year`, `month`, `day`, `hour`, `minute` 字段必须是整数（integer）。将所有口语化时间（如“晚上9点半”）精确转换为24小时制的数字。
4.  **空值处理**: 如果任何信息未被提供，对应的字段值必须为`null`。如果未提及分钟，`minute`默认为`0`。
5.  **时辰优先级**: 只有当用户明确使用“子时”、“丑时”等十二地支时辰词语时，才填充`traditional_hour_branch`字段（例如'丑时'），并同时将`hour`字段设为`null`。这是唯一`hour`可以为`null`而`traditional_hour_branch`有值的场景。

**【输出格式定义】**
<output_format>
{
  "year": "integer | null - 出生公历年份",
  "month": "integer | null - 出生月份 (1-12)",
  "day": "integer | null - 出生日期 (1-31)",
  "hour": "integer | null - 24小时制出生小时 (0-23)",
  "minute": "integer | null - 出生分钟 (0-59)",
  "gender": "string | null - '男' 或 '女'",
  "is_lunar": "boolean - 是否为农历，默认为false",
  "traditional_hour_branch": "string | null - 传统时辰，如 '子时', '丑时'等"
}
</output_format>

**【处理示例】**
这里有一些你必须严格遵循的输入输出范例。

<example>
  <user_input>我的阳历出生日期是1990年1月1日早上8点，我是女性</user_input>
  <assistant_output>
    {"year": 1990, "month": 1, "day": 1, "hour": 8, "minute": 0, "gender": "女", "is_lunar": false, "traditional_hour_branch": null}
  </assistant_output>
</example>
<example>
  <user_input>农历1985年冬月十五晚上9点半，女</user_input>
  <assistant_output>
    {"year": 1985, "month": 11, "day": 15, "hour": 21, "minute": 30, "gender": "女", "is_lunar": true, "traditional_hour_branch": null}
  </assistant_output>
</example>
<example>
  <user_input>我是2000年3月3日丑时出生的女性</user_input>
  <assistant_output>
    {"year": 2000, "month": 3, "day": 3, "hour": null, "minute": 0, "gender": "女", "is_lunar": false, "traditional_hour_branch": "丑时"}
  </assistant_output>
</example>
<example>
  <user_input>我是个女孩，生日是2001年5月20号</user_input>
  <assistant_output>
    {"year": 2001, "month": 5, "day": 20, "hour": null, "minute": 0, "gender": "女", "is_lunar": false, "traditional_hour_branch": null}
  </assistant_output>
</example>
<example>
  <user_input>1990年生的，我想问问12月12日运势如何</user_input>
  <assistant_output>
    {"year": 1990, "month": null, "day": null, "hour": null, "minute": 0, "gender": null, "is_lunar": false, "traditional_hour_branch": null}
  </assistant_output>
</example>
<example>
  <user_input>我的命宫怎么样？</user_input>
  <assistant_output>
    {"year": null, "month": null, "day": null, "hour": null, "minute": 0, "gender": null, "is_lunar": false, "traditional_hour_branch": null}
  </assistant_output>
</example>
<example>
  <user_input>我的阳历出生日期是1990年1月1日早上8点多，我是女性</user_input>
  <assistant_output>
    {"year": 1990, "month": 1, "day": 1, "hour": 8, "minute": 30, "gender": "女", "is_lunar": false, "traditional_hour_branch": null}
  </assistant_output>
</example>
<example>
  <user_input>我的阳历出生日期是1990年1月1日5点多，我是女性</user_input>
  <assistant_output>
    {"year": 1990, "month": 1, "day": 1, "hour": 5, "minute": 30, "gender": "女", "is_lunar": false, "traditional_hour_branch": null}
  </assistant_output>
</example>
<|im_end|>""")

    user_prompt_content = f"<user_input>{user_input}</user_input>"

    payload = {
        "model": VLLM_MODEL_NAME,  # 假设这是定义的常量
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_prompt_content}
        ],
        "temperature": 0.0,  # 可以稍微提高一点温度，让重试时有不同的结果，但不要太高
        "max_tokens": 300,
        "response_format": {"type": "json_object", "schema": BirthInfo.model_json_schema()}
        # 假设 BirthInfo 是你的 Pydantic 模型
    }
    print("payload", payload)

    last_exception = None
    for attempt in range(max_retries):
        logger.info(f"正在尝试提取出生信息 (第 {attempt + 1}/{max_retries} 次)...")
        try:
            structured_response, in_tokens, out_tokens = await aiohttp_vllm_invoke(payload)

            if not isinstance(structured_response, dict):
                raise ValueError(f"LLM did not return a dictionary. Raw response: {structured_response}")

            # 调用我们全新升级的、更强大的清洗函数
            cleaned_response = simple_clean_birth_info(structured_response)

            validated_model = BirthInfo.model_validate(cleaned_response)
            validated_info_dict = validated_model.model_dump()

            # --- 【新增】4. 业务逻辑验证 ---
            # 对年月日时等进行深度逻辑校验
            validate_birth_info_logic(validated_info_dict)
            # 如果 validate_birth_info_logic 发现问题，会抛出 ValueError,
            # 该异常会被下面的 except 块捕获，并触发重试。

            # 5. 可选的额外处理 (您的原始逻辑)
            result_with_branch_check = validate_branch_in_prompt(validated_info_dict, user_input)

            logger.info("出生信息提取、清洗、验证并逻辑校验成功！")
            return result_with_branch_check

        except (ValidationError, ValueError) as e:
            log_response = "N/A"
            try:
                log_response = str(structured_response)
            except NameError:
                pass

            logger.warning(
                f"第 {attempt + 1} 次尝试失败。错误: {e}。清洗前的原始输出: {log_response}"
            )
            last_exception = e
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"第 {attempt + 1} 次尝试时发生意外错误: {e}", exc_info=True)
            last_exception = e
            if attempt < max_retries - 1:
                await asyncio.sleep(1)

    logger.error(f"在 {max_retries} 次尝试后，仍无法提取有效的出生信息。最后一次错误: {last_exception}")
    return {"error": "AI多次尝试后仍无法解析有效的出生信息。"}


async def classify_query_time_type(
        user_input: str,
        current_dt: datetime,
        time_range_info: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
) -> Optional[QueryTypeClassificationResult]:
    """
    第一步：判断查询的时间粒度类型（单流年、单大运、单流月、单流日、一生查询）。
    
    Args:
        user_input: 用户输入
        current_dt: 当前日期时间
        time_range_info: 可选的时间范围检测结果，包含 has_time_range, end_date, time_expression, reason
        max_retries: 最大重试次数
    
    Returns:
        QueryTypeClassificationResult对象，如果判断失败则返回None
    """
    from prompt_logic import load_xml_prompt
    
    # 加载提示词模板
    prompt_template = load_xml_prompt("query_type_classification.xml")
    if not prompt_template:
        logger.warning("无法加载 query_type_classification.xml 提示词，跳过查询类型分类")
        return None
    
    # 构建schema
    schema_as_dict = QueryTypeClassificationResult.model_json_schema()
    schema_as_string = json.dumps(schema_as_dict, indent=2, ensure_ascii=False)
    
    # 替换提示词中的schema占位符
    system_prompt_content = prompt_template.replace("{schema_as_string}", schema_as_string)
    
    # 构建时间范围信息部分
    time_range_context = ""
    if time_range_info:
        has_time_range = time_range_info.get("has_time_range", False)
        end_date = time_range_info.get("end_date")
        time_expression = time_range_info.get("time_expression")
        reason = time_range_info.get("reason", "")
        
        time_range_context = f"""
    <time_range_detection>
        <has_time_range>{has_time_range}</has_time_range>
        <end_date>{end_date if end_date else 'null'}</end_date>
        <time_expression>{time_expression if time_expression else 'null'}</time_expression>
        <reason>{reason}</reason>
    </time_range_detection>
    """
    
    user_prompt_content = f"""
    <context>
        <current_date>{current_dt.strftime('%Y-%m-%d')}</current_date>
    </context>
    {time_range_context}
    <user_input>
    {user_input}
    </user_input>
    """
    
    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_prompt_content},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object", "schema": schema_as_dict}
    }
    
    last_exception = None
    for attempt in range(max_retries):
        try:
            logger.info(f"正在尝试进行查询类型分类 (第 {attempt + 1}/{max_retries} 次)...")
            structured_response, input_tokens, output_tokens = await aiohttp_vllm_invoke(payload)
            result = QueryTypeClassificationResult.model_validate(structured_response)
            
            logger.info(f"✅ 查询类型分类成功：{result.query_time_type}")
            return result
            
        except (ValidationError, ValueError) as e:
            logger.warning(f"查询类型分类尝试 {attempt + 1} 失败，LLM输出不符合格式。错误: {e}")
            last_exception = e
        except Exception as e:
            logger.error(f"在第 {attempt + 1} 次查询类型分类尝试中发生意外错误: {e}", exc_info=True)
            last_exception = e
        
        if attempt < max_retries - 1:
            await asyncio.sleep(0.5)
    
    logger.error(f"在 {max_retries} 次尝试后，无法有效进行查询类型分类。最后一次错误: {last_exception}")
    return None


async def check_multi_time_analysis(
        user_input: str,
        current_dt: datetime,
        max_retries: int = 3
) -> Optional[MultiTimeAnalysisResult]:
    """
    检查用户查询是否属于多流年、多大运或多流月分析类型。
    如果属于这三种类型之一，返回分析结果；否则返回None。
    
    Args:
        user_input: 用户输入
        current_dt: 当前日期时间
        max_retries: 最大重试次数
    
    Returns:
        MultiTimeAnalysisResult对象，如果不属于这三种类型则返回None
    """
    from prompt_logic import load_xml_prompt
    
    # 加载提示词模板
    prompt_template = load_xml_prompt("multi_time_analysis.xml")
    if not prompt_template:
        logger.warning("无法加载 multi_time_analysis.xml 提示词，跳过多时间分析检查")
        return None
    
    # 构建schema
    schema_as_dict = MultiTimeAnalysisResult.model_json_schema()
    schema_as_string = json.dumps(schema_as_dict, indent=2, ensure_ascii=False)
    
    # 替换提示词中的schema占位符
    system_prompt_content = prompt_template.replace("{schema_as_string}", schema_as_string)
    
    user_prompt_content = f"""
    <context>
        <current_date>{current_dt.strftime('%Y-%m-%d')}</current_date>
    </context>
    <user_input>
    {user_input}
    </user_input>
    """
    
    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_prompt_content},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object", "schema": schema_as_dict}
    }
    
    last_exception = None
    for attempt in range(max_retries):
        try:
            logger.info(f"正在尝试进行多时间分析判断 (第 {attempt + 1}/{max_retries} 次)...")
            structured_response, input_tokens, output_tokens = await aiohttp_vllm_invoke(payload)
            result = MultiTimeAnalysisResult.model_validate(structured_response)
            
            # 如果判断为none，返回None
            if result.query_type == "none":
                logger.info("多时间分析判断：不属于多流年/多大运/多流月类型")
                return None
            
            logger.info(f"✅ 多时间分析判断成功：{result.query_type}")
            return result
            
        except (ValidationError, ValueError) as e:
            logger.warning(f"多时间分析判断尝试 {attempt + 1} 失败，LLM输出不符合格式。错误: {e}")
            last_exception = e
        except Exception as e:
            logger.error(f"在第 {attempt + 1} 次多时间分析判断尝试中发生意外错误: {e}", exc_info=True)
            last_exception = e
        
        if attempt < max_retries - 1:
            await asyncio.sleep(0.5)
    
    logger.error(f"在 {max_retries} 次尝试后，无法有效进行多时间分析判断。最后一次错误: {last_exception}")
    return None


async def extract_detailed_intent_info(
        user_input: str,
        query_time_type: str,
        time_expression: Optional[str],
        current_dt: datetime,
        max_retries: int = 3
) -> Optional[DetailedIntentExtractionResult]:
    """
    第二步：根据查询类型提取详细的意图信息。
    
    Args:
        user_input: 用户输入
        query_time_type: 查询类型（single_yearly, single_decadal, single_monthly, single_daily, lifetime）
        time_expression: 第一步提取的时间表达式
        current_dt: 当前日期时间
        max_retries: 最大重试次数
    
    Returns:
        DetailedIntentExtractionResult对象，如果提取失败则返回None
    """
    from prompt_logic import load_xml_prompt
    
    # 加载提示词模板
    prompt_template = load_xml_prompt("detailed_intent_extraction.xml")
    if not prompt_template:
        logger.warning("无法加载 detailed_intent_extraction.xml 提示词，跳过详细意图提取")
        return None
    
    # 构建schema
    schema_as_dict = DetailedIntentExtractionResult.model_json_schema()
    schema_as_string = json.dumps(schema_as_dict, indent=2, ensure_ascii=False)
    
    # 根据查询类型构建特定字段（只保留single_decadal的特殊字段）
    # 时间解析字段已统一在detailed_intent_extraction.xml中定义，所有类型都会提取
    type_specific_fields = ""
    if query_time_type == "single_decadal":
        type_specific_fields = """
        <!-- ====================================================== -->
        <!-- explicit_decadal_ganzhi: 提取用户点名的大运干支       -->
        <!-- ====================================================== -->
        <field name="explicit_decadal_ganzhi">
            <description>
                [行为: 精准提取]
                如果用户在问题中**明确点名了某个大运的干支名称**（如"壬戌大运"、"庚申大运"），
                则需要提取出这个干支的两个汉字，作为字符串返回。
                - 只在**明确出现"XX大运"或"XX大限"**且"XX"是标准干支组合时才填充本字段。
                - 如果用户只说"下一个大运"、"当前大运"而没有给出干支名称，必须返回 null。
                【注意】本字段**只返回干支本身**，不包含"大运"二字。
            </description>
            <examples>
                <example input="下一个大运(壬戌)对我的健康趋势会有怎样的变化？" output="壬戌" />
                <example input="想问壬戌大运期间的事业走势" output="壬戌" />
                <example input="当前大运对财运的影响" output="null" />
            </examples>
        </field>

        <!-- ====================================================== -->
        <!-- is_next_decadal_query: 判断是否询问下一个大运        -->
        <!-- ====================================================== -->
        <field name="is_next_decadal_query">
            <description>
                [行为: 精准识别]
                判断用户是否询问**下一个大运/大限**（单个，不是多个）。
                - **返回true的情况**：'下一个大运'、'下个大运'、'下一大运'、'下一个大限'、'下个大限'、'后大运'
                - **返回false的情况**：'当前大运'、'这个大运'、'上一个大运'
            </description>
            <examples>
                <example input="下一个大运" output="true" />
                <example input="下个大运" output="true" />
                <example input="当前大运" output="false" />
            </examples>
        </field>
        """
    
    # 替换提示词中的占位符
    STANDARDIZED_TOPICS = [
        "财运", "赌运", "投资", "股票", "理财",
        "事业", "工作", "学业",
        "婚姻", "感情", "桃花",
        "健康", "疾病",
        "出行",
        "人际", "社交",
        "整体运势", "命盘", "性格",
        "娱乐消遣",
        "装修", "搬家", "未知", "居家", "家庭", "田宅"
    ]
    topics_str = ", ".join(STANDARDIZED_TOPICS)
    
    palace_reference = """
        - 命宫
        - 福德宫
        - 事业宫
        - 财帛宫
        - 迁移宫
        - 田宅宫
        - 夫妻宫
        - 子女宫
        - 兄弟宫
        - 父母宫
        - 交友宫
        - 疾厄宫
    """.strip()
    
    system_prompt_content = prompt_template.replace("{schema_as_string}", schema_as_string)
    system_prompt_content = system_prompt_content.replace("{topics_str}", topics_str)
    system_prompt_content = system_prompt_content.replace("{palace_reference}", palace_reference)
    system_prompt_content = system_prompt_content.replace("{type_specific_fields}", type_specific_fields)
    
    user_prompt_content = f"""
    <context>
        <current_date>{current_dt.strftime('%Y-%m-%d')}</current_date>
        <query_time_type>{query_time_type}</query_time_type>
        <time_expression>{time_expression or 'null'}</time_expression>
    </context>
    <user_input>
    {user_input}
    </user_input>
    """
    
    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_prompt_content},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
        "response_format": {"type": "json_object", "schema": schema_as_dict}
    }
    
    last_exception = None
    for attempt in range(max_retries):
        try:
            logger.info(f"正在尝试进行详细意图提取 (第 {attempt + 1}/{max_retries} 次)...")
            structured_response, input_tokens, output_tokens = await aiohttp_vllm_invoke(payload)
            result = DetailedIntentExtractionResult.model_validate(structured_response)
            
            logger.info(f"✅ 详细意图提取成功")
            return result
            
        except (ValidationError, ValueError) as e:
            logger.warning(f"详细意图提取尝试 {attempt + 1} 失败，LLM输出不符合格式。错误: {e}")
            last_exception = e
        except Exception as e:
            logger.error(f"在第 {attempt + 1} 次详细意图提取尝试中发生意外错误: {e}", exc_info=True)
            last_exception = e
        
        if attempt < max_retries - 1:
            await asyncio.sleep(0.5)
    
    logger.error(f"在 {max_retries} 次尝试后，无法有效进行详细意图提取。最后一次错误: {last_exception}")
    return None


async def aiohttp_vllm_invoke_text(
        payload: dict,
        max_retries: int = MAX_API_CALL_RETRIES
) -> str:
    """
    调用VLLM获取纯文本响应（不解析JSON）
    专门用于专业知识问答等场景
    返回纯文本内容
    """
    input_tokens = count_tokens_for_messages(payload.get("messages", []))

    if not shared_client.async_aiohttp_client:
        logger.error("AIOHTTP客户端未初始化。")
        raise HTTPException(503, "AIOHTTP客户端未初始化。")

    if vllm_semaphore is None:
        logger.warning("VLLM信号量未初始化，正在初始化...")
        initialize_vllm_semaphore()

    url = f"{VLLM_API_BASE_URL}/chat/completions"
    logger.info(f"准备调用VLLM API (文本模式): {url}")
    last_exception = None
    headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
    for attempt in range(max_retries):
        logger.info(f"正在调用VLLM文本模式 (第 {attempt + 1}/{max_retries} 次)...")
        raw_content_for_logging = ""

        try:
            await asyncio.wait_for(vllm_semaphore.acquire(), timeout=VLLM_SLOT_WAIT_TIMEOUT_SECONDS)
            try:
                timeout = aiohttp.ClientTimeout(total=VLLM_REQUEST_TIMEOUT_SECONDS)
                logger.debug(f"获取到VLLM许可，准备发送请求到 {url} (超时: {timeout.total}s)")
                async with shared_client.async_aiohttp_client.post(url, json=payload, headers=headers,timeout=timeout) as response:
                    if response.status != 200:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"VLLM返回非200状态码: {response.status}",
                            headers=response.headers,
                        )

                    json_response = await response.json()
                    content = json_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    raw_content_for_logging = content

                    if not content or not content.strip():
                        raise ValueError("VLLM返回的 content 为空。")

                    # 直接返回纯文本，不进行JSON解析
                    logger.info(f"VLLM文本调用成功 (第 {attempt + 1} 次尝试)。")
                    return content.strip()

            finally:
                vllm_semaphore.release()
                logger.debug("VLLM许可已释放。")

        except (asyncio.TimeoutError, aiohttp.ClientError, ValueError) as e:
            logger.warning(
                f"VLLM文本调用在第 {attempt + 1} 次尝试时失败。错误类型: {type(e).__name__}, 错误: {e}. "
                f"原始Content(如果可用): '{raw_content_for_logging}'"
            )
            last_exception = e

            if attempt < max_retries - 1:
                wait_time = 1.0 if isinstance(e, aiohttp.ClientError) else 0.5
                await asyncio.sleep(wait_time)
            else:
                break

        except Exception as e:
            logger.error(f"【FATAL】aiohttp_vllm_invoke_text: 发生不可重试的未知错误: {e}", exc_info=True)
            raise HTTPException(500, f"VLLM调用时发生未知且严重的错误: {e}")

    # 所有重试都失败后的最终处理
    logger.error(f"在 {max_retries} 次尝试后，仍无法从VLLM获取有效响应。最后一次错误: {last_exception}")

    if isinstance(last_exception, asyncio.TimeoutError):
        raise HTTPException(503, "AI服务持续繁忙，请稍后再试。")
    if isinstance(last_exception, aiohttp.ClientError):
        raise HTTPException(503, f"连接AI服务失败: {last_exception}")
    if isinstance(last_exception, ValueError):
        raise HTTPException(500, "AI服务持续返回无效格式的数据。")

    raise HTTPException(500, "调用AI服务多次失败，请联系管理员。")




