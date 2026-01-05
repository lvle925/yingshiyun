# -*- coding: utf-8 -*-
# 这是一个基于 FastAPI 的 LLM 调用服务，支持并发处理请求并与 VLLM 交互。
# 新增功能：
#   - **【移除】对用户问题进行分类，移除选择类问题的所有处理逻辑。**
#   - 新增功能：根据 score_level (0, 1, 2) 调整LLM的回答风格和内容。
#   - 新增核心功能：对用户问题进行预转换，根据问题类别和score_level生成更精准的LLM问题。
#   - 【优化】针对 score_level = "0" 的回答内容，要求更加务实、具体，紧扣解决方案，且表述更具人情味。
#   - 【优化】移除LLM输出中的思考性引导语。
#   - 【优化】回答内容的时间维度始终围绕“今天”展开。
#   - **【重点优化】确保LLM回答内容与用户问题类型（如财富、事业等）的关联性，将牌意在特定领域内进行解读和转译。**

import asyncio
import uvicorn
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
import re
import aiomysql
from typing import Optional, List, AsyncGenerator, Dict, Any
from fastapi import FastAPI, HTTPException, Request, status

from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

load_dotenv()

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 配置 ---
VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME")

VLLM_MAX_RETRIES = 5
VLLM_RETRY_DELAY_SECONDS = 5.0
VLLM_REQUEST_TIMEOUT_SECONDS = float(os.getenv("VLLM_REQUEST_TIMEOUT_SECONDS"))

DB_CONFIG = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT')),
    'db': os.getenv('DB_NAME'),
    'autocommit': True
}
db_pool: Optional[aiomysql.Pool] = None

# --- 签名密钥配置 ---
APP_SECRETS: Dict[str, str] = {
    "yingshi_appid": os.getenv("APP_SECRET_yingshi_appid"),
    "test_app": os.getenv("APP_SECRET_test_app")
}

# --- CSV 数据文件路径 ---
CARDS_CSV_PATH = os.getenv('CARDS_CSV_PATH')
MEANINGS_CSV_PATH = os.getenv('MEANINGS_CSV_PATH')

renomann_cards_df: Optional[pd.DataFrame] = None
renomann_meanings_df: Optional[pd.DataFrame] = None

app = FastAPI()

async_aiohttp_client: Optional[aiohttp.ClientSession] = None
VLLM_CONCURRENT_LIMIT = int(os.getenv('VLLM_CONCURRENT_LIMIT'))
VLLM_SLOT_WAIT_TIMEOUT_SECONDS = int(os.getenv('VLLM_SLOT_WAIT_TIMEOUT_SECONDS'))
vllm_semaphore: Optional[asyncio.Semaphore] = None
next_request_id_counter = 0


# --- Pydantic 请求模型 ---
class ClientRequest(BaseModel):
    appid: str = Field(..., description="应用ID")
    prompt: str = Field(..., description="用户的问题，将用于生成LLM提示词")
    format: str = Field("json", description="响应格式，默认为json")
    ftime: int = Field(..., description="时间戳 (整数)，用于签名验证")
    sign: str = Field(..., description="请求签名，用于验证请求完整性")
    session_id: Optional[str] = Field(None, description="会话ID")
    hl_ymd: Optional[str] = Field(None, description="可选的日期参数")
    card_number_pool: Optional[List[int]] = Field(
        None,
        description="可选的卡牌编号列表，将从这个列表中随机抽取3个数字作为卡牌编号。"
    )
    score_level: str = Field(
        "0",
        description="占卜风格等级：'0' (低分解惑), '1' (提升指南), '2' (好运传导)"
    )

    # 原代码（废弃）：@validator('card_number_pool', pre=True, always=True)
    # 新代码（替换后，局部）
    @validator('card_number_pool', pre=True, always=True)  # 仅替换 validator 为 field_validator
    def parse_and_validate_card_number_pool(cls, v):
        # 内部逻辑完全不变，直接复用
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

    # 原代码（废弃）：@validator('score_level', pre=True, always=True)
    # 新代码（替换后，局部）
    @validator('score_level', pre=True, always=True)  # 仅替换 validator 为 field_validator
    def validate_score_level(cls, v):
        # 内部逻辑完全不变，直接复用
        if v not in ["0", "1", "2"]:
            raise ValueError('score_level 必须是 "0", "1", 或 "2"')
        return v


# --- 签名生成函数 ---
def generate_signature(params: Dict[str, Any], app_secret: str) -> str:
    sorted_params = dict(
        sorted({k: str(v) for k, v in params.items() if k not in ['sign', 'card_number_pool', 'hl_ymd','score_level']}.items()))
    string_to_sign = "".join(f"{k}{v}" for k, v in sorted_params.items())
    secret_bytes = app_secret.encode('utf-8')
    string_to_sign_bytes = string_to_sign.encode('utf-8')
    hmac_sha256 = hmac.new(secret_bytes, string_to_sign_bytes, hashlib.sha256)
    calculated_sign = hmac_sha256.hexdigest()
    return calculated_sign

# --- 应用启动和关闭事件 ---
@app.on_event("startup")  
async def startup_event():
    global async_aiohttp_client, vllm_semaphore
    logger.info("FastAPI 应用启动中...")
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

    try:
        app.state.db_pool = await aiomysql.create_pool(**DB_CONFIG)
        logger.info("数据库连接池创建成功并储存在 app.state 中。")
    except Exception as e:
        logger.critical(f"创建数据库连接池失败: {e}", exc_info=True)
        app.state.db_pool = None

    logger.info("FastAPI 应用启动完成。")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI 应用关闭中...")
    if async_aiohttp_client:
        await async_aiohttp_client.close()

    db_pool = getattr(app.state, 'db_pool', None)
    if db_pool:
        db_pool.close()
        await db_pool.wait_closed()
        logger.info("数据库连接池已关闭。")

    logger.info("FastAPI 应用关闭完成。")


# --- CSV 数据加载函数 ---
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


# --- 数据库日志记录函数 ---
async def log_qa_to_db(db_pool: Optional[aiomysql.Pool], session_id: str, app_id: str, user_query: str,
                       final_response: str):
    """
    将一次完整的问答记录异步写入数据库。
    """
    if not db_pool:
        logger.error(f"无法记录日志到数据库：数据库连接池不可用。 Session: {session_id}")
        return

    if not final_response or not final_response.strip():
        logger.warning(f"最终响应为空，取消数据库日志记录。 Session: {session_id}")
        return
    try:
        async with db_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                sql = """
                      INSERT INTO qa_logs (session_id, app_id, user_query, final_response)
                      VALUES (%s, %s, %s, %s) \
                      """
                await cursor.execute(sql, (session_id, app_id, user_query, final_response))
        logger.info(f"成功将日志写入数据库。 Session: {session_id}")
    except Exception as e:
        logger.error(f"将会话 {session_id} 写入数据库时发生错误: {e}", exc_info=True)


# --- 【优化】问题转换函数 ---
def transform_user_prompt(original_prompt: str, score_level: str) -> str:
    """
    根据原始用户问题和score_level转换成更具体的、倾向性的问题。
    确保时间维度为“今天”。
    """
    logger.info(f"开始转换原始问题: '{original_prompt}' (Score Level: {score_level})")

    # 定义识别问题类别的正则表达式
    category_patterns = {
        "财富": r"财富|财运|金钱|收入|投资",
        "事业": r"事业|工作|学业|职业|晋升|项目",
        "感情": r"感情|爱情|姻缘|伴侣|对象|关系",
        "出行": r"出行|旅游|旅行|远行",
        "人际": r"人际|关系|朋友|同事|家人",
        "健康": r"健康|身体|精力",
    }

    detected_category = None
    for category, pattern in category_patterns.items():
        if re.search(pattern, original_prompt):
            detected_category = category
            break

    # 基础模板，不包含“今天”前缀
    templates_base = {
        "财富": {
            "0": "我的财运近期不顺，有什么具体的解决方案可以帮助我化解困境，提升财务安全感？",
            "1": "在财富方面运势一般般，我如何才能发现并抓住新的机遇，采取有效行动实现显著的财务增长？",
            "2": "我的财富运势很好，如何能够更好地迎接、保持并传导这份好运，吸引更丰盛的财富流入？"
        },
        "事业": {
            "0": "我的事业发展遇到阻碍，有什么务实的解决方案可以帮助我走出困境，获得职业稳定和安全感？",
            "1": "在事业上运势一般般，我如何才能洞察新的发展机会，提升我的能力和行动力以取得更大突破，实现职业目标？",
            "2": "我的事业运势很旺，如何能更好地把握这份好运，让事业持续高飞，创造更大的成功？"
        },
        "感情": {
            "0": "我的感情最近有些问题，有什么具体的行动方案可以帮助我化解误会，找到感情的稳定与和谐？",
            "1": "在感情关系中运势一般般，我如何才能发现新的契机，提升沟通和积极行动，让感情更加深入和发展？",
            "2": "我的感情运势非常好，如何能更好地享受和传导这份好运，让感情生活更加甜蜜幸福？"
        },
        "出行": {
            "0": "我计划出行，但有些顾虑，有什么具体的建议能确保旅途安全顺利，并避免潜在问题？",
            "1": "我计划出行，但运势一般般，如何能抓住旅途中的新体验和机会，让这次出行更有意义，拓展视野？",
            "2": "我即将出行，如何能迎接并分享这份好运，让旅途充满惊喜和美好的回忆？"
        },
        "人际": {
            "0": "我的人际关系遇到了一些难题，有什么务实的解决方案可以帮助我修复关系，获得和谐的人际环境？",
            "1": "在人际交往中运势一般般，我如何才能更好地洞察沟通机会，提升我的亲和力和行动力，建立更积极的关系？",
            "2": "我的人际运势很好，如何能更好地传导这份好运，让我在人际关系中更加受欢迎，获得更多支持？"
        },
        "健康": {
            "0": "我的健康状况有些担忧，有什么具体的调理方案可以帮助我改善身体，找到更健康的生活方式？",
            "1": "在健康方面运势一般般，我如何才能抓住提升体质的机会，采取更积极的行动，让身体更健康、更有活力？",
            "2": "我的健康运势很好，如何能更好地保持这份好运，让身体充满活力，享受健康的生活？"
        }
    }

    transformed_question = ""

    if detected_category:
        transformed_question = templates_base.get(detected_category, {}).get(score_level, "")
        print("transformed_question", transformed_question)

    # 如果通过类别和等级没有得到具体问题，或者原始问题比较泛泛（如“今日指引”），则使用通用模板
    if not transformed_question:
        if score_level == "0":
            transformed_question = "我最近感到困惑，希望得到今天具体的指引和务实的解决方案。我的情况如何，我该如何行动？"
        elif score_level == "1":
            transformed_question = "我希望得到今天具体的指引，如何抓住新的机会，提升行动力，实现更好的发展？"
        elif score_level == "2":
            transformed_question = "我希望得到今天具体的指引，如何更好地迎接并传导好运，让生活充满惊喜和丰盛？"

    print("transformed_question", transformed_question)
    # 确保问题以“今天”开头，并清除原始问题中的日期信息，避免重复
    is_today_explicitly_mentioned = re.search(r'(今日|今天|(\d{1,2}月\d{1,2}日))', original_prompt) is not None

    if not transformed_question.startswith("今天") and is_today_explicitly_mentioned:
        transformed_question = "今天" + transformed_question

    # 避免“今天今天”的重复
    transformed_question = re.sub(r'今天今天', '今天', transformed_question)

    logger.info(f"转换后的问题: '{transformed_question}'")
    return transformed_question.strip()


# --- 核心抽牌逻辑 ---
def _draw_and_get_card_data(
        cards_df: pd.DataFrame,
        meanings_df: pd.DataFrame,
        num_cards: int,
        card_number_pool: Optional[List[int]] = None
) -> tuple[Optional[tuple[str, ...]], Optional[tuple[int, ...]], Optional[tuple[str, ...]]]:
    """
    内部辅助函数：执行抽牌、获取牌名和牌意。
    """
    if cards_df is None or meanings_df is None:
        logger.error("错误: CSV 数据未加载。无法抽牌。")
        return None, None, None

    try:
        sampling_pool = cards_df["牌号"].tolist()
        if card_number_pool and len(card_number_pool) >= num_cards:
            all_valid_card_numbers_in_df = set(cards_df["牌号"].tolist())
            filtered_pool = [num for num in card_number_pool if num in all_valid_card_numbers_in_df]
            if len(filtered_pool) >= num_cards:
                sampling_pool = filtered_pool
                logger.info(f"使用提供的有效卡牌池进行抽样，池大小: {len(sampling_pool)}")

        if len(sampling_pool) < num_cards:
            logger.error(f"错误: 有效卡牌池不足 {num_cards} 张。")
            return None, None, None

        selected_card_numbers = random.sample(sampling_pool, num_cards)

        selected_cards_df = cards_df[cards_df["牌号"].isin(selected_card_numbers)].copy()
        if len(selected_cards_df) < num_cards:
            return None, None, None

        selected_cards_df["牌号_ordered"] = pd.Categorical(selected_cards_df["牌号"], categories=selected_card_numbers,
                                                           ordered=True)
        selected_cards_df = selected_cards_df.sort_values("牌号_ordered")
        selected_cards_data = selected_cards_df.to_dict(orient="records")

        card_numbers = tuple(card['牌号'] for card in selected_cards_data)
        card_names = tuple(card['卡牌'] for card in selected_cards_data)

        task_texts = []
        for card_name in card_names:
            meaning_row_df = meanings_df.loc[meanings_df['卡牌1'] == card_name]
            if not meaning_row_df.empty:
                meaning_columns = [col for col in meaning_row_df.columns if col.startswith('卡牌1的汇总')]
                valid_meanings = [str(meaning_row_df.iloc[0][col]) for col in meaning_columns if
                                  pd.notna(meaning_row_df.iloc[0][col])]
                task_texts.append(", ".join(valid_meanings) if valid_meanings else "N/A")
            else:
                task_texts.append("N/A")

        return card_names, card_numbers, tuple(task_texts)

    except Exception as e:
        logger.error(f"抽牌并获取数据时发生错误: {e}", exc_info=True)
        return None, None, None


# --- 【优化】生成提示词函数 (现在是唯一的占卜模式) ---
def generate_prompt_for_interpretation(
        cards_df: pd.DataFrame,
        meanings_df: pd.DataFrame,
        user_question: str,  # 这里现在接收的是转换后的问题，已包含“今天”的意图和领域倾向
        score_level: str,
        card_number_pool: Optional[List[int]] = None
) -> tuple[Optional[str], Optional[tuple[str, ...]], Optional[tuple[int, ...]]]:
    """
    为占卜问题生成雷诺曼占卜提示词。
    根据 score_level 提供不同风格的解读，确保围绕“今天”进行，并将牌意与用户问题类型紧密结合。
    """
    card_data = _draw_and_get_card_data(cards_df, meanings_df, 3, card_number_pool)
    if not all(card_data):
        logger.error("无法获取卡牌数据。")
        return None, None, None

    card_names, card_numbers, task_texts = card_data
    logger.info(f"占卜问题 (Level {score_level}): 选定卡牌 {card_names} (编号: {card_numbers})")

    llm_prompt = ""

    # 从 user_question 中尝试提取领域，用于提示词中的强调
    detected_category_match = re.search(
        r'(财运|财富|事业|工作|学业|感情|爱情|姻缘|人际|关系|朋友|出行|旅游|旅行|健康|身体)', user_question)
    print("detected_category_match", detected_category_match)
    print("user_question", user_question)
    domain_emphasis = f"您『{detected_category_match.group(0)}』方面" if detected_category_match else "您"
    print("domain_emphasis", domain_emphasis)

    if score_level == "0":
        # 低分解惑，提供安全感与解决方案 (务实、具体、有人情味，围绕“今天”，紧扣问题领域)
        llm_prompt = f"""
<prompt>
    <role>
    你是一位经验丰富、亲切体贴的雷诺曼占卜师，专注于**务实、具体地为用户提供“今天”的解决方案和安全感**。你擅长将用户当下感到困扰的问题拆解，用清晰、温暖的语言帮助他们理解今天在**{domain_emphasis}**的情况，并提供**具体、可执行且切合实际的应对策略**，就像一位贴心的朋友给出真诚的指导，帮助他们找到内心的平静和明确的方向，专注于**“今天”**在**{domain_emphasis}**的行动。
    </role>

    <input_data>
        <user_question>{user_question}</user_question>
        <card_spread>
            <card position="1" name="{card_names[0]}">
                <interpretation>{task_texts[0]}</interpretation>
            </card>
            <card position="2" name="{card_names[1]}">
                <interpretation>{task_texts[1]}</interpretation>
            </card>
            <card position="3" name="{card_names[2]}">
                <interpretation>{task_texts[2]}</interpretation>
            </card>
        </card_spread>
    </input_data>

    <analysis_framework>
        <rule name="时间限定">所有解读和建议必须严格限定在**“今天”**这个时间维度内。</rule>
        <rule name="领域聚焦">**将每一张牌的通用含义，严格在用户提问的『{user_question}』这个具体领域内进行精准转译和解读。务必确保所有解读都与{domain_emphasis}直接相关，避免泛泛而谈。**</rule>
        <rule name="单牌定位">每张牌的含义应与其在牌阵中的位置（如：核心、过程、结果）结合，形成针对“今天”在{domain_emphasis}的解读。</rule>
        <rule name="组合逻辑">分析牌与牌之间的相互影响和能量流动，特别是相邻牌和整体牌阵的叙事性，揭示“今天”在{domain_emphasis}事件的深层联系。</rule>
        <rule name="重复性质">关注是否存在重复的主题或相似的能量，这可能代表“今天”在{domain_emphasis}问题中的核心议题或潜在的强烈倾向。</rule>
        <rule name="深度解析维度">从心理、情感、物质、人际等多个维度进行深入分析，提供多层面的洞察，但**始终聚焦于今天在{domain_emphasis}可采取的实际行动和解决方案**。</rule>
        <rule name="问题拆解">将用户“今天”在{domain_emphasis}的问题分解为核心担忧点，并针对性地进行牌意解读。</rule>
        <rule name="安全感与掌控力">解读中注重识别用户今天在{domain_emphasis}可控的因素，通过提供明确的解决方案来增强用户的掌控感和内心的平静。</rule>
        <rule name="解决方案聚焦">**必须针对牌阵揭示的“今天”在{domain_emphasis}的问题，提供具体、清晰且具有高度可操作性的解决方案和步骤。**</rule>
        <rule name="务实性原则">所有解读和建议都应基于现实考量，避免空泛或不切实际的期望，确保用户今天在{domain_emphasis}能够实际应用。</rule>
    </analysis_framework>

    <output_instructions>
        <style>语气温暖、富有同情心，旨在抚慰你的情绪，给予你稳定感和希望。在提供建议时，语气应**务实而直接**，就像一位贴心的朋友给出真诚的指导，确保方案清晰明了，不做任何套话或虚假的鼓励。</style>
        <formatting_rules>
            <rule>在报告全文中，每当提到牌的名称时，必须使用Markdown语法将其**加粗**。</rule>
            <rule>每个主要部分都必须严格按照下面 `report_structure` 中定义的【格式示例】和【任务描述】进行渲染。</rule>
            <rule>报告内容中所有时间相关的描述都必须明确指向“今天”，并与{domain_emphasis}紧密结合。</rule>
        </formatting_rules>

        <report_structure>
            <section name="直接回答与核心判断">
                <task_description>
                **【核心任务：直击问题，给出明确结论】**
                针对您『{user_question}』的疑问，您抽到的三张牌是 **{card_names[0]}、{card_names[1]}、{card_names[2]}**（从左至右）。我会结合牌面，为您分析**今天在{domain_emphasis}**的情况并给出最实用的建议。
                </task_description>
                <format_example>
<![CDATA[
🔮 雷诺曼解读：
### 给您今天的指引：
**[此处给出精炼的核心判断标题，例如：“别担心，今天在{domain_emphasis}能找到走出困境的方法”，“今天在{domain_emphasis}虽然有挑战，但通过[具体行动]可有效规避风险”]**

]]>
                </format_example>
            </section>

            <section name="牌面解读">
                <task_description>
                此为报告核心。你必须严格遵循下面的结构和逻辑，**严禁给出模糊或描述性的结论,必须直言不讳结果！！**。

                **【占卜师的判断逻辑】**
                1.  **引言**：用一句话温暖地引入牌阵和用户问题，明确聚焦**今天**在**{domain_emphasis}**。
                2.  **基本含义**：逐一列出抽出的三张牌，并简要概括其核心含义。
                3.  **整体解读**：此部分为核心论证。你必须将三张牌进行两两组合，并为每个组合提供一个具体的解读，说明其如何共同影响**今天**在**{domain_emphasis}**的事件进程，并**直接关联到解决方案和务实应对**。
                </task_description>
                <format_example>
<![CDATA[

---

### 牌面解读与您今天在{domain_emphasis}的困惑：
您抽到的这三张牌，就像一面镜子，映照出您**今天**在**{domain_emphasis}**的困境与出路：
1. **{card_names[0]} → {card_names[1]}**：
    [这里我会为您深入分析第一张牌和第二张牌在**今天{domain_emphasis}**的联系。它可能意味着，您今天在{domain_emphasis}问题的根源是...，这导致了现在您面临的状况...。所以，我们需要务实地从...入手。]

2. **{card_names[1]} → {card_names[2]}**：
    [接着，我们会看看第二张牌和第三张牌在**今天{domain_emphasis}**的衔接。它告诉我们，您今天在{domain_emphasis}处理当前状况时，可能会遇到...，而要想顺利过渡到更好的结果，关键在于...。]

3. **{card_names[0]} + {card_names[2]}**：
    [最后，我们整体来看第一张牌和第三张牌在**今天{domain_emphasis}**的能量。它指明了，从今天在{domain_emphasis}事情的起点到最终的解决方向，需要您特别关注...，才能找到那个让您安心的务实方案。]
]]>
                </format_example>
            </section>

            <section name="建议与总结">
                <task_description>
                **核心任务**：基于你对本次牌阵揭示的所有信息，为用户提供**具体、务实、可操作性强**的解决方案和建议，并用一句话进行总结，所有建议都必须针对**“今天”**在**{domain_emphasis}**的问题。
                **设计原则**：
                1.  **建议**: 建议应与牌面解读紧密关联，侧重于提供安全感和解决问题的实际步骤，**务必具体到今天在{domain_emphasis}可执行的层面**。
                2.  **总结**: 总结语句应充满洞见和支持性，强调稳定和希望。
                </task_description>
                <format_example>
<![CDATA[
---

### 给您今天在{domain_emphasis}最实用的行动建议：
别担心，针对您**今天**在**{domain_emphasis}**的困扰，有几个小建议或许能帮到您：
- **具体行动方案**：[此处给出第一条针对核心问题或主要担忧的**今天在{domain_emphasis}可执行的步骤或策略**。例如，如果问题是财运不顺，可建议：**今天详细审视当前财务状况，列出收支清单，识别不必要的开支，并寻求专业人士的初步建议**。]
- **风险规避与准备**：[此处给出第二条关于如何规避今天在{domain_emphasis}潜在风险或做好应对准备的**务实建议**。例如：**今天如果遇到新的投资机会，务必进行初步的市场调研，不要轻信小道消息，保持谨慎**。]
- **心态调整与支持**：[此处给出第三条关于心理调适或寻求外部支持的**今天实用方法**，以增强安全感。例如：**今天可以与信任的朋友或家人沟通您在{domain_emphasis}的担忧，获得情感支持，避免独自承受压力**。]

🌟 总结：请记住，您拥有解决问题的力量。通过**今天**在**{domain_emphasis}**这些具体而务实的行动，您一定能有效应对挑战，重新找到内心的平静与掌控感。
]]>
                </format_example>
            </section>

            <section name="建议的追问方向">
                <task_description>
                **核心任务**: 基于您对**今天**在**{domain_emphasis}**情况和牌阵解读的理解，为您设计**三个**富有洞察力的、可以进一步占卜的追问问题。这些问题旨在帮助您更深入地思考**今天**在**{domain_emphasis}**的解决方案和未来的方向，务必是具体、可占卜的。
                **设计原则**:
                1.  **高度相关**: 每个问题都必须紧密围绕您的**今天**在**{domain_emphasis}**的困境 `{user_question}` 展开，是本次占卜的自然延伸。
                2.  **聚焦关键**: 问题应该指向本次牌阵中揭示的**今天**在**{domain_emphasis}**的关键点、风险点或潜在机遇，以及**今天具体的解决方案实施细节**。
                3.  **可操作性**: 问题应该是具体的、可占卜的，而不是宽泛的哲学问题。
                </task_description>
                <format_example>
<![CDATA[
@ 是否想问
- <**此处生成第一个追问问题**。例如：针对我**今天**在{domain_emphasis}面临的[具体问题]，牌阵中是否有更详细的建议行动步骤？>
- <**此处生成第二个追问问题**。例如：我应该如何评估和选择最适合我**今天**在{domain_emphasis}的[解决方案类型]，以确保其有效性，避免走弯路？>
- <**此处生成第三个追问问题**。例如：在实施[某个建议]时，我**今天**需要警惕哪些潜在的阻碍，并提前做哪些具体准备来应对？>
]]>
                </format_example>
            </section>

        </report_structure>

        <global_constraint>
        报告总字数控制在300字左右。
        </global_constraint>
    </output_instructions>

    <final_command>
    现在，请根据以上所有信息，为用户生成一份结构清晰、内容饱满、格式精美的占卜报告。请务必遵守每个部分的任务描述，特别是字数限制和提炼要求。请直接开始撰写报告，不要包含任何XML标签或指令性文字。
    </final_command>
</prompt>
"""
    elif score_level == "1":
        # 提升指南与机会洞察，提升行动力 (更具人情味，围绕“今天”，紧扣问题领域)
        llm_prompt = f"""
<prompt>
    <role>
    你是一位富有启发性、热情洋溢的雷诺曼占卜师，专注于**为用户提供“今天”的提升指南、洞察机会，并激发行动力**。你擅长从牌阵中发现成长的契机和潜在的可能性，鼓励用户积极主动地把握**今天**在**{domain_emphasis}**的机遇，勇敢地向前迈进，实现更高的目标。
    </role>

    <input_data>
        <user_question>{user_question}</user_question>
        <card_spread>
            <card position="1" name="{card_names[0]}">
                <interpretation>{task_texts[0]}</interpretation>
            </card>
            <card position="2" name="{card_names[1]}">
                <interpretation>{task_texts[1]}</interpretation>
            </card>
            <card position="3" name="{card_names[2]}">
                <interpretation>{task_texts[2]}</interpretation>
            </card>
        </card_spread>
    </input_data>

    <analysis_framework>
        <rule name="时间限定">所有解读和建议必须严格限定在**“今天”**这个时间维度内。</rule>
        <rule name="领域聚焦">**将每一张牌的通用含义，严格在用户提问的『{user_question}』这个具体领域内进行精准转译和解读。务必确保所有解读都与{domain_emphasis}直接相关，避免泛泛而谈。**</rule>
        <rule name="单牌定位">每张牌的含义应与其在牌阵中的位置（如：核心、过程、结果）结合，形成针对“今天”在{domain_emphasis}的解读。</rule>
        <rule name="组合逻辑">分析牌与牌之间的相互影响和能量流动，特别是相邻牌和整体牌阵的叙事性，揭示“今天”在{domain_emphasis}事件的深层联系。</rule>
        <rule name="重复性质">关注是否存在重复的主题或相似的能量，这可能代表“今天”在{domain_emphasis}问题中的核心议题或潜在的强烈倾向。</rule>
        <rule name="深度解析维度">从心理、情感、物质、人际等多个维度进行深入分析，提供多层面的洞察，但**始终聚焦于今天在{domain_emphasis}可把握的机遇和可采取的行动**。</rule>
        <rule name="潜力挖掘">解读牌意时，着重挖掘用户和事件今天可能展现的潜在优势和成长空间，**在{domain_emphasis}领域内**。</rule>
        <rule name="机会识别">明确指出牌阵中预示的今天即将出现的机遇、有利条件和可利用资源，**在{domain_emphasis}领域内**。</rule>
        <rule name="行动力激发">提供具体、激励性的行动建议，促使用户**今天**积极采取步骤，**在{domain_emphasis}领域内**。</rule>
        <rule name="目标导向">将牌意与用户达成更高目标结合，提供针对**今天**在**{domain_emphasis}**的战略性指导。</rule>
    </analysis_framework>

    <output_instructions>
        <style>语气积极、鼓舞人心、充满力量，就像一位智慧的导师，旨在激发你的自信和进取心，鼓励你勇于探索和突破**今天**在**{domain_emphasis}**的可能。</style>
        <formatting_rules>
            <rule>在报告全文中，每当提到牌的名称时，必须使用Markdown语法将其**加粗**。</rule>
            <rule>每个主要部分都必须严格按照下面 `report_structure` 中定义的【格式示例】和【任务描述】进行渲染。</rule>
            <rule>报告内容中所有时间相关的描述都必须明确指向“今天”，并与{domain_emphasis}紧密结合。</rule>
        </formatting_rules>

        <report_structure>
            <section name="直接回答与核心判断">
                <task_description>
                **【核心任务：直击问题，给出明确结论】**
                针对您『{user_question}』的疑问，您抽到的三张牌是 **{card_names[0]}、{card_names[1]}、{card_names[2]}**（从左至右）。我会结合牌面，为您分析**今天在{domain_emphasis}**的情况并给出最实用的建议。
                </task_description>
                <format_example>
<![CDATA[
🔮 雷诺曼解读：
### 给您今天的指引：
**[此处给出精炼的核心判断标题，例如：“大胆前行，今天在{domain_emphasis}机遇正向您招手”，“抓住今天在{domain_emphasis}的机会，您的努力会带来突破”]**

]]>
                </format_example>
            </section>

            <section name="牌面解读">
                <task_description>
                此为报告核心。你必须严格遵循下面的结构和逻辑，**严禁给出模糊或描述性的结论,必须直言不讳结果！！**。

                **【占卜师的判断逻辑】**
                1.  **引言**：用一句话积极地引入牌阵和用户问题，明确聚焦**今天**在**{domain_emphasis}**。
                2.  **基本含义**：逐一列出抽出的三张牌，并简要概括其核心含义。
                3.  **整体解读**：此部分为核心论证。你必须将三张牌进行两两组合，并为每个组合提供一个具体的解读，说明其如何共同影响**今天**在**{domain_emphasis}**事件的进程，**侧重于如何将今天的挑战转化为机遇，激发潜能，并引导积极的行动**。
                </task_description>
                <format_example>
<![CDATA[

---

### 牌面解读与您今天在{domain_emphasis}的机遇：
您抽到的这三张牌，就像一张行动地图，为您指明了您**今天**在**{domain_emphasis}**的成长路径：
1. **{card_names[0]} → {card_names[1]}**：
    [这里我会为您深入分析第一张牌和第二张牌在**今天**针对**{domain_emphasis}**的联系。它暗示着，您**今天**在**{domain_emphasis}**初期所具备的优势或面临的挑战，正是在促使您向着...的方向前进。抓住**今天**这个转化点，您将看到新的进展。]

2. **{card_names[1]} → {card_names[2]}**：
    [接着，我们会看看第二张牌和第三张牌在**今天**针对**{domain_emphasis}**的衔接。它告诉我们，您**今天**在**{domain_emphasis}**行动过程中所做的努力和决策，将直接影响到最终的成果。聚焦于...，您会发现突破口。]

3. **{card_names[0]} + {card_names[2]}**：
    [最后，我们整体来看第一张牌和第三张牌在**今天**针对**{domain_emphasis}**的能量。它预示着，从**今天**在**{domain_emphasis}**的萌芽到最终的成就，有一股强大的能量在支持着您。积极利用...，您将成功抵达目标。]
]]>
                </format_example>
            </section>

            <section name="建议与总结">
                <task_description>
                **核心任务**：基于你对本次牌阵揭示的所有信息，为用户提供具体、可执行的、**能激发行动力和把握“今天”在{domain_emphasis}机会**的建议，并用一句话进行总结。
                **设计原则**：
                1.  **建议**: 建议应与牌面解读紧密关联，侧重于提升**今天**在**{domain_emphasis}**的行动力、抓住**今天**在**{domain_emphasis}**的机会和克服**今天**在**{domain_emphasis}**的障碍。
                2.  **总结**: 总结语句应充满洞见和支持性，强调成长和成就。
                </task_description>
                <format_example>
<![CDATA[
---

### 给您今天在{domain_emphasis}最积极的行动建议：
您**今天**的牌面充满了积极的能量，为了更好地把握这些机会，这里有几点建议助您一臂之力：
- **主动出击，抓住机遇**：[此处给出第一条具体建议，侧重于如何主动把握**今天**在**{domain_emphasis}**牌阵揭示的机遇。例如：**不要犹豫，今天主动联系您在{domain_emphasis}方面感兴趣的导师或合作伙伴，他们的经验和资源会是您宝贵的助力**。]
- **积极应对，化解挑战**：[此处给出第二条具体建议，侧重于如何克服**今天**在**{domain_emphasis}**潜在挑战，将其转化为动力。例如：**今天在面对{domain_emphasis}方面的决策时，可以多方收集信息，但最终要相信自己的直觉，果断做出选择，避免拖延**。]
- **拓展视野，持续学习**：[此处给出第三条具体建议，侧重于**今天**在**{domain_emphasis}**可以采取的、能带来显著进步的行动。例如：**今天投入时间学习一项新的{domain_emphasis}相关技能或拓展人脉圈，这会为您的未来发展打开新的大门**。]

🌟 总结：您**今天**在**{domain_emphasis}**正处于一个充满无限可能的时期！勇敢地迈出每一步，积极把握每一个机会，您将创造出超乎想象的精彩成就。加油！
]]>
                </format_example>
            </section>

            <section name="建议的追问方向">
                <task_description>
                **核心任务**: 基于您对**今天**在**{domain_emphasis}**情况和牌阵解读的理解，为您设计**三个**富有洞察力的、可以进一步占卜的追问问题。这些问题旨在帮助您更深入地探索**今天**在**{domain_emphasis}**的机会和行动策略，务必是具体、可占卜的。
                **设计原则**:
                1.  **高度相关**: 每个问题都必须紧密围绕您的**今天**在**{domain_emphasis}**的困境 `{user_question}` 展开，是本次占卜的自然延伸。
                2.  **聚焦关键**: 问题应该指向本次牌阵中揭示的**今天**在**{domain_emphasis}**的关键点、风险点或潜在机遇，以及**今天具体的行动策略**。
                3.  **可操作性**: 问题应该是具体的、可占卜的，而不是宽泛的哲学问题。
                </task_description>
                <format_example>
<![CDATA[
@ 是否想问
- <**此处生成第一个追问问题**。例如：我应该如何准备，才能更好地抓住**今天**在**{domain_emphasis}**即将出现的关键机遇？>
- <**此处生成第二个追问问题**。例如：在**今天**在**{domain_emphasis}**的行动中，哪些潜在的障碍是我需要特别关注并提前制定应对策略的？>
- <**此处生成第三个追问问题**。例如：为了更快实现**{domain_emphasis}**的[某个目标]，我**今天**最需要提升或学习的关键技能是什么？>
]]>
                </format_example>
            </section>

        </report_structure>

        <global_constraint>
        报告总字数控制在300字左右。
        </global_constraint>
    </output_instructions>

    <final_command>
    现在，请根据以上所有信息，为用户生成一份结构清晰、内容饱满、格式精美的占卜报告。请务必遵守每个部分的任务描述，特别是字数限制和提炼要求。请直接开始撰写报告，不要包含任何XML标签或指令性文字。
    </final_command>
</prompt>
"""
    elif score_level == "2":
        # 好运传导 (更具人情味，围绕“今天”，紧扣问题领域)
        llm_prompt = f"""
<prompt>
    <role>
    你是一位充满阳光、传递好运的雷诺曼占卜师，专注于**为用户传递“今天”的好运和祝福**。你的解析深刻、精准，并能根据用户的问题和抽出的牌，提供富有洞见和同理心的指导。你的报告不仅内容充实，格式也极为优雅、清晰。你的风格是好运传导，擅长从牌阵中发现一切积极、吉祥的征兆，将好运和祝福传递给用户，帮助他们建立信心，吸引丰盛，并以乐观积极的心态迎接**今天**在**{domain_emphasis}**的美好。
    </role>

    <input_data>
        <user_question>{user_question}</user_question>
        <card_spread>
            <card position="1" name="{card_names[0]}">
                <interpretation>{task_texts[0]}</interpretation>
            </card>
            <card position="2" name="{card_names[1]}">
                <interpretation>{task_texts[1]}</interpretation>
            </card>
            <card position="3" name="{card_names[2]}">
                <interpretation>{task_texts[2]}</interpretation>
            </card>
        </card_spread>
    </input_data>

    <analysis_framework>
        <rule name="时间限定">所有解读和建议必须严格限定在**“今天”**这个时间维度内。</rule>
        <rule name="领域聚焦">**将每一张牌的通用含义，严格在用户提问的『{user_question}』这个具体领域内进行精准转译和解读。务必确保所有解读都与{domain_emphasis}直接相关，避免泛泛而谈。**</rule>
        <rule name="单牌定位">每张牌的含义应与其在牌阵中的位置（如：核心、过程、结果）结合，形成针对“今天”在{domain_emphasis}的解读。</rule>
        <rule name="组合逻辑">分析牌与牌之间的相互影响和能量流动，特别是相邻牌和整体牌阵的叙事性，揭示“今天”在{domain_emphasis}事件的深层联系。</rule>
        <rule name="重复性质">关注是否存在重复的主题或相似的能量，这可能代表“今天”在{domain_emphasis}问题中的核心议题或潜在的强烈倾向。</rule>
        <rule name="深度解析维度">从心理、情感、物质、人际等多个维度进行深入分析，提供多层面的洞察，但**始终聚焦于今天在{domain_emphasis}可能发生的好运和丰盛**。</rule>
        <rule name="积极视角">对所有牌意都从积极、乐观的角度进行解读，强调其为**今天**在**{domain_emphasis}**带来的好运和福泽。</rule>
        <rule name="祝福传递">语言中充满祝福、鼓励和正向引导，强化**今天**在**{domain_emphasis}**好运的信念。</rule>
        <rule name="丰盛吸引">指出如何通过积极心态和行动吸引**今天**在**{domain_emphasis}**更多好运和丰盛的能量。</rule>
        <rule name="感恩与肯定">鼓励用户感恩**今天**在**{domain_emphasis}**已有的好运，并肯定他们积极面对未来的能力。</rule>
    </analysis_framework>

    <output_instructions>
        <style>语气热情洋溢、充满喜悦和真诚的祝福，旨在感染你，让你充分感受到好运的到来，并用积极的心态去迎接和创造**今天**在**{domain_emphasis}**更多美好。</style>
        <formatting_rules>
            <rule>在报告全文中，每当提到牌的名称时，必须使用Markdown语法将其**加粗**。</rule>
            <rule>每个主要部分都必须严格按照下面 `report_structure` 中定义的【格式示例】和【任务描述】进行渲染。</rule>
            <rule>报告内容中所有时间相关的描述都必须明确指向“今天”，并与{domain_emphasis}紧密结合。</rule>
        </formatting_rules>

        <report_structure>
            <section name="直接回答与核心判断">
                <task_description>
                **【核心任务：直击问题，给出明确结论】**
                针对您『{user_question}』的疑问，您抽到的三张牌是 **{card_names[0]}、{card_names[1]}、{card_names[2]}**（从左至右）。我会结合牌面，为您分析**今天**在**{domain_emphasis}**的情况并给出最美好的指引。
                </task_description>
                <format_example>
<![CDATA[
🔮 雷诺曼解读：
### 给您今天的指引：
**[此处给出精炼的核心判断标题，例如：“太棒了！今天在{domain_emphasis}好运正降临，一切顺遂”，“福气满满，今天在{domain_emphasis}将充满惊喜”]**

]]>
                </format_example>
            </section>

            <section name="牌面解读">
                <task_description>
                此为报告核心。你必须严格遵循下面的结构和逻辑，**严禁给出模糊或描述性的结论,必须直言不讳结果！！**。

                **【占卜师的判断逻辑】**
                1.  **引言**：用一句话积极地引入牌阵和用户问题，明确聚焦**今天**在**{domain_emphasis}**。
                2.  **基本含义**：逐一列出抽出的三张牌，并简要概括其核心含义。
                3.  **整体解读**：此部分为核心论证。你必须将三张牌进行两两组合，并为每个组合提供一个具体的解读，说明其如何共同影响**今天**在**{domain_emphasis}**事件的进程，**着重强调今天在{domain_emphasis}的好运的流动、机遇的到来和积极的结果**。
                </task_description>
                <format_example>
<![CDATA[

---

### 牌面解读与您今天在{domain_emphasis}的好运：
您抽到的这三张牌，正为您描绘了一幅美好的画卷，预示着**今天**在**{domain_emphasis}**『{user_question}』的积极走向：
1. **{card_names[0]} → {card_names[1]}**：
    [这里我会为您深入分析第一张牌和第二张牌在**今天**针对**{domain_emphasis}**的联系。它告诉我们，您**今天**在**{domain_emphasis}**的开端就充满了吉兆，这份积极的能量将顺利推动您在...方面获得进展，好运连连！]

2. **{card_names[1]} → {card_names[2]}**：
    [接着，我们会看看第二张牌和第三张牌在**今天**针对**{domain_emphasis}**的衔接。它预示着，在您积极的行动下，**今天**在**{domain_emphasis}**的事情会自然而然地向着更好的方向发展，甚至会有意想不到的惊喜，最终达到圆满的结局。]

3. **{card_names[0]} + {card_names[2]}**：
    [最后，我们整体来看第一张牌和第三张牌在**今天**针对**{domain_emphasis}**的能量。这组合意味着，从**今天**在**{domain_emphasis}**开始到最终，您都被好运环绕，一切都会顺心如意，丰盛的能量将持续伴随您！]
]]>
                </format_example>
            </section>

            <section name="建议与总结">
                <task_description>
                **核心任务**：基于你对本次牌阵揭示的所有信息，为用户提供具体、可执行的、**能帮助他们吸引、保持和分享今天在{domain_emphasis}好运**的建议，并用一句话进行总结。
                **设计原则**：
                1.  **建议**: 建议应与牌面解读紧密关联，侧重于如何吸引并保持**今天**在**{domain_emphasis}**好运、感恩和积极心态。
                2.  **总结**: 总结语句应充满洞见和支持性，强调好运和祝福。
                </task_description>
                <format_example>
<![CDATA[
---

### 给您今天在{domain_emphasis}最美好的祝福与好运提示：
恭喜您，牌面显示您的好运正当时！为了让这份美好的能量持续下去，这里有几点温馨提示给您：
- **积极接纳，感恩生活**：[此处给出第一条具体建议，侧重于如何感恩并放大**今天**在**{domain_emphasis}**的好运。例如：**保持一颗感恩的心，对身边**今天**在**{domain_emphasis}**的小确幸多一份留意和珍惜，这会吸引更多美好的事物进入您的生活**。]
- **保持乐观，传递喜悦**：[此处给出第二条具体建议，侧重于如何保持积极乐观的心态，吸引更多丰盛。例如：**用积极乐观的心态去面对**今天**在**{domain_emphasis}**的每一天，并尝试将您的好心情和正能量分享给身边的人，喜悦是会传染的哦**！]
- **大胆尝试，顺势而为**：[此处给出第三条具体建议，侧重于如何将这份好运分享给他人，形成良性循环。例如：**如果今天在**{domain_emphasis}**有什么一直想尝试的事情，现在正是顺势而为的好时机，勇敢迈出一步，可能会有惊喜！**]

🌟 总结：您是如此幸运！请尽情享受**今天**在**{domain_emphasis}**这份美好的运势，相信并接纳所有的好，未来定会更加丰盛与精彩！祝您**今天**充满阳光与喜悦！
]]>
                </format_example>
            </section>

            <section name="建议的追问方向">
                <task_description>
                **核心任务**: 基于您对**今天**在**{domain_emphasis}**情况和牌阵解读的理解，为您设计**三个**富有洞察力的、可以进一步占卜的追问问题。这些问题旨在帮助您更深入地探索如何吸引和保持**今天**在**{domain_emphasis}**的好运，务必是具体、可占卜的。
                **设计原则**:
                1.  **高度相关**: 每个问题都必须紧密围绕您的**今天**在**{domain_emphasis}**的困境 `{user_question}` 展开，是本次占卜的自然延伸。
                2.  **聚焦关键**: 问题应该指向本次牌阵中揭示的**今天**在**{domain_emphasis}**的关键点、风险点或潜在机遇，以及**如何放大好运的具体方法**。
                3.  **可操作性**: 问题应该是具体的、可占卜的，而不是宽泛的哲学问题。
                </task_description>
                <format_example>
<![CDATA[
@ 是否想问
- <**此处生成第一个追问问题**。例如：**今天**我应该如何在**{domain_emphasis}**投入精力，才能最大化地吸引和巩固这份好运？>
- <**此处生成第二个追问问题**。例如：在未来几天，我最有可能在**{domain_emphasis}**哪个意想不到的领域收到好消息或惊喜？>
- <**此处生成第三个追问问题**。例如：为了更好地保持这份积极的能量，我**今天**在**{domain_emphasis}**可以做些什么来提升自己的幸福感和满足感？>

]]>
                </format_example>
            </section>

        </report_structure>

        <global_constraint>
        报告总字数控制在300字左右。
        </global_constraint>
    </output_instructions>

    <final_command>
    现在，请根据以上所有信息，为用户生成一份结构清晰、内容饱满、格式精美的占卜报告。请务必遵守每个部分的任务描述，特别是字数限制和提炼要求。请直接开始撰写报告，不要包含任何XML标签或指令性文字。
    </final_command>
</prompt>
"""
    else:
        logger.error(f"无效的 score_level: {score_level}")
        return None, None, None

    return llm_prompt, card_names, card_numbers


# --- VLLM 流式响应函数 (不变) ---
async def stream_vllm_response_with_retry(prompt: str, request_id: int) -> AsyncGenerator[str, None]:
    global async_aiohttp_client
    if async_aiohttp_client is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="错误: 服务内部错误，客户端未准备好。")

    messages = [{"role": "user", "content": prompt}]
    llm_payload = {"model": VLLM_MODEL_NAME, "messages": messages, "temperature": 0.3, "stream": True}
    headers = {'Content-Type': 'application/json'}
    attempt = 0
    last_error_detail = "未知错误"

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
                                logger.info(f"请求 {request_id}: 收到 VLLM 流结束信号 [DONE]。")
                                return
                            try:
                                payload = json.loads(data_json_str)
                                content = payload.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if content:
                                    yield content
                            except (json.JSONDecodeError, IndexError):
                                logger.warning(f"请求 {request_id}: 无法解析 VLLM 数据块: {data_json_str}")
                                continue

                logger.info(f"请求 {request_id}: VLLM 响应流结束。")
                return

        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            last_error_detail = f"网络或客户端错误: {type(e).__name__} - {e}"
            logger.error(f"请求 {request_id}: 调用 VLLM 失败 (尝试 {attempt + 1}): {last_error_detail}")
            attempt += 1
        except Exception as e:
            logger.critical(f"请求 {request_id}: 调用 VLLM 期间发生未知错误: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"调用 LLM 服务期间发生意外错误: {e}")

    logger.error(f"请求 {request_id}: 达到最大 VLLM 重试次数，最终失败。最后错误: {last_error_detail}")
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"错误: 调用 LLM 服务失败，已重试 {VLLM_MAX_RETRIES} 次。")

# --- FastAPI 接口端点 (核心改造，集成问题转换) ---
# --- FastAPI 接口端点 (核心改造，集成问题转换) ---
@app.post("/chat_daily_leipai")
async def chat_endpoint(client_request: ClientRequest, request: Request):
    global next_request_id_counter
    req_id = client_request.session_id if client_request.session_id else f"req-{next_request_id_counter}"
    next_request_id_counter += 1

    db_pool_from_state = getattr(request.app.state, 'db_pool', None)

    score_level = client_request.score_level  # 获取 score_level
    original_user_prompt = client_request.prompt  # 存储原始用户问题用于日志

    try:
        # 1. 签名验证
        logger.info(
            f"请求 {req_id}: 收到新请求, AppID: '{client_request.appid}', 原始问题: '{original_user_prompt}', ScoreLevel: '{score_level}'")
        app_secret = APP_SECRETS.get(client_request.appid)
        if not app_secret:
            raise HTTPException(status_code=401, detail="未授权: 无效的 AppID。")

        params_for_sig = client_request.dict(exclude={'sign', 'card_number_pool', 'hl_ymd'})

        if generate_signature(params_for_sig, app_secret) != client_request.sign:
            raise HTTPException(status_code=403, detail="禁止访问: 签名验证失败。")
        logger.info(f"请求 {req_id}: 签名验证成功。")

        # 2. **问题转换**
        transformed_user_question = transform_user_prompt(original_user_prompt, score_level)
        logger.info(f"请求 {req_id}: 转换后的问题: '{transformed_user_question}'")

        # 3. **生成占卜提示词**
        prompt_data = generate_prompt_for_interpretation(
            renomann_cards_df, renomann_meanings_df, transformed_user_question, score_level,
            client_request.card_number_pool
        )
        prompt_text, card_names, card_numbers = prompt_data

        if not prompt_text:
            raise HTTPException(status_code=500, detail="生成最终占卜提示词失败。")

        # 4. **流式返回占卜结果**
        logger.info(f"请求 {req_id}: 提示词生成完毕，准备进行占卜解读并流式返回。")

        async def generate_stream_and_log():
            response_parts = []
            final_response = ""
            try:
                async with asyncio.timeout(VLLM_SLOT_WAIT_TIMEOUT_SECONDS):
                    async with vllm_semaphore:
                        async for chunk in stream_vllm_response_with_retry(prompt_text, req_id):
                            response_parts.append(chunk)
                            yield chunk
                    yield "[DONE]"

                final_response = "".join(response_parts)

            except Exception as e:
                logger.error(f"请求 {req_id}: 流式响应生成时发生错误: {e}", exc_info=True)
                yield f"错误: 流式响应时发生内部错误。\n"
            finally:
                session_id_to_log = f"{req_id}"
                app_id_to_log = client_request.appid if client_request.appid else "unknown_app"

                # 数据库日志记录使用原始问题
                asyncio.create_task(
                    log_qa_to_db(
                        db_pool=db_pool_from_state,
                        session_id=session_id_to_log,
                        app_id=app_id_to_log,
                        user_query=original_user_prompt,  # 记录原始问题
                        final_response=final_response
                    )
                )

        return StreamingResponse(generate_stream_and_log(), media_type="text/plain; charset=utf-8")

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.critical(f"请求 {req_id}: 处理请求时发生未知严重错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}")


if __name__ == "__main__":
    uvicorn.run(app="__main__:app", host="0.0.0.0", port=8001,reload=True)