import uvicorn
import os
import uuid
import json
import re
import traceback
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Literal, Optional, AsyncGenerator
from typing import List, Dict, Tuple, Optional
import asyncio
import hmac
import hashlib
from zhdate import ZhDate
import logging
import random
import pandas as pd
import requests
import aiohttp
from tokenizer import count_tokens_for_messages, count_tokens_for_string
from typing import Tuple, Dict, Any, AsyncGenerator
from collections import defaultdict
from urllib.parse import urlparse
from dotenv import load_dotenv
import calendar
from lunardate import LunarDate
from fastapi import FastAPI, HTTPException, Depends, Body
from pydantic import BaseModel, Field, ValidationError, field_validator
from prometheus_client import Counter, make_asgi_app
import prometheus_client
from monitor import StepMonitor, log_step, generate_request_id, monitor_logger

import db_manager2 as db_manager

# --- 本地辅助函数导入 (假设这些文件存在) ---
from tiangan_function import parse_bazi_components, replace_key_names, key_mapping, ziweids2, \
    describe_ziwei_chart
from liushifunction import key_translation_map, value_translation_map, translate_json, zhuxing, ziweids_conzayao

from ziwei_ai_function import transform_horoscope_scope_data, transform_palace_data_new, \
    DI_ZHI_ORDER, DIZHI_DUIGONG_MAP, TRADITIONAL_HOUR_TO_TIME_INDEX, transform_palace_data, \
    parse_palace_data, calculate_score_by_rules, get_lunar_month_range_string, \
    get_judgment_word_bank_for_score, generate_score_based_judgment, _parse_lenient_json

from xingyaoxingzhi import preprocess_chart, analyze_chart_final_optimized, PALACE_ORDER, perform_astrological_analysis

from utils import (
    calculate_time_index,
)


# --- 日志配置 ---
prometheus_client.REGISTRY.unregister(prometheus_client.GC_COLLECTOR)
prometheus_client.REGISTRY.unregister(prometheus_client.PLATFORM_COLLECTOR)
prometheus_client.REGISTRY.unregister(prometheus_client.PROCESS_COLLECTOR)

REQUESTS_RECEIVED = Counter(
    "api_requests_received_total",
    "Total number of requests received at the /chat_year endpoint."
)

VLLM_REQUESTS_SENT_ATTEMPTS = Counter(
    "vllm_requests_sent_attempts_total",
    "Total attempts to send requests to VLLM after acquiring the semaphore."
)

VLLM_RESPONSES_SUCCESS = Counter(
    "vllm_responses_success_total",
    "Total number of successful and complete responses from VLLM."
)

VLLM_RESPONSES_FAILED = Counter(
    "vllm_responses_failed_total",
    "Total number of failed/errored responses from VLLM.",
    ["reason"]
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))  # 默认INFO，若需更详细日志可设为DEBUG
logger = logging.getLogger(__name__)

# --- 配置加载 ---
load_dotenv()

VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "qwen3-next-80b-a3b-instruct")
API_KEY = os.getenv("API_KEY","sk-81d4dbe056f94030998f0639f709bff4")

ZIWEI_API_URL = os.getenv("ZIWEI_API_URL","http://192.168.1.102:3000/astro_with_option")

VLLM_REQUEST_TIMEOUT_SECONDS = 100
ZIWEI_API_TIMEOUT_SECONDS = 250
VLLM_CONCURRENT_LIMIT = 100
VLLM_SLOT_WAIT_TIMEOUT_SECONDS = 500


async_aiohttp_client: Optional[aiohttp.ClientSession] = None
vllm_semaphore = asyncio.Semaphore(VLLM_CONCURRENT_LIMIT)
logger.info(f"VLLM并发访问限制已设置为: {VLLM_CONCURRENT_LIMIT}")

excel_path = r'xyhs.xlsx'
sheet_name = '组合性质判断'
rules_df = None  # 将在启动时异步加载

def _load_rules_df_sync():
    """同步加载规则DataFrame的辅助函数"""
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, keep_default_na=False)
        print(f"成功从 '{excel_path}' 的 '{sheet_name}' 工作表中加载 {len(df)} 条规则。")
        
        df['来源'] = '原始'
        new_rules = []
        for index, rule in df.iterrows():
            zayao_content = str(rule.get('杂曜', ''))
            if (zayao_content.strip() and
                    not str(rule.get('会照', '')).strip() and
                    not str(rule.get('会和', '')).strip() and
                    not str(rule.get('对照', '')).strip()):
                new_rule = rule.copy()
                new_rule['会照'] = new_rule['杂曜']
                new_rule['杂曜'] = ''
                new_rule['来源'] = '杂曜衍生'
                new_rules.append(new_rule)
        if new_rules:
            new_rules_df = pd.DataFrame(new_rules)
            df = pd.concat([df, new_rules_df], ignore_index=True)
            print(f"根据'杂曜'规则衍生出 {len(new_rules)} 条新的'会照'规则。总规则数变为: {len(df)}")
        return df
    except Exception as e:
        print(f"加载Excel文件时出错: {e}")
        raise

ALL_PALACES = ["命宫", "兄弟宫", "夫妻宫", "子女宫", "财帛宫", "疾厄宫", "迁移宫", "交友宫", "事业宫", "田宅宫",
               "福德宫", "父母宫"]

# 宫位到维度类型的映射（用于JSON输出）
PALACE_TO_DIMENSION_TYPE = {
    # "命宫": "整体",
    "事业宫": "事业",
    "财帛宫": "财富",
    "夫妻宫": "感情",
    "疾厄宫": "健康",
    "交友宫": "人际",
    "迁移宫": "出行"
}

# --- Pydantic模型 (为请求体新增签名所需字段) ---
class APIRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="会话ID")
    gender: str = Field(..., description="性别：'男' 或 '女'")
    birthday: str = Field(..., description="出生日期时间，格式：'YYYY-MM-DD HH:MM:SS'，例如：'1995-05-06 14:30:00'")


# 在 parse_fixed_birth_info 函数之后添加
def parse_birthday_string(birthday: str) -> Dict[str, Any]:
    """
    从生日字符串中解析出生信息。
    格式: "YYYY-MM-DD HH:MM:SS"，例如 "1995-05-06 14:30:00"
    """
    try:
        # 解析日期时间字符串
        dt = datetime.strptime(birthday, "%Y-%m-%d %H:%M:%S")
        birth_info = {
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            "minute": dt.minute,
            "second": dt.second,
            "is_lunar": False,  # 默认公历
            "traditional_hour_branch": None
        }
        return birth_info
    except ValueError as e:
        raise ValueError(f"生日格式错误，请使用 'YYYY-MM-DD HH:MM:SS' 格式。错误: {e}")


# --- 核心分析函数 (已简化和去除历史依赖) ---
def get_cross_level_interaction_analysis(current_branch: str, current_disk_data: dict, previous_disk_data: dict,
                                         current_disk_name: str, previous_disk_name: str) -> str:
    """执行跨层叠盘分析，检测星曜的成对和加强。"""
    current_idx = DI_ZHI_ORDER.index(current_branch)
    sfsz_branches_indices = {
        current_idx, (current_idx + 4) % 12, (current_idx - 4 + 12) % 12, (current_idx + 6) % 12
    }

    current_level_stars = defaultdict(list)
    previous_level_stars = defaultdict(list)
    current_level_sihua = defaultdict(list)
    previous_level_sihua = defaultdict(list)

    for branch_idx in sfsz_branches_indices:
        branch = DI_ZHI_ORDER[branch_idx]

        if branch in current_disk_data:
            info = current_disk_data[branch]
            all_aux = (info.get('辅星', '') + ',' + info.get('煞星及杂曜', '')).split(',')
            for star in all_aux:
                star = star.strip()
                if star: current_level_stars[star].append(f"{branch}宫")
            if info.get('四化'):
                sihua_type = "化" + info.get('四化', '')[-1]
                current_level_sihua[sihua_type].append(f"{branch}宫")

        if branch in previous_disk_data:
            info = previous_disk_data[branch]
            all_aux = (info.get('辅星', '') + ',' + info.get('煞星及杂曜', '')).split(',')
            for star in all_aux:
                star = star.strip()
                if star: previous_level_stars[star].append(f"{branch}宫")
            if info.get('四化'):
                sihua_type = "化" + info.get('四化', '')[-1]
                previous_level_sihua[sihua_type].append(f"{branch}宫")

    output_parts = []

    reinforced_stars_info = []
    common_stars = set(current_level_stars.keys()) & set(previous_level_stars.keys())
    for star in sorted(list(common_stars)):
        reinforced_stars_info.append(f"【{star}】")
    if reinforced_stars_info:
        output_parts.append(f"      - 同星叠加: {', '.join(reinforced_stars_info)}，力量得到加强。")

    sihua_interaction = []
    for sihua_type in ['化禄', '化权', '化科', '化忌']:
        count = len(current_level_sihua.get(sihua_type, [])) + len(previous_level_sihua.get(sihua_type, []))
        if count > 1:
            sihua_interaction.append(f"【双{sihua_type[-1]}】")
    if sihua_interaction:
        output_parts.append(f"      - 四化叠加: 见{', '.join(sihua_interaction)}，吉凶效应加剧。")

    if not output_parts:
        return ""

    return f"    - 叠盘影响 ({previous_disk_name.replace('盘', '')}冲{current_disk_name.replace('盘', '')}):\n" + "\n".join(
        output_parts)


def get_comprehensive_sfsz_info(current_branch: str, disk_data_by_branch: dict) -> str:
    current_idx = DI_ZHI_ORDER.index(current_branch)
    san_he_branches = [(current_idx + 4) % 12, (current_idx - 4 + 12) % 12]
    dui_gong_branch_idx = (current_idx + 6) % 12
    palace_positions = {"本宫": disk_data_by_branch.get(current_branch),
                        "三合宫位A": disk_data_by_branch.get(DI_ZHI_ORDER[san_he_branches[0]]),
                        "三合宫位B": disk_data_by_branch.get(DI_ZHI_ORDER[san_he_branches[1]]),
                        "对宫": disk_data_by_branch.get(DI_ZHI_ORDER[dui_gong_branch_idx]), }
    all_stars_in_sfsz = set()
    for pos_info in palace_positions.values():
        if pos_info:
            stars_to_add = (pos_info.get('主星', '') + ',' + pos_info.get('辅星', '') + ',' + pos_info.get('煞星及杂曜',
                                                                                                           ''))
            for star in stars_to_add.split(','):
                if star.strip(): all_stars_in_sfsz.add(star.strip())
    output_parts = []

    sfsz_details = []
    for pos_name, pos_info in palace_positions.items():
        if pos_name == "本宫" or not pos_info: continue
        detail_line = f"  - {pos_name} ({pos_info['地支']}-{pos_info['宫位']}) 为【{pos_info['主星']}】"
        all_other_stars = [s.strip() for s in (pos_info['辅星'] + ',' + pos_info['煞星及杂曜']).split(',') if s.strip()]
        if all_other_stars: detail_line += f"，带有【{', '.join(all_other_stars)}】"
        if pos_info['四化']: detail_line += f"，并见【{pos_info['四化']}】"
        sfsz_details.append(detail_line)
    if sfsz_details: output_parts.append("    - 三方四正详情:\n" + "\n".join(sfsz_details))
    return "\n".join(output_parts)


def check_tianxiang_jia_gong(prev_info: dict, next_info: dict) -> list:
    if not prev_info or not next_info: return []
    results, prev_main_stars, next_main_stars, prev_sihua, next_sihua = [], prev_info.get('主星', '').split(
        ','), next_info.get('主星', '').split(','), prev_info.get('四化', ''), next_info.get('四化', '')
    is_cai_yin = (('天梁' in prev_main_stars and ('天同化禄' in next_sihua or '巨门化禄' in next_sihua)) or (
            '天梁' in next_main_stars and ('天同化禄' in prev_sihua or '巨门化禄' in prev_sihua)))
    if is_cai_yin: results.append("    - 格局: 构成【财荫夹印】，主福气、庇荫，易得贵人相助或财务机遇。")
    is_xing_ji = (('天梁' in prev_main_stars and ('天同化忌' in next_sihua or '巨门化忌' in next_sihua)) or (
            '天梁' in next_main_stars and ('天同化忌' in prev_sihua or '巨门化忌' in prev_sihua)))
    if is_xing_ji: results.append("    - 格局: 构成【刑忌夹印】，主内心纠结、人际压力、或因他人之事而生困扰。")
    prev_all_stars, next_all_stars = prev_info.get('辅星', '') + ',' + prev_info.get('煞星及杂曜', ''), next_info.get(
        '辅星', '') + ',' + next_info.get('煞星及杂曜', '')
    is_kong_jie = (('地空' in prev_all_stars and '地劫' in next_all_stars) or (
            '地劫' in prev_all_stars and '地空' in next_all_stars))
    if is_kong_jie: results.append("    - 格局: 构成【空劫夹】，主思想超脱，但也易有财务波动、白忙一场或精神空虚之象。")
    return results


def analyze_functional_palace_evolution_with_stacking(
        data: dict,
        analysis_level: str,
        palaces_to_analyze: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    主分析函数，根据 analysis_level 动态裁剪输出深度，并返回一个按宫位划分的字典。
    新增 palaces_to_analyze 参数用于内容筛选。
    """

    LEVEL_TO_DISK_MAPPING = {
        'birth_chart': ['原局盘'], 'decadal': ['原局盘', '大限盘'], 'yearly': ['原局盘', '大限盘', '流年盘'],
        'monthly': ['原局盘', '大限盘', '流年盘', '流月盘'],
        'daily': ['原局盘', '大限盘', '流年盘', '流月盘', '流日盘'],
        'hourly': ['原局盘', '大限盘', '流年盘', '流月盘', '流日盘', '流时盘']
    }

    if analysis_level not in LEVEL_TO_DISK_MAPPING:
        return {}

    disk_order = LEVEL_TO_DISK_MAPPING[analysis_level]

    if palaces_to_analyze is None:
        target_palaces = ALL_PALACES
    else:
        target_palaces = [p for p in palaces_to_analyze if p in ALL_PALACES]  # Ensure requested palaces are valid

    reorganized_by_function = {disk: {} for disk in disk_order}
    reorganized_by_branch = {disk: {} for disk in disk_order}

    for i, disk_name in enumerate(disk_order):
        if disk_name in data:
            for palace_info in data[disk_name]:
                if len(palace_info) < 6: continue
                main_stars, earthly_branch, palace_name, sihua, fuxing, shaxing = palace_info

                # Standardize palace names if needed for internal consistency
                if palace_name == '仆役宫': palace_name = '交友宫'
                if palace_name == '官禄宫': palace_name = '事业宫'

                info_dict = {'宫位': palace_name, '地支': earthly_branch, '主星': main_stars, '四化': sihua,
                             '辅星': fuxing, '煞星及杂曜': shaxing}
                reorganized_by_function[disk_name][palace_name] = info_dict
                reorganized_by_branch[disk_name][earthly_branch] = info_dict

    analysis_results = {}

    for target_palace in target_palaces:
        description_parts = []

        for i, disk_name in enumerate(disk_order):
            line_parts = []
            if disk_name in reorganized_by_function and target_palace in reorganized_by_function[disk_name]:
                current_info = reorganized_by_function[disk_name][target_palace]
                current_branch, current_stars, current_sihua = current_info['地支'], current_info['主星'], current_info[
                    '四化']

                prefix = f"▶ 进入【{disk_name.replace('盘', '')}】" if disk_name != '原局盘' else "▶ 在【原局】"
                main_line = f"{prefix}，您的“{target_palace}”位于【{current_branch}】宫，由主星【{current_stars}】坐守"
                if current_sihua:
                    main_line += f"，并被引动，产生【{current_sihua}】的变化。"
                else:
                    main_line += "。"
                line_parts.append(main_line)

                if i > 0:
                    previous_disk_name, previous_disk_data, current_disk_data = disk_order[
                        i - 1], reorganized_by_branch.get(disk_order[i - 1], {}), reorganized_by_branch.get(disk_name,
                                                                                                            {})
                    upward_info = get_cross_level_interaction_analysis(current_branch, current_disk_data,
                                                                       previous_disk_data, disk_name,
                                                                       previous_disk_name)
                    if upward_info: line_parts.append(upward_info)

                if '天相' in current_stars.split(','):
                    current_idx = DI_ZHI_ORDER.index(current_branch)
                    prev_branch, next_branch = DI_ZHI_ORDER[(current_idx - 1 + 12) % 12], DI_ZHI_ORDER[
                        (current_idx + 1) % 12]
                    disk_data_by_branch = reorganized_by_branch[disk_name]
                    jia_gong_info = check_tianxiang_jia_gong(disk_data_by_branch.get(prev_branch),
                                                             disk_data_by_branch.get(next_branch))
                    if jia_gong_info: line_parts.extend(jia_gong_info)

                sfsz_info = get_comprehensive_sfsz_info(current_branch, reorganized_by_branch[disk_name])
                if sfsz_info: line_parts.append(sfsz_info)

            else:
                line_parts.append(f"▶ 在【{disk_name.replace('盘', '')}】中，未直接定义“{target_palace}”。")

            description_parts.append("\n".join(line_parts))

        analysis_results[target_palace] = "\n".join(description_parts)

    return analysis_results


# CSV_DATA_BASE_PATH and CSV_HOROSCOPE_DATA_BASE_PATH are still used by analysis functions
CSV_DATA_BASE_PATH = os.path.join(os.path.dirname(__file__), "combined_galaxy_data_new2")
CSV_HOROSCOPE_DATA_BASE_PATH = os.path.join(os.path.dirname(__file__), "combined_galaxy_data_star_new3")


async def robust_api_call_with_retry(
        session: aiohttp.ClientSession,
        url: str,
        payload: dict,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        max_delay: float = 16.0,
        timeout: int = 30
) -> dict:
    """
    一个带指数退避和抖动的健壮异步API调用函数。
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.warning(f"API调用失败，将在 {delay:.2f} 秒后进行第 {attempt} 次重试...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
                delay += random.uniform(0, delay * 0.5)

            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status >= 500:
                    response.raise_for_status()

                response.raise_for_status()
                return await response.json()

        except (aiohttp.ClientConnectionError, aiohttp.ClientOSError, asyncio.TimeoutError,
                aiohttp.ServerDisconnectedError) as e:
            logger.warning(f"API调用尝试 #{attempt} 遇到可重试的错误: {type(e).__name__}: {e}")
            last_exception = e
            continue

        except aiohttp.ClientResponseError as e:
            if e.status >= 500:
                logger.warning(f"API调用尝试 #{attempt} 遇到可重试的服务器错误: {e.status} {e.message}")
                last_exception = e
                continue
            else:
                logger.error(f"API调用遇到不可重试的客户端错误: {e.status} {e.message}")
                raise e

    logger.error(f"API调用在 {max_retries} 次重试后彻底失败。")
    raise last_exception


async def aiohttp_vllm_invoke(payload: dict, max_retries: int = 5) -> Tuple[Dict[str, Any], int, int]:
    """
    使用全局 aiohttp 客户端调用 VLLM，并带有重试机制以确保获取到有效的JSON响应。
    """
    input_tokens = count_tokens_for_messages(payload.get("messages", []))

    if not async_aiohttp_client:
        logger.error("AIOHTTP客户端未初始化。")
        raise HTTPException(503, "AIOHTTP客户端未初始化。")

    url = f"{VLLM_API_BASE_URL}/chat/completions"
    last_exception = None

    for attempt in range(max_retries):
        logger.info(f"正在调用VLLM (第 {attempt + 1}/{max_retries} 次)...")
        raw_content_for_logging = ""

        try:
            await asyncio.wait_for(vllm_semaphore.acquire(), timeout=VLLM_SLOT_WAIT_TIMEOUT_SECONDS)
            try:
                timeout = aiohttp.ClientTimeout(total=VLLM_REQUEST_TIMEOUT_SECONDS)
                logger.debug(f"获取到VLLM许可，准备发送请求到 {url} (超时: {timeout.total}s)")
                
                vllm_start_time = time.perf_counter()
                async with async_aiohttp_client.post(url, json=payload) as response:
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
                    output_tokens = count_tokens_for_string(content)

                    if not content or not content.strip():
                        raise ValueError("VLLM返回的 content 为空。")

                    parsed_json = _parse_lenient_json(content)
                    
                    vllm_end_time = time.perf_counter()
                    vllm_response_time = round((vllm_end_time - vllm_start_time) * 1000, 2)
                    
                    monitor_logger.info(f'{{"type": "vllm_response_time", "response_time_ms": {vllm_response_time}, "input_tokens": {input_tokens}, "output_tokens": {output_tokens}}}')

                    logger.info(f"VLLM调用成功 (第 {attempt + 1} 次尝试)，响应时长: {vllm_response_time}ms")
                    return parsed_json, input_tokens, output_tokens

            finally:
                vllm_semaphore.release()
                logger.debug("VLLM许可已释放。")

        except (asyncio.TimeoutError, aiohttp.ClientError, json.JSONDecodeError, ValueError) as e:
            logger.warning(
                f"VLLM调用在第 {attempt + 1} 次尝试时失败。错误类型: {type(e).__name__}, 错误: {e}. "
                f"原始Content(如果可用): '{raw_content_for_logging}'"
            )
            last_exception = e
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0)  # Increased sleep for more robust retry
            else:
                break

        except Exception as e:
            logger.error(f"【FATAL】aiohttp_vllm_invoke: 发生不可重试的未知错误: {e}", exc_info=True)
            return None, input_tokens, 0

    logger.error(f"在 {max_retries} 次尝试后，仍无法从VLLM获取有效响应。最后一次错误: {last_exception}")

    if isinstance(last_exception, asyncio.TimeoutError):
        return None, input_tokens, 0
    if isinstance(last_exception, aiohttp.ClientError):
        return None, input_tokens, 0
    if isinstance(last_exception, (json.JSONDecodeError, ValueError)):
        return None, input_tokens, 0

    raise HTTPException(500, "调用AI服务多次失败，请联系管理员。")


async def aiohttp_vllm_stream(
        payload: dict,
        max_retries: int = 5,
        initial_delay: float = 0.5,
        max_delay: float = 4.0
) -> AsyncGenerator[Tuple[str, int, int], None]:
    """
    使用全局 aiohttp 客户端调用 VLLM 并流式处理响应。
    """
    input_tokens = count_tokens_for_messages(payload.get("messages", []))
    output_tokens_accumulator = 0

    if not async_aiohttp_client:
        logger.error("AIOHTTP客户端未初始化。")
        yield "[错误: AIOHTTP客户端未初始化。][DONE]", 0, 0
        return

    payload['stream'] = True
    url = f"{VLLM_API_BASE_URL}/chat/completions"
    delay = initial_delay
    last_exception = None

    try:
        await asyncio.wait_for(vllm_semaphore.acquire(), timeout=VLLM_SLOT_WAIT_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.error("等待VLLM许可超时 (aiohttp_vllm_stream)。")
        VLLM_RESPONSES_FAILED.labels(reason="semaphore_timeout").inc()
        yield "[错误: AI服务正忙，请稍后再试。][DONE]", 0, 0
        return

    try:
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
                async with async_aiohttp_client.post(url, json=payload) as response:
                    response.raise_for_status()

                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith("data:"):
                            line_data = line[5:].strip()
                            if line_data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(line_data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content_piece = delta.get("content")
                                if content_piece:
                                    output_tokens_accumulator += count_tokens_for_string(content_piece)
                                    yield content_piece, input_tokens, output_tokens_accumulator
                            except json.JSONDecodeError:
                                logger.warning(f"无法解析VLLM流中的JSON行: {line_data}")
                                continue
                    VLLM_RESPONSES_SUCCESS.inc()
                    logger.info(f"计数: VLLM成功响应数 +1。当前总数: {VLLM_RESPONSES_SUCCESS._value.get()}")
                    return

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"VLLM(stream)连接尝试 #{attempt + 1} 遇到错误: {type(e).__name__}")
                last_exception = e
                if attempt == max_retries:
                    logger.error(f"VLLM(stream)在 {max_retries} 次重试后连接彻底失败。", exc_info=True)
                    VLLM_RESPONSES_FAILED.labels(reason="connection_error").inc()
                    error_message = "[错误: AI服务繁忙...]"
                    yield error_message, input_tokens, 0

                continue

    except Exception as e:
        logger.error(f"处理VLLM(stream)时发生未知错误: {e}", exc_info=True)
        VLLM_RESPONSES_FAILED.labels(reason="unknown_stream_error").inc()
        error_message = "[错误: AI服务繁忙...]"
        yield error_message, input_tokens, 0

    finally:
        vllm_semaphore.release()
        logger.debug("VLLM(stream)许可已释放。")


# --- 新增函数: 用于处理分号后的追问 ---
async def call_llm_for_follow_up(full_report_content: str, follow_up_question: str) -> str:
    """
    使用LLM针对生成的报告和追问问题进行回答。
    """
    system_prompt = f"""
    你是一位专业的紫微斗数命理分析师，擅长结合已有的年度运势报告，对用户提出的具体问题进行深入、精准的解答，必须锐评。

你已经为用户生成了一份详细的年度运势报告，现在用户对报告中的某些内容或报告未覆盖到的具体方面有进一步的追问。

你的任务是：
1. 仔细阅读用户提供的“已生成的年度运势报告”。
2. 针对用户提出的“追问问题”，结合报告内容进行回答。
3. 如果报告中已包含相关信息，请提炼并总结；如果报告中没有直接答案，请基于紫微斗数的常识进行合理推断和补充，但严禁过度解读。
4. 回答要清晰、具体、有帮助，全程以“你”来称呼用户，保持直接、冷静、专业、不带任何情感色彩的口吻，**绝不允许出现任何温和、安慰、鼓励、共情、积极引导或个人见解的语气。**
5. **最核心约束：严格禁止描述用户的性格特质、心理状态、情感反应、内心挣扎、自我怀疑、过往模式、未来改变或任何与用户主观能动性相关的行为。回答内容必须且只能专注于描述“事件发生的时间点”、“可能发生的地点”、“可能遇到的人的客观特质（例如：不热情、务实、不善表达情绪，行动上会默默照顾生活）”等纯粹的命理现象和客观信息。禁止任何形式的场景化描述、故事性描绘、情感铺垫或任何暗示用户心理或行为是事件发生前提的表达。**
6. 避免使用专业术语，用通俗易懂的白话文进行解释。
7. 判断必须基于命理逻辑和报告内容，不做任何不切实际、模糊或带有暗示性的推断。
8. 输出的回答内容长度控制在80-200字之间，保持极致精炼，惜字如金。
9. **绝对核心原则**：所有报告内容和分析始终围绕用户本人。例如，涉及感情问题时，夫妻宫所代表的信息是用户的运势表现，而非描述对方的命盘特质。
10. **回答内容必须紧扣用户追问，不发散，不引申，不展开任何话题。**
11. **内容必须是极致白话文，直截了当，不设任何铺垫或转折，说话言简意赅，单刀直入，将用户视作仅需获取纯粹、未经加工、不带任何主观推测的事实信息的对象。**
12. **话风务必极端简单、直接、硬朗，不弯弯绕绕，不带任何转折或试探。回答只需点明核心命理现象，绝不需要任何疗愈、安慰、共情、心理分析、比喻或哲学探讨的成分。**
13. **你要站在这份年度运势报告是你个人所生成的角度，以报告作者的身份，以最高度的客观性、不带任何情感色彩地回应用户的追问。**
14. **严禁任何疗愈、心理学、情感引导或“你其实是...”的风格。说话必须极其严谨、客观，禁止猜测用户情况或使用任何带有主观判断和引导的表达。所有分析和理由必须且只能来源于紫微斗数报告或命理常识，严禁主观推测或注入个人情感。**
15. **语气必须单刀直入，仅纯粹描述命理层面所示的“时间点”、“地点”、“对象特征（纯客观描述，如“不张扬”、“务实”）”。严禁描述伴侣的具体行为细节（如“默默照顾你的生活”这类具体行动细节亦要避免，仅保留“务实”等泛指特质），严禁描述用户或伴侣的情感状态或关系发展过程。**
16. **严禁指导用户采取任何具体行动（例如“你只需要...”），禁止使用否定性表述来指代用户（例如“他不是...”、“你不是...”），坚决杜绝任何中庸、模棱两可、带有引导性、鼓励性或预测用户反应的文风。**
17. **禁止任何关于“等待”或“行动”的哲学观讨论。只陈述“窗口期”或“可能遇到的对象类型”。**
18. **禁止任何关于“运气”、“改变自己”、“完美时机”、“完美信号”或“对的人”的讨论或定义。只陈述流年所指的“缘分显现”的时期和纯粹的命理现象。**
19. **禁止一切比喻、拟人、或任何非平铺直叙的修辞手法。**
20. **特别禁止以下表达模式：**
    * **“这段关系不是突然降临，而是在你专注自身、不刻意追求时自然出现。”**
    * **“你过去的情感模式会在此时被修正，你不再等待完美信号，而是接受‘不完美但真实’的陪伴。”**
    * **“正缘不在远方，就在你愿意停止自我怀疑的那一刻，悄然靠近。”**
    * **任何将“缘分出现”与用户心态或行为挂钩的措辞。**
    * **所有带有人文关怀、情感共鸣或心理建设意味的词句。**
    """

    user_message = f"""
    以下是您的年度运势报告：
    {full_report_content}

    我的追问是：{follow_up_question}
    """

    vllm_payload = {

        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 800,  # 给予更多token空间来回答追问
    }

    try:
        content_parts = []
        async for chunk_data in aiohttp_vllm_stream(vllm_payload):
            if isinstance(chunk_data, tuple) and len(chunk_data) == 3:
                chunk, input_tokens, output_tokens = chunk_data
                if chunk and not chunk.startswith("[错误:"):
                    content_parts.append(chunk)

        if content_parts:
            content = "".join(content_parts)
            logger.info(f"追问问题LLM调用完成，输出tokens: {output_tokens}")
            return content
        else:
            logger.warning("追问问题LLM调用返回空内容")
            return "抱歉，无法针对您的问题给出进一步的回答。"
    except Exception as e:
        logger.error(f"追问问题LLM调用异常: {e}", exc_info=True)
        return "抱歉，在处理您的追问时发生了错误，请稍后再试。"


def generate_ziwei_analysis(birth_info: Dict[str, Any], response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    通过调用外部API和处理本地CSV数据来生成紫微斗数分析。
    根据小时和分钟精确判断时辰。
    返回结构化的宫位数据字典。
    """
    if not (birth_info.get("year") and birth_info.get("month") and birth_info.get("day") and
            (birth_info.get("hour") is not None or birth_info.get("traditional_hour_branch")) and
            birth_info.get("gender") in ["男", "女"]):
        logger.error(f"【generate_ziwei_analysis】出生信息不完整或无效。返回错误。详情: {birth_info}")
        return {"error": "出生信息不完整或无效。"}

    hour = birth_info.get('hour')
    minute = birth_info.get('minute', 0)
    traditional_hour_branch = birth_info.get('traditional_hour_branch')

    logger.debug(f"DEBUG(gza): Received birth_info: {birth_info}")
    if traditional_hour_branch:
        traditional_hour_branch = traditional_hour_branch.strip()  # Strip whitespace
    logger.debug(f"DEBUG(gza): Extracted and stripped traditional_hour_branch: '{traditional_hour_branch}'")

    time_index = None
    # --- 时辰解析逻辑修正：优先使用 calculate_time_index ---
    if hour is not None:
        time_index = calculate_time_index(hour, minute)
        logger.debug(f"DEBUG(gza): Resolved time_index from hour/minute: {time_index}")
    elif traditional_hour_branch and traditional_hour_branch in TRADITIONAL_HOUR_TO_TIME_INDEX:
        time_index = TRADITIONAL_HOUR_TO_TIME_INDEX[traditional_hour_branch]
        logger.debug(f"DEBUG(gza): Resolved time_index from traditional_hour_branch (fallback): {time_index}")
    # --- 时辰解析逻辑修正结束 ---

    if time_index is None:
        logger.error(f"【generate_ziwei_analysis】未能解析时辰信息。birth_info: {birth_info}")
        return {"error": "抱歉，无法解析您的出生时辰信息。"}

    if not (response_data and response_data.get("data") and response_data["data"].get("astrolabe")):
        logger.error(f"【generate_ziwei_analysis】API响应数据结构不符合预期：缺少'data'或'astrolabe'。{response_data}")
        return {"error": "抱歉，API返回了意外的数据结构，无法继续分析。"}

    natal_astrolabe = response_data["data"]["astrolabe"]
    if not (natal_astrolabe.get("palaces") and isinstance(natal_astrolabe["palaces"], list) and len(
            natal_astrolabe["palaces"]) == 12):
        logger.error(f"【generate_ziwei_analysis】API响应中的 'palaces' 数据无效或数量不足12个。")
        return {"error": "抱歉，API返回的命盘宫位数据不完整或格式不正确。"}

    data_ch = natal_astrolabe["rawDates"]["chineseDate"]
    parsed_data = parse_bazi_components(data_ch)

    original_data = response_data
    converted_data = replace_key_names(original_data, key_mapping)
    ziwei_data2 = ziweids2(converted_data)

    translated_data = translate_json(original_data, key_translation_map, value_translation_map)
    ziwei_data_原局 = ziweids_conzayao(translated_data['数据'], "命盘", '命主星', '身主星', '五行局', '宫位',
                                       '主星', '地支', '化曜', '辅星煞星', '杂曜', '名称', zhuxing)
    cleaned_chart_array = []
    for item in ziwei_data_原局:
        processed_item = ["", "", "", "", ""]
        if len(item) > 0: processed_item[0] = item[0]
        if len(item) > 1: processed_item[1] = item[1]
        if len(item) > 2: processed_item[2] = item[2]
        if len(item) > 3: processed_item[3] = item[3]
        all_other_stars_parts = []
        if len(item) > 4:
            for i in range(4, len(item)):
                if isinstance(item[i], str) and item[i].strip():
                    all_other_stars_parts.append(item[i].replace("，", ",").strip())
        processed_item[4] = ",".join(filter(None, all_other_stars_parts)).replace(',,', ',').strip(',')
        cleaned_chart_array.append(processed_item)
    chart_description = describe_ziwei_chart(cleaned_chart_array)

    palace_key_info_summaries = {}
    for i in ziwei_data2:
        if len(i) < 3:
            logger.warning(f"【generate_ziwei_analysis】ziwei_data2 中的元素格式不正确，跳过: {i}")
            continue

        x = re.sub(r'[和,，]\s*', '_', i[0])
        y = i[2].replace("宫宫", "宫")
        y = y.replace("官禄宫", "事业宫")
        y = y.replace("仆役宫", "交友宫")

        original_dizhi = i[1]
        file_read_successful = False

        file_path = os.path.join(CSV_DATA_BASE_PATH, f"{x}_{original_dizhi}_{y}_all_time_combinations.csv")
        try:
            data = pd.read_csv(file_path)
            data = data.astype(str).replace('nan', '')
            file_read_successful = True
        except FileNotFoundError:
            duigong_dizhi = DIZHI_DUIGONG_MAP.get(original_dizhi)
            if duigong_dizhi:
                duigong_file_path = os.path.join(CSV_DATA_BASE_PATH,
                                                 f"grouped_{x}_{duigong_dizhi}_{y}_all_time_combinations.csv")
                try:
                    data = pd.read_csv(duigong_file_path)
                    data = data.astype(str).replace('nan', '')
                    file_read_successful = True
                except FileNotFoundError:
                    pass
                except KeyError as ke:
                    logger.error(f"【generate_ziwei_analysis】错误: 对宫文件 {duigong_file_path} 中缺少预期列: {ke}。")
                except Exception as e:
                    logger.error(f"【generate_ziwei_analysis】处理对宫地支文件 {duigong_file_path} 时发生错误: {e}")
            else:
                pass
        except KeyError as ke:
            logger.error(f"【generate_ziwei_analysis】错误: 文件 {file_path} 中缺少预期列: {ke}。")
        except Exception as e:
            logger.error(f"【generate_ziwei_analysis】处理文件 {file_path} 时发生错误: {e}")

        if file_read_successful:
            filtered_data = data.loc[(data["年干"] == parsed_data.get('year_stem')) &
                                     (data["年支"] == parsed_data.get('year_branch')) &
                                     (data["月支"] == parsed_data.get('month_branch')) &
                                     (data["时辰"] == parsed_data.get('hour_branch'))]

            if not filtered_data.empty:
                for index, row in filtered_data.iterrows():
                    if "宫位" in row and "现象" in row and row["宫位"] != "" and row["现象"] != "":
                        palace_name = row['宫位']
                        summary_text = row['现象']
                        if palace_name not in palace_key_info_summaries:
                            palace_key_info_summaries[palace_name] = {}
                        if "原局盘" not in palace_key_info_summaries[palace_name]:
                            palace_key_info_summaries[palace_name]["原局盘"] = []
                        palace_key_info_summaries[palace_name]["原局盘"].append(summary_text)
            else:
                pass

    parsed_palace_descriptions_for_birth_chart = parse_chart_description_block(chart_description)

    structured_palace_details_for_llm = {palace: {} for palace in ALL_PALACES}

    for palace_name in ALL_PALACES:
        description = parsed_palace_descriptions_for_birth_chart.get(palace_name, "")
        summaries = palace_key_info_summaries.get(palace_name, {}).get("原局盘", [])

        if description or summaries:
            if "原局盘" not in structured_palace_details_for_llm[palace_name]:
                structured_palace_details_for_llm[palace_name]["原局盘"] = {}
            structured_palace_details_for_llm[palace_name]["原局盘"]["宫位信息"] = description
            structured_palace_details_for_llm[palace_name]["原局盘"]["关键信息汇总"] = "; ".join(summaries[:10])
        else:
            if palace_name not in structured_palace_details_for_llm:
                structured_palace_details_for_llm[palace_name] = {}
            structured_palace_details_for_llm[palace_name]["原局盘"] = {
                "宫位信息": "",
                "关键信息汇总": ""
            }
    return structured_palace_details_for_llm


# 辅助函数：解析 describe_ziwei_chart 的输出
def parse_chart_description_block(chart_description_block: str) -> Dict[str, str]:
    """
    解析 describe_ziwei_chart 函数生成的宫位描述块，
    并提取每个宫位的详细描述。
    返回一个字典，键为宫位名称，值为对应的描述字符串。
    """
    palace_descriptions = {}
    separator = "现在，让我们逐一看看您其他主要宫位的配置，及其三方四正和夹宫情况："
    parts = chart_description_block.split(separator, 1)

    if len(parts) > 0:
        main_palace_block = parts[0].strip()
        if main_palace_block:
            lines = main_palace_block.split('\n')
            first_line = lines[0].strip()
            re_match_result = re.match(r'^(您的)?(\w+宫)坐落于(.*)', first_line)
            if re_match_result:
                palace_name = re_match_result.group(2)
                desc_content = "坐落于" + re_match_result.group(3).strip()
                full_description = desc_content
                if len(lines) > 1:
                    full_description += "\n" + "\n".join(lines[1:]).strip()
                palace_descriptions[palace_name] = full_description.strip()
            else:
                if "命宫" in main_palace_block:
                    palace_descriptions["命宫"] = main_palace_block.strip()

    if len(parts) > 1:
        other_palaces_block = parts[1].strip()
        lines = other_palaces_block.split('\n')

        current_palace_name = None
        current_palace_desc_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            re_match_other_palace = re.match(r'^- (\w+宫)坐落于(.*)', line)

            if re_match_other_palace:
                if current_palace_name and current_palace_desc_lines:
                    palace_descriptions[current_palace_name] = " ".join(current_palace_desc_lines).strip()

                current_palace_name = re_match_other_palace.group(1)
                desc_content = "坐落于" + re_match_other_palace.group(2).strip()
                current_palace_desc_lines = [desc_content]
            else:
                if current_palace_desc_lines:
                    current_palace_desc_lines.append(line)

        if current_palace_name and current_palace_desc_lines:
            palace_descriptions[current_palace_name] = " ".join(current_palace_desc_lines).strip()

    return palace_descriptions


async def generate_horoscope_analysis(birth_info: Dict[str, Any], horoscope_date_str: Optional[str], analysis_level: str,
                                response_data: Dict[str, Any], query_intent_data: Dict[str, Any],
                                prompt_input: str) -> (
        Dict[str, Any], str, str, Dict[str, Any], str, str, str):
    """
    通过调用外部API生成指定运势日期的紫微斗数分析。
    返回结构化的宫位数据字典和用于LLM的Prompt键。
    """
    if not (birth_info.get("year") and birth_info.get("month") and birth_info.get("day") and
            (birth_info.get("hour") is not None or birth_info.get("traditional_hour_branch")) and
            birth_info.get("gender") in ["男", "女"]):
        logger.error(f"【generate_horoscope_analysis】出生信息不完整或无效。详情: {birth_info}")
        # Ensure we return 7 values, even in case of error, to prevent ValueError
        return {"error": "出生信息不完整或无效。"}, "general_question", "{}", {}, "", "", "{}"

    hour = birth_info.get('hour')
    minute = birth_info.get('minute', 0)
    traditional_hour_branch = birth_info.get('traditional_hour_branch')

    logger.debug(f"DEBUG(gha_horoscope): Received birth_info: {birth_info}")
    if traditional_hour_branch:
        traditional_hour_branch = traditional_hour_branch.strip()  # Strip whitespace
    logger.debug(f"DEBUG(gha_horoscope): Extracted and stripped traditional_hour_branch: '{traditional_hour_branch}'")

    time_index = None
    # --- 时辰解析逻辑修正：优先使用 calculate_time_index ---
    if hour is not None:
        time_index = calculate_time_index(hour, minute)
        logger.debug(f"DEBUG(gha_horoscope): Resolved time_index from hour/minute: {time_index}")
    elif traditional_hour_branch and traditional_hour_branch in TRADITIONAL_HOUR_TO_TIME_INDEX:
        time_index = TRADITIONAL_HOUR_TO_TIME_INDEX[traditional_hour_branch]
        logger.debug(f"DEBUG(gha_horoscope): Resolved time_index from traditional_hour_branch (fallback): {time_index}")
    # --- 时辰解析逻辑修正结束 ---

    if time_index is None:
        logger.error(f"【generate_horoscope_analysis】未能解析时辰信息。birth_info: {birth_info}")
        # Ensure we return 7 values, even in case of error, to prevent ValueError
        return {"error": "抱歉，无法解析您的出生时辰信息。"}, "general_question", "{}", {}, "", "", "{}"

    if not (response_data and response_data.get("data") and response_data["data"].get("astrolabe") and
            response_data["data"]["astrolabe"].get("palaces") and isinstance(
                response_data["data"]["astrolabe"]["palaces"], list) and len(
                response_data["data"]["astrolabe"]["palaces"]) == 12):
        logger.error(f"【generate_horoscope_analysis】API响应数据结构不符合预期：缺少'data'或'astrolabe'。{response_data}")
        # Ensure we return 7 values, even in case of error, to prevent ValueError
        return {"error": "抱歉，API返回了意外的数据结构，无法继续分析。"}, "general_question", "{}", {}, "", "", "{}"

    if analysis_level not in ["birth_chart", "unpredictable_future"] and not (
            response_data["data"].get("horoscope") and isinstance(response_data["data"]["horoscope"], dict)):
        logger.error(f"【generate_horoscope_analysis】运势API响应数据结构不符合预期：缺少关键运势数据或格式不正确。")
        # Ensure we return 7 values, even in case of error, to prevent ValueError
        return {
            "error": "抱歉，运势API返回的运势数据不完整或格式不正确，无法继续分析。"}, "general_question", "{}", {}, "", "", "{}"

    astrolabe_palaces = response_data['data']['astrolabe']['palaces']

    decadal_stem_branch_str = ""
    if analysis_level == "decadal":
        decadal_horoscope_data = response_data['data']['horoscope'].get('decadal', {})
        if decadal_horoscope_data and "heavenlyStem" in decadal_horoscope_data and "earthlyBranch" in decadal_horoscope_data:
            decadal_stem_branch_str = f"{decadal_horoscope_data['heavenlyStem']}{decadal_horoscope_data['earthlyBranch']}大运"

    # 执行转换，将 API 响应数据转换为统一的格式
    transformed_horoscope_scopes = {
        "原局盘": transform_palace_data(response_data['data']['astrolabe']),  # 原始命盘宫位
        "原局盘_new": transform_palace_data_new(response_data['data']['astrolabe']),
        "大限盘": transform_horoscope_scope_data(
            response_data['data']['horoscope'].get('decadal', {}), astrolabe_palaces),
        "流年盘": transform_horoscope_scope_data(
            response_data['data']['horoscope'].get('yearly', {}), astrolabe_palaces),
        "流月盘": transform_horoscope_scope_data(
            response_data['data']['horoscope'].get('monthly', {}), astrolabe_palaces),
        "流日盘": transform_horoscope_scope_data(
            response_data['data']['horoscope'].get('daily', {}), astrolabe_palaces),
        "流时盘": transform_horoscope_scope_data(
            response_data['data']['horoscope'].get('hourly', {}), astrolabe_palaces),
    }

    processed_chart_data = preprocess_chart(transformed_horoscope_scopes["原局盘"])
    chart_by_name = {v['宫位名']: v for k, v in processed_chart_data.items()}

    # 确保 rules_df 已加载
    global rules_df
    if rules_df is None:
        logger.error("rules_df 尚未加载，无法进行分析")
        return {"error": "规则数据未加载，请稍后重试。"}, "general_question", "{}", {}, "", "", "{}"

    tiangan = response_data['data']['astrolabe']['rawDates']['chineseDate']['yearly'][0]
    analysis_result = analyze_chart_final_optimized(processed_chart_data, rules_df, tiangan)

    # START OF NEW USER REQUESTED MODIFICATION (Part 1 - Extract suiqian12 and jiangqian12)
    # Safely get yearlyDecStar, handling potential missing keys
    yearly_horoscope_data = response_data['data']['horoscope'].get('yearly', {})
    yearly_dec_star_data = yearly_horoscope_data.get('yearlyDecStar', {})
    suiqian12 = yearly_dec_star_data.get('suiqian12', [])
    jiangqian12 = yearly_dec_star_data.get('jiangqian12', [])

    # Define the fixed earthly branch order for these star lists
    DIZHI_ORDER_FOR_DECASTARS = ['寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥', '子', '丑']

    # Create mappings for suiqian12 and jiangqian12
    dizhi_to_suiqian12_map = dict(zip(DIZHI_ORDER_FOR_DECASTARS, suiqian12))
    dizhi_to_jiangqian12_map = dict(zip(DIZHI_ORDER_FOR_DECASTARS, jiangqian12))
    # END OF NEW USER REQUESTED MODIFICATION (Part 1)

    # 原局盘星耀信息
    # logger.debug("原局盘星耀信息" + str(transform_palace_data_new(response_data['data']['astrolabe'])))

    palace_info_map = {p[2]: (p[0], p[1]) for p in transformed_horoscope_scopes["原局盘"]}
    summary_output = []

    for palace_name in PALACE_ORDER:
        palace_details = chart_by_name.get(palace_name)
        if not palace_details: continue

        properties = analysis_result.get(palace_name, {})

        if not properties:
            pass
        else:
            sorted_properties = sorted(properties.items(), key=lambda item: item[1]['total_score'], reverse=True)

        main_stars, dizhi = palace_info_map.get(palace_name, ('未知', '?'))
        if not properties:
            summary_str = "未匹配到任何性质"
        else:
            sorted_properties = sorted(properties.items(), key=lambda item: item[1]['total_score'], reverse=True)
            summary_str = ','.join([f"{prop}:{data['total_score']:.1f}" for prop, data in sorted_properties])
        summary_output.append([main_stars, dizhi, summary_str])

    # Pass relevant_palaces to filter the output of analyze_functional_palace_evolution_with_stacking
    full_analysis_report = analyze_functional_palace_evolution_with_stacking(
        transformed_horoscope_scopes,
        analysis_level=analysis_level,
        palaces_to_analyze=query_intent_data.get("relevant_palaces")  # Use the filtered list here
    )

    relevant_chart_keys_in_order = []
    selected_prompt_key_for_horoscope = "general_question"
    analysis_level_new = ''

    if analysis_level == "hourly":
        relevant_chart_keys_in_order = ["流时盘", "流日盘"]
        selected_prompt_key_for_horoscope = "hourly"
        analysis_level_new = '流时'
    elif analysis_level == "daily":
        relevant_chart_keys_in_order = ["流日盘", "流月盘"]
        selected_prompt_key_for_horoscope = "daily"
        analysis_level_new = '流日'
    elif analysis_level == "monthly":
        relevant_chart_keys_in_order = ["流月盘", "流年盘"]
        selected_prompt_key_for_horoscope = "monthly"
        analysis_level_new = '流月'
    elif analysis_level == "yearly":
        relevant_chart_keys_in_order = ["流年盘", "大限盘"]
        selected_prompt_key_for_horoscope = "yearly"
        analysis_level_new = '流年'
    elif analysis_level == "decadal":
        relevant_chart_keys_in_order = ["大限盘", "原局盘"]
        selected_prompt_key_for_horoscope = "decadal"
        analysis_level_new = '大限'
    elif analysis_level == "birth_chart":
        relevant_chart_keys_in_order = ["原局盘"]
        selected_prompt_key_for_horoscope = "birth_chart_analysis"
        analysis_level_new = 'all'
    else:
        relevant_chart_keys_in_order = []
        selected_prompt_key_for_horoscope = "general_question"
        analysis_level_new = 'all'

    daily_results = perform_astrological_analysis(
        raw_data=transformed_horoscope_scopes,
        scores_data=summary_output,
        excel_path=excel_path,
        analysis_level=analysis_level_new
    )

    xingzhi = ''
    # Only iterate through relevant palaces
    for i in query_intent_data['relevant_palaces']:
        i_palace = i.replace("事业宫", "官禄宫").replace("交友宫", "仆役宫")
        ming_gong_daily = daily_results.get(i_palace, {}).get(analysis_level_new, [])
        if ming_gong_daily:
            for rule in ming_gong_daily:
                xingzhi = xingzhi + f" 宫位：{i_palace} , 性质: {rule['性质']}, 描述: {rule['现象描述']}, 性质得分: {rule['得分']}"
        else:
            pass

    # structured_palace_details_for_llm now only includes the relevant palaces
    structured_palace_details_for_llm = {palace: {} for palace in query_intent_data['relevant_palaces']}
    collected_relevant_charts = []

    # START OF PREVIOUS USER'S REQUESTED MODIFICATION (Dizhi to Zayaos map)
    ori_palace = transform_palace_data_new(response_data['data']['astrolabe'])
    # 创建地支到杂曜的映射，方便查找
    dizhi_to_zayaos_map = {item[1]: item[5] for item in ori_palace if len(item) > 5}
    # END OF PREVIOUS USER'S REQUESTED MODIFICATION

    # 原局盘星耀信息
    # logger.debug("原局盘星耀信息" + str(transform_palace_data_new(response_data['data']['astrolabe'])))

    for chart_key in relevant_chart_keys_in_order:
        if chart_key in transformed_horoscope_scopes and transformed_horoscope_scopes[chart_key]:
            chart_data_list = transformed_horoscope_scopes[chart_key]

            # START OF MODIFICATION: 为 chart_data_list 中的每个宫位信息添加杂曜、岁前十二神、将前十二神
            augmented_chart_data_list = []
            for item_j in chart_data_list:
                mutable_j = list(item_j)  # 创建一个可变的列表副本
                dizhi = mutable_j[1] if len(mutable_j) > 1 else None

                # Add zayaos from ori_palace (first modification - at index 6)
                zayaos_from_ori = dizhi_to_zayaos_map.get(dizhi, "")
                # Ensure mutable_j has at least 6 elements (indices 0-5) before appending at index 6.
                while len(mutable_j) < 6:
                    mutable_j.append("")
                mutable_j.append(zayaos_from_ori)  # Zayaos is now at index 6

                # Add suiqian12 and jiangqian12 ONLY for "流年盘" (at indices 7 and 8)
                if chart_key == "流年盘":
                    suiqian_star = dizhi_to_suiqian12_map.get(dizhi, "")
                    jiangqian_star = dizhi_to_jiangqian12_map.get(dizhi, "")
                    mutable_j.append(suiqian_star)  # Suiqian12 is now at index 7
                    mutable_j.append(jiangqian_star)  # Jiangqian12 is now at index 8

                augmented_chart_data_list.append(mutable_j)
            # END OF MODIFICATION: 为 chart_data_list 中的每个宫位信息添加杂曜、岁前十二神、将前十二神

            # Filter augmented_chart_data_list to only include relevant_palaces
            filtered_augmented_chart_data_list = []
            for item in augmented_chart_data_list:
                palace_name_raw = item[2]  # Palace name is at index 2
                # Standardize palace name for comparison
                palace_name_standardized = palace_name_raw.replace("官禄宫", "事业宫").replace("仆役宫", "交友宫")
                if palace_name_standardized in query_intent_data['relevant_palaces']:
                    filtered_augmented_chart_data_list.append(item)

            collected_relevant_charts.append(filtered_augmented_chart_data_list)  # 使用增强且过滤后的列表
            full_chart_description_block = describe_ziwei_chart(filtered_augmented_chart_data_list)  # 使用增强且过滤后的列表
            parsed_palace_descriptions = parse_chart_description_block(full_chart_description_block)

            for palace_name, description in parsed_palace_descriptions.items():
                if palace_name not in structured_palace_details_for_llm:
                    structured_palace_details_for_llm[palace_name] = {}
                if chart_key not in structured_palace_details_for_llm[palace_name]:
                    structured_palace_details_for_llm[palace_name][chart_key] = {}
                structured_palace_details_for_llm[palace_name][chart_key]["宫位信息"] = description

            # START OF MODIFICATION: 将杂曜信息、岁前十二神、将前十二神插入 structured_palace_details_for_llm
            for j_augmented in filtered_augmented_chart_data_list:  # Now j is a potentially longer list
                # logger.debug("j_augmented: " + str(j_augmented))
                palace_name_from_chart = j_augmented[2]  # 宫位名在索引 2
                # Standardize palace name
                palace_name_from_chart_standardized = palace_name_from_chart.replace("官禄宫", "事业宫").replace(
                    "仆役宫", "交友宫")

                if palace_name_from_chart_standardized not in query_intent_data['relevant_palaces']:
                    continue  # Skip if not a relevant palace

                # Zayaos (from ori_palace) - at index 6
                if len(j_augmented) > 6:
                    zayaos_val = j_augmented[6]
                    if zayaos_val:
                        if palace_name_from_chart_standardized not in structured_palace_details_for_llm:
                            structured_palace_details_for_llm[palace_name_from_chart_standardized] = {}
                        if chart_key not in structured_palace_details_for_llm[palace_name_from_chart_standardized]:
                            structured_palace_details_for_llm[palace_name_from_chart_standardized][chart_key] = {}
                        structured_palace_details_for_llm[palace_name_from_chart_standardized][chart_key][
                            "杂曜"] = zayaos_val

                # Suiqian12 and Jiangqian12 - ONLY for "流年盘", at indices 7 and 8
                if chart_key == "流年盘" and len(j_augmented) > 8:
                    suiqian_val = j_augmented[7]
                    jiangqian_val = j_augmented[8]

                    if suiqian_val:
                        if palace_name_from_chart_standardized not in structured_palace_details_for_llm:
                            structured_palace_details_for_llm[palace_name_from_chart_standardized] = {}
                        if chart_key not in structured_palace_details_for_llm[palace_name_from_chart_standardized]:
                            structured_palace_details_for_llm[palace_name_from_chart_standardized][chart_key] = {}
                        structured_palace_details_for_llm[palace_name_from_chart_standardized][chart_key][
                            "岁前十二神"] = suiqian_val

                    if jiangqian_val:
                        if palace_name_from_chart_standardized not in structured_palace_details_for_llm:
                            structured_palace_details_for_llm[palace_name_from_chart_standardized] = {}
                        if chart_key not in structured_palace_details_for_llm[palace_name_from_chart_standardized]:
                            structured_palace_details_for_llm[palace_name_from_chart_standardized][chart_key] = {}
                        structured_palace_details_for_llm[palace_name_from_chart_standardized][chart_key][
                            "将前十二神"] = jiangqian_val
            # END OF MODIFICATION: 将杂曜信息、岁前十二神、将前十二神插入 structured_palace_details_for_llm

            for j in filtered_augmented_chart_data_list:  # Now j is a potentially longer list
                # logger.debug("j after filtering" + str(j))
                if len(j) < 6: continue

                x = re.sub(r'[和,，]\s*', '_', j[0])
                y_display = j[2]
                y_file = y_display.replace("宫宫", "宫").replace("官禄宫", "事业宫").replace("仆役宫", "交友宫")

                original_dizhi = j[1]
                hua_star_val = j[3].split(',')[0]
                minor_stars_val = j[4]
                match_stars_val = ",".join(filter(None, [minor_stars_val])).replace(" ", "")

                file_read_successful = False
                data = pd.DataFrame()

                file_path = os.path.join(CSV_HOROSCOPE_DATA_BASE_PATH,
                                         f"grouped_{x}_{original_dizhi}_{y_file}_all_time_combinations.csv")
                try:
                    data = pd.read_csv(file_path)
                    data = data.astype(str).replace('nan', '')
                    file_read_successful = True
                except FileNotFoundError:
                    duigong_dizhi = DIZHI_DUIGONG_MAP.get(original_dizhi)
                    if duigong_dizhi:
                        duigong_file_path = os.path.join(CSV_HOROSCOPE_DATA_BASE_PATH,
                                                         f"grouped_{x}_{duigong_dizhi}_{y_file}_all_time_combinations.csv")
                        try:
                            data = pd.read_csv(duigong_file_path)
                            data = data.astype(str).replace('nan', '')
                            file_read_successful = True
                        except FileNotFoundError:
                            pass
                        except KeyError as ke:
                            logger.error(
                                f"【generate_horoscope_analysis】错误: 对宫文件 {duigong_file_path} 中缺少预期列: {ke}。")
                        except Exception as e:
                            logger.error(
                                f"【generate_horoscope_analysis】处理对宫地支文件 {duigong_file_path} 时发生错误: {e}")
                    else:
                        pass
                except KeyError as ke:
                    logger.error(f"【generate_horoscope_analysis】错误: 文件 {file_path} 中缺少预期列: {ke}。")
                except Exception as e:
                    logger.error(f"【generate_horoscope_analysis】处理文件 {file_path} 时发生错误: {e}")

                if file_read_successful and not data.empty:
                    filtered_data = data.loc[(data["匹配星曜"] == match_stars_val) & (data["化星"] == hua_star_val)]

                    palace_name_standardized = y_file  # Assume standardized name for consistency
                    if palace_name_standardized not in structured_palace_details_for_llm:
                        structured_palace_details_for_llm[palace_name_standardized] = {}
                    if chart_key not in structured_palace_details_for_llm[palace_name_standardized]:
                        structured_palace_details_for_llm[palace_name_standardized][chart_key] = {}
                    if "关键信息汇总_raw_list" not in structured_palace_details_for_llm[palace_name_standardized][
                        chart_key]:
                        structured_palace_details_for_llm[palace_name_standardized][chart_key][
                            "关键信息汇总_raw_list"] = []

                    if not filtered_data.empty:
                        for index, row in filtered_data.iterrows():
                            if "宫位" in row and "汇总现象" in row and row["宫位"] != "" and row["汇总现象"] != "":
                                # Note: palace_name from row might be '官禄宫', '仆役宫', etc.
                                # Use palace_name_standardized (derived from y_file) as the consistent key for structured_palace_details_for_llm
                                summary_text = row['汇总现象']
                                structured_palace_details_for_llm[palace_name_standardized][chart_key][
                                    "关键信息汇总_raw_list"].append(summary_text)
                    else:
                        # If no matching data, raw_list remains empty, which is fine.
                        pass

    for palace_name, chart_data in structured_palace_details_for_llm.items():
        for chart_key, details in chart_data.items():
            if "关键信息汇总_raw_list" in details and isinstance(details["关键信息汇总_raw_list"], list):
                raw_summaries = details["关键信息汇总_raw_list"]
                unique_points = set()
                for s in raw_summaries:
                    points = re.split(r'[;；。\n]', s)
                    for p in points:
                        p_stripped = p.strip()
                        if p_stripped:
                            unique_points.add(p_stripped)
                details["关键信息汇总"] = "; ".join(sorted(list(unique_points)))
                del details["关键信息汇总_raw_list"]
            else:
                details["关键信息汇总"] = ""

    structured_palace_details_for_llm["_decadal_info"] = decadal_stem_branch_str

    # 调整 collected_relevant_charts 的长度检查逻辑
    needs_multiple_charts = analysis_level in ["yearly", "decadal", "monthly", "daily", "hourly"]
    if needs_multiple_charts and len(collected_relevant_charts) < 2:
        logger.error("运势分析需要至少两个层级的盘数据，但缺少其中一个。")
        return {"error": "运势分析数据不完整。"}, "general_question", "{}", {}, "", "", "{}"
    elif analysis_level == "birth_chart" and len(collected_relevant_charts) < 1:
        logger.error("原局盘分析缺少数据。")
        return {"error": "原局盘分析数据不完整。"}, "general_question", "{}", {}, "", "", "{}"

    final_score_json_str = "{}"
    final_sum = "{}"

    # 只有在有足够图表进行评分计算时才执行此部分
    if needs_multiple_charts and len(collected_relevant_charts) >= 2:
        liu_nian_parsed = parse_palace_data(collected_relevant_charts[0])
        da_xian_parsed = parse_palace_data(collected_relevant_charts[1])

        final_results_list = []
        for palace_info_dict in liu_nian_parsed:
            palace_name = palace_info_dict.get('palace_name', '未知宫位')

            # Only calculate scores for relevant palaces
            palace_name_standardized = palace_name.replace("官禄宫", "事业宫").replace("仆役宫", "交友宫")
            if palace_name_standardized not in query_intent_data['relevant_palaces']:
                continue

            di_zhi = palace_info_dict.get('di_zhi', '未知地支')
            score, source_scores, summary_dict = calculate_score_by_rules(
                palace_info_dict,
                liu_nian_parsed,
                da_xian_parsed
            )
            final_results_list.append({
                "palace_name": palace_name_standardized,  # Use standardized name
                "di_zhi": di_zhi,
                "final_score": round(score, 1),
                "combination_source_scores": source_scores,
                "summary": summary_dict
            })

        final_output = {}
        for result in final_results_list:
            final_output[result['palace_name']] = result['final_score']
        final_score_json_str = json.dumps(final_output, indent=2, ensure_ascii=False)

        final_output_sum = {}
        for result in final_results_list:
            final_output_sum[result['palace_name']] = result['summary']["overall_evaluation"]
        final_sum = json.dumps(final_output_sum, indent=2, ensure_ascii=False)

    # --- 最终返回 7 个值 ---
    return structured_palace_details_for_llm, selected_prompt_key_for_horoscope, final_score_json_str, full_analysis_report, str(
        summary_output), xingzhi.replace("官禄", "事业").replace("仆役", "交友"), final_sum


async def generate_annual_report_by_domains(
        result_list: dict,
        user_info: dict,
        analysis_scope: str,
        question: str,
        full_analysis_report: dict,
        xingzhi: str,
        analysis_data_result: Dict[str, Any],
        relevant_palaces_for_report: List[str],
        month_data_raw: list = None,
        final_score_json_str: str = "{}",
        target_year: int = 2025,
        final_sum: str = "{}",
) -> dict:
    """
    生成2025年一定会发生的3件事
    直接使用年度分数数据，不需要分析所有领域
    """
    logger.info("开始生成年度预测...")
    
    # 直接使用年度分数数据生成，不需要分析所有领域
    domain_results = {}
    
    # 生成年度关键词和必然发生的事情
    annual_keywords = await generate_annual_keywords(domain_results, user_info, final_score_json_str, target_year, final_sum)

    logger.info("年度预测生成完成")

    # 返回结构化数据
    return annual_keywords



async def generate_annual_keywords(domain_results: dict, user_info: dict, final_score_json_str: str = "{}", target_year: int = 2025, final_sum: str = "{}") -> dict:
    """
    基于年度分数数据生成2025年一定会发生的3件事
    优先选择征象最多的宫位，重点关注财运、事业、感情、健康
    """
    # 解析年度分数，用于生成必然发生的事情
    scores_info = ""
    top_palaces_info = []
    bottom_palaces_info = []
    
    try:
        if final_score_json_str and final_score_json_str != "{}":
            scores_dict = json.loads(final_score_json_str) if isinstance(final_score_json_str, str) else final_score_json_str
            if scores_dict:
                # 解析宫位解释（如果有）
                sum_dict = {}
                if final_sum and final_sum != "{}":
                    try:
                        sum_dict = json.loads(final_sum) if isinstance(final_sum, str) else final_sum
                    except:
                        sum_dict = {}
                
                # 按分数排序
                sorted_palaces = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
                top_two_palaces = sorted_palaces[:2] if len(sorted_palaces) >= 2 else sorted_palaces
                bottom_two_palaces = sorted_palaces[-2:] if len(sorted_palaces) >= 2 else []
                
                for palace, score in top_two_palaces:
                    explanation = sum_dict.get(palace, "")
                    top_palaces_info.append(f"{palace}（分数：{score}分）{explanation}")
                
                for palace, score in bottom_two_palaces:
                    explanation = sum_dict.get(palace, "")
                    bottom_palaces_info.append(f"{palace}（分数：{score}分）{explanation}")
                
                scores_info = f"""
各宫位分数情况：
分数最高的两个宫位：
{chr(10).join(top_palaces_info) if top_palaces_info else "无数据"}

分数最低的两个宫位：
{chr(10).join(bottom_palaces_info) if bottom_palaces_info else "无数据"}

所有宫位分数：
{chr(10).join([f"{palace}：{score}分" for palace, score in sorted_palaces])}
"""
    except Exception as e:
        logger.warning(f"解析年度分数数据失败: {e}")

    system_prompt = f"""
你是一个专业的紫微斗数分析师。请根据用户的年度运势分数，生成3条确定会发生的事件：2条来自分数最高的宫位（正向），1条来自分数最低的宫位（需要注意）。

要求：
1. 必须基于提供的宫位分数数据，优先选择征象最多的宫位，重点关注财运、事业、感情、健康
2. 严格输出 3 条：前 2 条对应分数最高的两个宫位的正向事件，第 3 条对应分数最低的一个宫位的需要注意的事件
3. 直接陈述事实，不要使用“将”“会”“必然”等将来时用词
4. 每条描述要非常具体、明确，包含事件类型、影响程度、具体结果
5. 不要包含宫位名称（如“你的XX宫”），直接描述事件
6. 不要使用模糊词（可能、或许、一些等）
7. 每条 40-80 字，通俗易懂

输出JSON格式：
{{
    "lastYearFortuneReview": ["描述1", "描述2", "描述3"]
}}
"""

    user_prompt = f"""请根据以下年度运势分数，生成3条描述，说明{target_year}年一定会发生的事情。

输出结构：前2条为分数最高的两个宫位的正向事件，第3条为分数最低的一个宫位的需要注意的事件。

要求：
- 基于宫位分数，重点关注财运、事业、感情、健康
- 不要包含宫位名称（如"你的xx宫"）
- 不要使用将来时用词（如"将"、"会"、"必然"）
- 直接描述确定会发生的事件

年度运势分数数据：
{scores_info}

请输出JSON格式。每条lastYearFortuneReview都要具体明确，让人一看就知道会发生什么具体事件。"""

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.85,
        "max_tokens": 800,
        "response_format": {"type": "json_object"}
    }

    try:
        result, _, _ = await aiohttp_vllm_invoke(payload)
        if result and isinstance(result, dict):
            # 处理lastYearFortuneReview
            last_year_fortune_review = result.get("lastYearFortuneReview", [])
            if isinstance(last_year_fortune_review, list):
                # 确保数量正好是3个
                if len(last_year_fortune_review) >= 3:
                    # 取前3个，并清理将来时用词和宫位名称
                    cleaned_review = []
                    for item in last_year_fortune_review[:3]:
                        cleaned = clean_fortune_review_text(item)
                        cleaned_review.append(cleaned)
                    return {"lastYearFortuneReview": cleaned_review}
                elif len(last_year_fortune_review) > 0:
                    # 如果少于3个，补充到3个
                    cleaned_review = []
                    for item in last_year_fortune_review:
                        cleaned = clean_fortune_review_text(item)
                        cleaned_review.append(cleaned)
                    while len(cleaned_review) < 3:
                        cleaned_review.append(generate_default_fortune_review_item(target_year))
                    return {"lastYearFortuneReview": cleaned_review[:3]}
                else:
                    # 如果没有返回，生成默认值
                    default_review = generate_default_fortune_review(target_year, top_palaces_info, bottom_palaces_info)
                    return {"lastYearFortuneReview": default_review[:3]}
            else:
                default_review = generate_default_fortune_review(target_year, top_palaces_info, bottom_palaces_info)
                return {"lastYearFortuneReview": default_review[:3]}
    except Exception as e:
        logger.error(f"生成必然发生的事情失败: {e}", exc_info=True)

    # 默认返回值
    default_review = generate_default_fortune_review(target_year, top_palaces_info, bottom_palaces_info)
    return {"lastYearFortuneReview": default_review[:3]}


def clean_fortune_review_text(text: str) -> str:
    """清理运势描述文本，去掉宫位名称和将来时用词"""
    if not text:
        return text
    
    # 去掉"你的xx宫"这样的模式（保留后面的内容）
    text = re.sub(r'你的[^，。！？\s]+宫[，。！？\s]*', '', text)
    text = re.sub(r'[^，。！？\s]+宫[，。！？\s]*', '', text)
    
    # 替换将来时用词为现在时或直接去掉
    text = text.replace('将会', '').replace('将获得', '获得').replace('将面临', '面临')
    text = text.replace('将出现', '出现').replace('将迎来', '迎来').replace('将出现', '出现')
    text = text.replace('会遇到', '遇到').replace('会带来', '带来').replace('会进一步', '进一步')
    text = text.replace('会获得', '获得').replace('会面临', '面临').replace('会出现', '出现')
    text = text.replace('将', '').replace('会', '').replace('必然', '')
    
    # 清理多余的空格和标点
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'，\s*，', '，', text)
    text = re.sub(r'^\s*[，。！？]\s*', '', text)  # 去掉开头的标点
    text = text.strip()
    
    return text


def generate_default_fortune_review_item(target_year: int) -> str:
    """生成单个默认的必然发生的事情描述"""
    return f"{target_year}年需要保持积极心态，把握机遇，主动应对各种变化。"


def generate_default_fortune_review(target_year: int, top_palaces_info: list, bottom_palaces_info: list) -> list:
    """生成默认的必然发生的事情描述（3条）"""
    # 宫位到具体事件的映射（不包含宫位名称）
    palace_event_map = {
        "事业宫": "事业上有调岗，但是未获得晋升或资源倾斜。",
        "财帛宫": "收入出现明显变化，需要谨慎处理投资和支出。",
        "夫妻宫": "年中因沟通问题导致感情出现摩擦或冷战期。",
        "疾厄宫": "身体状态需要特别关注，注意劳逸结合。",
        "迁移宫": "出差、外派或异地发展机会被取消或推迟。",
        "交友宫": "人际关系出现明显变化，需要谨慎处理。",
        "子女宫": "与子女或晚辈的关系出现变化。",
        "田宅宫": "房产或居住环境面临变动。",
        "福德宫": "心态或情绪出现波动。",
        "父母宫": "与长辈或领导的关系需要关注。",
        "兄弟宫": "与手足或朋友的关系出现变化。",
        "命宫": "整体运势出现重要变化。"
    }

    pos_events = []
    neg_events = []

    # 取前2个高分宫位生成正向事件
    if top_palaces_info:
        for info in top_palaces_info[:2]:
            palace = info.split("（")[0] if "（" in info else ""
            if palace:
                event_desc = palace_event_map.get(palace, "出现重要变化。")
                pos_events.append(event_desc)
    
    # 取1个低分宫位生成需要注意的事件
    if bottom_palaces_info:
        for info in bottom_palaces_info[:1]:
            palace = info.split("（")[0] if "（" in info else ""
            if palace:
                event_desc = palace_event_map.get(palace, "出现重要变化。")
                neg_events.append(event_desc)
    
    default_descriptions = pos_events[:2] + neg_events[:1]
    
    # 确保总数为3条
    while len(default_descriptions) < 3:
        default_descriptions.append(generate_default_fortune_review_item(target_year))

    return default_descriptions[:3]


# --- FastAPI 应用 ---
app = FastAPI(
    title="紫微斗数AI年度报告API (优化版)",
    description="一个针对固定输入格式和年度报告需求的优化版AI接口。",
    version="1.0.0"
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.on_event("startup")
async def startup_event():
    """
    应用启动时执行的事件处理函数，用于初始化全局客户端。
    """
    await db_manager.init_db_pool()
    global async_aiohttp_client, rules_df
    
    logger.info("应用启动，正在初始化全局客户端...")
    
    # 异步加载Excel文件（使用线程池）
    logger.info("正在异步加载Excel规则文件...")
    loop = asyncio.get_event_loop()
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        rules_df = await loop.run_in_executor(executor, _load_rules_df_sync)
        logger.info(f"Excel规则文件加载完成，共 {len(rules_df)} 条规则。")
    except Exception as e:
        logger.error(f"加载Excel文件失败: {e}", exc_info=True)
        raise RuntimeError(f"无法加载Excel规则文件: {e}")
    finally:
        executor.shutdown(wait=False)

    connector = aiohttp.TCPConnector(
        limit=1000,
        limit_per_host=1000,
        enable_cleanup_closed=True,
        force_close=False,
        keepalive_timeout=120
    )
    async_aiohttp_client = aiohttp.ClientSession(
       connector=connector,
       timeout=aiohttp.ClientTimeout(
           connect=VLLM_REQUEST_TIMEOUT_SECONDS,
           sock_read=None
       ),
       headers={
           "Authorization": f"Bearer {API_KEY}",
           "Content-Type": "application/json"
       }
   )
    logger.info("AIOHTTP客户端初始化完成。")


@app.on_event("shutdown")
async def shutdown_event():
    """
    应用关闭时执行的事件处理函数，用于清理全局客户端。
    """
    logger.info("应用关闭，正在清理全局客户端...")
    if async_aiohttp_client:
        await async_aiohttp_client.close()
    await db_manager.close_db_pool()


@app.post("/lastYearFortuneReview", summary="生成年度运势报告 (固定格式输入)")
async def chat_year(request_body: dict = Body(...)):
    """
    处理用户请求，从JSON格式中直接获取出生信息和目标年份，
    并生成指定年份的整体运势年度报告。
    """
    REQUESTS_RECEIVED.inc()

    monitor_request_id = generate_request_id()
    overall_start = time.perf_counter()
    session_id: Optional[str] = None
    app_id: str = "default_app"
    request_status = "成功"

    try:
        with StepMonitor(
            "获取请求",
            monitor_request_id,
            {"request_body_keys": list(request_body.keys())},
        ) as request_step:
            request = APIRequest.model_validate(request_body)
            session_id = request.session_id or str(uuid.uuid4())
            app_id = "default_app"
            request_step.extra_data.update({"session_id": session_id, "app_id": app_id})
        logger.info(f"--- [会话: {session_id}] 收到新年度报告请求 ---")
    except ValidationError as e:
        request_status = "失败"
        logger.error(f"请求体验证失败: {e}")
        raise HTTPException(status_code=422, detail=f"请求体验证失败: {e}")

    try:
        with StepMonitor("解析出生信息", monitor_request_id, {"session_id": session_id}) as birth_step:
            birth_info_to_use = parse_birthday_string(request.birthday)
            birth_info_to_use["gender"] = request.gender
            birth_step.extra_data["gender"] = request.gender
        logger.info(f"解析出生信息成功: {birth_info_to_use}")

        with StepMonitor("构建提示词", monitor_request_id, {"session_id": session_id}) as prompt_step:
            target_year = 2025  # 固定为2025年
            horoscope_date_for_api = f"{target_year}-12-31 12:00:00"
            base_query = f"{target_year}年年度运势分析"
            # 需要计算所有宫位的分数，用于生成预测
            all_palaces = ["命宫", "兄弟宫", "夫妻宫", "子女宫", "财帛宫", "疾厄宫", 
                          "迁移宫", "交友宫", "事业宫", "田宅宫", "福德宫", "父母宫"]
            query_intent_data = {
                "intent_type": "horoscope_analysis",
                "analysis_level": "yearly",
                "target_year": target_year,
                "target_month": 12,
                "target_day": 31,
                "target_hour": 12,
                "target_minute": 0,
                "relative_time_indicator": f"{target_year}年",
                "relevant_palaces": all_palaces,  # 需要计算所有宫位的分数
                "is_query_about_other_person": False,
            }
            prompt_step.extra_data.update(
                {"target_year": target_year}
            )
        logger.info(f"目标年份: {target_year}")

        with StepMonitor("请求紫微API", monitor_request_id, {"session_id": session_id, "target_year": target_year}):
            time_index_for_api_payload = calculate_time_index(
                birth_info_to_use["hour"],
                birth_info_to_use["minute"],
            )
            if time_index_for_api_payload is None:
                raise ValueError(
                    f"无法根据提供的时辰信息计算 timeIndex。输入: Hour={birth_info_to_use['hour']}, Minute={birth_info_to_use['minute']}"
                )
            payload_api_yearly = {
                "dateStr": f"{birth_info_to_use['year']}-{birth_info_to_use['month']:02d}-{birth_info_to_use['day']:02d}",
                "type": "solar",
                "timeIndex": time_index_for_api_payload,
                "gender": birth_info_to_use.get("gender"),
                "horoscopeDate": horoscope_date_for_api,
            }
            api_response_data_yearly = await robust_api_call_with_retry(
                session=async_aiohttp_client,
                url=ZIWEI_API_URL,
                payload=payload_api_yearly,
                timeout=ZIWEI_API_TIMEOUT_SECONDS,
            )
        logger.info("成功从外部紫微API获取年度数据。")

        with StepMonitor("计算年度分数", monitor_request_id, {"session_id": session_id, "target_year": target_year}):
            (
                analysis_data_result,
                selected_prompt_key_for_horoscope_yearly,
                final_score_json_str,
                full_analysis_report,
                summary_output_yearly,
                xingzhi,
                final_sum,
            ) = await generate_horoscope_analysis(
                birth_info_to_use,
                horoscope_date_for_api,
                "yearly",
                api_response_data_yearly,
                query_intent_data,
                base_query,
            )
        logger.info("紫微斗数核心年度分析数据生成完成。")

        with StepMonitor("生成年度预测", monitor_request_id, {"session_id": session_id, "target_year": target_year}):
            user_info_for_domains = {
                "gender": birth_info_to_use.get("gender"),
            }
            annual_keywords = await generate_annual_report_by_domains(
                result_list={},
                user_info=user_info_for_domains,
                analysis_scope="",
                question="",
                full_analysis_report={},
                xingzhi="",
                analysis_data_result={},
                relevant_palaces_for_report=[],
                month_data_raw=None,
                final_score_json_str=final_score_json_str,
                target_year=target_year,
                final_sum=final_sum,
            )

        # 从annual_keywords中提取lastYearFortuneReview（已整合在generate_annual_keywords中）
        last_year_fortune_review = annual_keywords.get("lastYearFortuneReview", [])
        
        # 只返回lastYearFortuneReview
        final_json_response = {
            "lastYearFortuneReview": last_year_fortune_review,
        }

        from fastapi.responses import JSONResponse

        return JSONResponse(content=final_json_response, media_type="application/json; charset=utf-8")

    except ValueError as e:
        request_status = "失败"
        logger.error(f"ValueError: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"输入格式错误: {str(e)}")
    except HTTPException as e:
        request_status = "失败"
        logger.error(f"HTTPException: {e}", exc_info=True)
        raise
    except aiohttp.ClientError as e:
        request_status = "失败"
        logger.error(f"aiohttp.ClientError: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"外部API请求失败: {str(e)}")
    except Exception as e:
        request_status = "失败"
        logger.error(f"未知异常: {e}", exc_info=True)
        logger.error(f"详细追溯: \n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")
    finally:
        total_time = round(time.perf_counter() - overall_start, 3)
        final_extra: Dict[str, Any] = {"total_time_sec": total_time}
        if session_id:
            final_extra["session_id"] = session_id
        log_step("输出给客户端", monitor_request_id, final_extra, status=request_status)

        if session_id:
            try:
                with StepMonitor(
                    "更新数据库用量",
                    monitor_request_id,
                    {"session_id": session_id, "app_id": app_id},
                ):
                    await db_manager.upsert_api_usage_stats(session_id, app_id)
                logger.info(f"用量统计已更新: session_id={session_id}, app_id={app_id}")
            except Exception as e:
                logger.error(f"更新用量统计时发生错误: {e}", exc_info=True)


if __name__ == "__main__":
    print("\n--- 启动说明 ---")
    print("uvicorn api_main:app --host 0.0.0.0 --port 7079 --reload")

    uvicorn.run("api_main:app", host="0.0.0.0", port=7079, reload=True)
