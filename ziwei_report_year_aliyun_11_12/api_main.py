import uvicorn
import os
import uuid
import json
import re
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Literal, Optional, AsyncGenerator
from typing import List, Dict, Tuple, Optional
import asyncio
import hmac
import hashlib
from zhdate import ZhDate
from astro_api_client import AstroAPIClient
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
from starlette.responses import StreamingResponse
from prometheus_client import Counter, make_asgi_app
import prometheus_client

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
    replace_mingnian_with_year,
    convert_chinese_month_to_number,
    replace_chinese_months_in_text,
    sort_months_by_number,
    convert_and_sort_months_in_dimensions,
    get_lunar_first_days,
    get_time_period_description,
    aggregate_monthly_data_by_period,
    extract_score_from_data,
    calculate_time_index,
    normalize_score,
    verify_signature
)

# api_main.py 顶部添加
from prompts import (
    NATURAL_CONVERSATION_ROLE,
    OSSP_XML_TEMPLATE_STR3,
    MING_GONG_DOMAIN_TEMPLATE,
    CAREER_DOMAIN_TEMPLATE,
    WEALTH_DOMAIN_TEMPLATE,
    EMOTION_DOMAIN_TEMPLATE,
    TRAVEL_DOMAIN_TEMPLATE,
    HEALTH_DOMAIN_TEMPLATE,
    FRIENDS_DOMAIN_TEMPLATE,
    FUD_DOMAIN_TEMPLATE,
    TIANZAI_DOMAIN_TEMPLATE,
    XIONGDI_DOMAIN_TEMPLATE,
    FUMU_DOMAIN_TEMPLATE,
    ZINV_DOMAIN_TEMPLATE
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


VLLM_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VLLM_MODEL_NAME = "qwen3-next-80b-a3b-instruct"
API_KEY = "sk-81d4dbe056f94030998f0639f709bff4"

VLLM_REQUEST_TIMEOUT_SECONDS = 100
ZIWEI_API_URL = "http://43.242.97.25:3000/astro_with_option"
ZIWEI_API_TIMEOUT_SECONDS = 250

VLLM_CONCURRENT_LIMIT = 100
VLLM_SLOT_WAIT_TIMEOUT_SECONDS = 500


async_aiohttp_client: Optional[aiohttp.ClientSession] = None
vllm_semaphore = asyncio.Semaphore(VLLM_CONCURRENT_LIMIT)
logger.info(f"VLLM并发访问限制已设置为: {VLLM_CONCURRENT_LIMIT}")

excel_path = r'xyhs.xlsx'
sheet_name = '组合性质判断'

try:
    rules_df = pd.read_excel(excel_path, sheet_name=sheet_name, keep_default_na=False)
    print(f"成功从 '{excel_path}' 的 '{sheet_name}' 工作表中加载 {len(rules_df)} 条规则。")
except Exception as e:
    print(f"加载Excel文件时出错: {e}")
    exit()

rules_df['来源'] = '原始'
new_rules = []
for index, rule in rules_df.iterrows():
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
    rules_df = pd.concat([rules_df, new_rules_df], ignore_index=True)
    print(f"根据“杂曜”规则衍生出 {len(new_rules)} 条新的“会照”规则。总规则数变为: {len(rules_df)}")

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

# 需要返回为维度详情的宫位（产品要求的输出）
DIMENSION_PALACES = ["事业宫", "财帛宫", "夫妻宫", "疾厄宫", "交友宫", "迁移宫"]


def analyze_lunar_data(month_data_raw, month_fin_sum_raw, analysis_data_result_month_data_raw,
                       relevant_palaces: List[str]):
    """
    【修正版本】
    根据给定的宫位名称，计算并分析每个月的总分，并返回总分最高的四个月份的详细信息。
    现在将根据领域特定的宫位来计算分数，而不是简单的命宫。
    """

    # 领域到其主要代表宫位的映射（用于总分计算和信息提取）
    # 确保这里的键是宫位名，值也是宫位名或其别名
    domain_to_main_palace_map_candidates = {
        "命宫": ["命宫"],
        "兄弟宫": ["兄弟宫"],
        "夫妻宫": ["夫妻宫"],
        "子女宫": ["子女宫"],
        "财帛宫": ["财帛宫"],
        "疾厄宫": ["疾厄宫"],
        "迁移宫": ["迁移宫"],
        "交友宫": ["交友宫", "仆役宫"],  # 考虑到可能存在的别名
        "事业宫": ["事业宫", "官禄宫"],  # 考虑到可能存在的别名
        "田宅宫": ["田宅宫"],
        "福德宫": ["福德宫"],
        "父母宫": ["父母宫"]
    }

    final_output = {}

    # 遍历所有要分析的领域，这里使用传入的 relevant_palaces
    for domain_name in relevant_palaces:  # 这里的 domain_name 直接就是宫位名

        processed_months_for_domain = []

        # 找到当前宫位可能对应的所有名称
        main_palace_candidates_for_this_domain = domain_to_main_palace_map_candidates.get(domain_name, [domain_name])

        for i in range(len(month_data_raw)):
            try:
                # 确保这些是字典
                scores_data = json.loads(month_data_raw[i])
                fin_sum_data = json.loads(month_fin_sum_raw[i])
                lunar_data = analysis_data_result_month_data_raw[i]  # lunar_data本身就是字典

                domain_score = 0
                combined_info_for_domain = {}

                actual_palace_for_score_lookup = None
                for candidate_palace in main_palace_candidates_for_this_domain:
                    if candidate_palace in scores_data:  # 确保宫位存在分数
                        actual_palace_for_score_lookup = candidate_palace
                        break

                if actual_palace_for_score_lookup:
                    domain_score = scores_data[actual_palace_for_score_lookup]
                    # 确保从 lunar_data 中提取信息时也使用正确的宫位名
                    combined_info_for_domain[domain_name] = {  # 统一用 domain_name 作为键
                        "宫位解释": fin_sum_data.get(actual_palace_for_score_lookup, ""),
                        "流月盘信息": lunar_data.get(actual_palace_for_score_lookup, {}).get("流月盘", {}).get(
                            "宫位信息", "")
                    }
                else:
                    logger.warning(
                        f"农历{i + 1}月，领域 '{domain_name}' 对应的主要宫位 '{main_palace_candidates_for_this_domain}' 未找到分数，将使用0分。")
                    domain_score = 0  # 默认0分
                    # 确保即使没有找到宫位信息，也能有一个结构化的输出，避免None或空
                    combined_info_for_domain[domain_name] = {  # 统一用 domain_name 作为键
                        "宫位解释": "未找到相关宫位解释。",
                        "流月盘信息": "未找到流月盘信息。"
                    }

                processed_months_for_domain.append({
                    "month_index": i + 1,  # 月份索引从1开始
                    "total_score": domain_score,
                    "combined_info": combined_info_for_domain
                })
            except (json.JSONDecodeError, IndexError, TypeError) as e:
                logger.error(f"处理农历{i + 1}月领域 '{domain_name}' 数据时出错: {e}", exc_info=True)
                continue

        # 对每个领域的好坏月份进行聚合 (调用更新后的 aggregate_monthly_data_by_period)
        aggregated_monthly_data = aggregate_monthly_data_by_period(
            processed_months_for_domain,  # 直接传递字典列表
            get_time_period_description(processed_months_for_domain)
        )

        final_output[domain_name] = {
            "time_period": get_time_period_description(processed_months_for_domain),
            "monthly_analysis": aggregated_monthly_data
        }
    return final_output


# --- Pydantic模型 (为请求体新增签名所需字段) ---
class SignableAPIRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="会话ID")
    gender: str = Field(..., description="性别：'男' 或 '女'")
    birthday: str = Field(..., description="出生日期时间，格式：'YYYY-MM-DD HH:MM:SS'，例如：'1995-05-06 14:30:00'")
    year: int = Field(..., description="要分析的年份，例如：2026")


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

                    logger.info(f"VLLM调用成功 (第 {attempt + 1} 次尝试)。")
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


def generate_horoscope_analysis(birth_info: Dict[str, Any], horoscope_date_str: Optional[str], analysis_level: str,
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


def preprocess_xingzhi_data(raw_xingzhi_string: str) -> Dict[str, List[Dict[str, Any]]]:
    # Domain name mapping for consistency, especially '官禄宫' to '事业宫'
    palace_name_mapping_for_xingzhi = {
        "事业": "官禄宫", "财富": "财帛宫", "感情": "夫妻宫", "出行": "迁移宫",
        "健康": "疾厄宫", "朋友": "交友宫", "整体运势": "命宫",
        "福德": "福德宫", "田宅": "田宅宫", "兄弟": "兄弟宫", "父母": "父母宫", "子女": "子女宫"  # 添加新宫位
    }
    domain_name_mapping = {v: k for k, v in palace_name_mapping_for_xingzhi.items()}

    cleaned_string = raw_xingzhi_string.strip()
    parts = re.split(r'(宫位：[^,]+,)', cleaned_string)

    processed_data: Dict[str, List[Dict[str, Any]]] = {}

    for i in range(1, len(parts), 2):
        header = parts[i]
        content = parts[i + 1]

        match = re.search(r'宫位：([^,]+)', header)
        if match:
            current_palace = match.group(1).strip()
            # Standardize palace name for storage in processed_data
            standardized_palace = current_palace.replace("官禄宫", "事业宫").replace("仆役宫", "交友宫")

            content_match = re.search(
                r'性质:\s*([^,]+),\s*描述:\s*([\s\S]*?),\s*性质得分:\s*([^\s]+.*)$',
                content.strip()
            )

            if content_match:
                xingzhi_data = {
                    "性质": content_match.group(1).strip(),
                    "描述": content_match.group(2).strip(),
                    "性质得分": content_match.group(3).strip(),
                }

                if standardized_palace not in processed_data:
                    processed_data[standardized_palace] = []

                processed_data[standardized_palace].append(xingzhi_data)
    return processed_data


async def analyze_single_domain(
        domain_name: str,
        domain_data: dict,
        monthly_data: dict,  # monthly_data 现在是字典类型
        user_info: dict,
        analysis_scope: str,
        question: str,
        xingzhi: str,
        analysis_data_result: Dict[str, Any],
        full_analysis_report: dict,
) -> str:
    """
    分析单个领域的运势
    """
    all_domain_templates = {
        "命宫": MING_GONG_DOMAIN_TEMPLATE,
        "兄弟宫": XIONGDI_DOMAIN_TEMPLATE,
        "夫妻宫": EMOTION_DOMAIN_TEMPLATE,
        "子女宫": ZINV_DOMAIN_TEMPLATE,
        "财帛宫": WEALTH_DOMAIN_TEMPLATE,
        "疾厄宫": HEALTH_DOMAIN_TEMPLATE,
        "迁移宫": TRAVEL_DOMAIN_TEMPLATE,
        "交友宫": FRIENDS_DOMAIN_TEMPLATE,
        "事业宫": CAREER_DOMAIN_TEMPLATE,
        "田宅宫": TIANZAI_DOMAIN_TEMPLATE,
        "福德宫": FUD_DOMAIN_TEMPLATE,
        "父母宫": FUMU_DOMAIN_TEMPLATE
    }

    # 修正：这些映射的键是模板中期望的变量名，值是领域名称
    # 这里的键必须是模板文件中定义的具体变量名
    domain_data_param_map = {
        "命宫": "overall_specific_data",
        "兄弟宫": "xiongdi_specific_data",
        "夫妻宫": "emotion_specific_data",
        "子女宫": "zinv_specific_data",
        "财帛宫": "wealth_specific_data",
        "疾厄宫": "health_specific_data",
        "迁移宫": "travel_specific_data",
        "交友宫": "friends_specific_data",
        "事业宫": "career_specific_data",
        "田宅宫": "tianzai_specific_data",
        "福德宫": "fude_specific_data",
        "父母宫": "fumu_specific_data",
    }
    monthly_data_param_map = {
        "命宫": "overall_monthly_data",
        "兄弟宫": "xiongdi_monthly_data",
        "夫妻宫": "emotion_monthly_data",
        "子女宫": "zinv_monthly_data",
        "财帛宫": "wealth_monthly_data",
        "疾厄宫": "health_monthly_data",
        "迁移宫": "travel_monthly_data",
        "交友宫": "friends_monthly_data",
        "事业宫": "career_monthly_data",
        "田宅宫": "tianzai_monthly_data",
        "福德宫": "fude_monthly_data",
        "父母宫": "fumu_monthly_data",
    }
    palace_info_param_map = {
        "命宫": "overall_palace_info",
        "兄弟宫": "xiongdi_palace_info",
        "夫妻宫": "emotion_palace_info",
        "子女宫": "zinv_palace_info",
        "财帛宫": "wealth_palace_info",
        "疾厄宫": "health_palace_info",
        "迁移宫": "travel_palace_info",
        "交友宫": "friends_palace_info",
        "事业宫": "career_palace_info",
        "田宅宫": "tianzai_palace_info",
        "福德宫": "fude_palace_info",
        "父母宫": "fumu_palace_info",
    }

    if domain_name not in all_domain_templates:
        logger.error(f"未知的领域名称: {domain_name}")
        return f"**标题：{domain_name}**\n\n抱歉，暂时无法分析该领域。"

    template_str = all_domain_templates[domain_name]

    # 从 full_analysis_report 中提取当前宫位的信息 (domain_name 现在是宫位名)
    palace_info = full_analysis_report.get(domain_name, "该宫位信息暂时无法获取")

    def extract_flow_year_info_regex(palace_content: str, target_palace: str) -> str:
        if not isinstance(palace_content, str):
            return f"▶ 进入【流年】，您的\"{target_palace}\"暂时无法获取详细信息。"
        pattern = r'▶ 进入【流年】(.*?)(?=▶|$)'
        match = re.search(pattern, palace_content, re.DOTALL)
        if match:
            flow_year_content = match.group(1).strip()
            return f"▶ 进入【流年】{flow_year_content}"
        basic_pattern = r'您的"' + re.escape(target_palace) + r'"位于【[^】]+】宫，由主星【[^】]+】坐守[^。]*。'
        basic_match = re.search(basic_pattern, palace_content)
        if basic_match:
            return basic_match.group(0)
        return palace_content[:200] + "..." if len(palace_content) > 200 else palace_content

    palace_flow_info = extract_flow_year_info_regex(palace_info, domain_name)  # 直接使用 domain_name 作为 target_palace

    # 命宫信息总是作为额外上下文传入
    ming_gong_info = full_analysis_report.get("命宫", "命宫信息暂时无法获取")
    ming_gong_flow_info = extract_flow_year_info_regex(ming_gong_info, "命宫")

    # monthly_data 现在直接就是该领域/宫位的聚合月度数据字典
    quarterly_data_to_use = monthly_data["monthly_analysis"] if isinstance(monthly_data,
                                                                           dict) and "monthly_analysis" in monthly_data else {
        "好月份": [], "坏月份": []}
    time_period_info = monthly_data["time_period"] if isinstance(monthly_data,
                                                                 dict) and "time_period" in monthly_data else "全年"

    try:
        preprocess_xingzhi_data_xingzhi = preprocess_xingzhi_data(xingzhi)
        # 修正: 确保能从preprocess_xingzhi_data_xingzhi中正确获取到领域对应的性质数据
        xingzhi_for_domain = preprocess_xingzhi_data_xingzhi.get(domain_name, [])

        # 获取该领域对应的 `analysis_data_result` 中的具体宫位信息
        palace_info_from_analysis_data_result = analysis_data_result.get(domain_name, {})  # 直接使用 domain_name

        # 动态构建模板参数字典
        template_kwargs = {
            "natural_conversation_role": NATURAL_CONVERSATION_ROLE,
            "question": question,
            "analysis_scope": analysis_scope,  # 修正: 使用 analysis_scope
            "user_solar_date_display": user_info.get('solar_date', ''),
            "user_chinese_date_display": user_info.get('chinese_date', ''),
            "ming_gong_info": ming_gong_flow_info,
            "xingzhi": xingzhi_for_domain,
            "analysis_data_result": json.dumps(palace_info_from_analysis_data_result, ensure_ascii=False),
            # 这里使用正确的模板变量名作为键，并确保其存在
            # 修正: 使用 .get() 提供默认值，防止 KeyErrors
            domain_data_param_map.get(domain_name, 'unknown_specific_data'): json.dumps(domain_data, ensure_ascii=False,
                                                                                        indent=2),
            monthly_data_param_map.get(domain_name, 'unknown_monthly_data'): json.dumps({
                "time_period": time_period_info,
                "monthly_details": quarterly_data_to_use
            }, ensure_ascii=False, indent=2),
            palace_info_param_map.get(domain_name, 'unknown_palace_info'): palace_flow_info
        }

        formatted_template = template_str.format(**template_kwargs)

        # --- 增强日志：打印 LLM 最终 System Prompt 内容 ---
        # print(f"{domain_name}formatted_template",formatted_template)

    except Exception as e:
        logger.error(f"{domain_name}领域模板格式化失败: {e}", exc_info=True)
        return f"**标题：{domain_name}**\n\n模板格式化失败: {str(e)}"

    # 只对需要JSON输出的宫位使用JSON模式
    if domain_name in DIMENSION_PALACES:
        vllm_payload = {
            "model": VLLM_MODEL_NAME,
            "messages": [
                {"role": "system", "content": formatted_template},
                {"role": "user", "content": f"请分析我的{domain_name}运势"}
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"}
        }

        try:
            json_result, input_tokens, output_tokens = await aiohttp_vllm_invoke(vllm_payload)
            if json_result and isinstance(json_result, dict):
                logger.info(f"{domain_name}领域JSON分析完成，输入tokens: {input_tokens}")
                logger.info(f"{domain_name}领域JSON分析完成，输出tokens: {output_tokens}")
                return json_result
            else:
                logger.warning(f"{domain_name}领域分析返回无效JSON")
                return {"error": f"{domain_name}分析失败"}
        except Exception as e:
            logger.error(f"{domain_name}领域JSON分析异常: {e}", exc_info=True)
            return {"error": f"{domain_name}分析异常"}
    else:
        # 其他宫位保持原来的stream模式返回文本
        vllm_payload = {
            "model": VLLM_MODEL_NAME,
            "messages": [
                {"role": "system", "content": formatted_template},
                {"role": "user", "content": f"请分析我的{domain_name}运势"}
            ],
            "temperature": 0.7,
            "stream": True
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
                logger.info(f"{domain_name}领域分析完成，输出tokens: {output_tokens}")
                return content
            else:
                logger.warning(f"{domain_name}领域分析返回空内容")
                return f"**标题：{domain_name}**\n\n分析过程中出现问题，请稍后重试。"

        except Exception as e:
            logger.error(f"{domain_name}领域分析异常: {e}", exc_info=True)
            return f"**标题：{domain_name}**\n\n分析过程中出现异常，请稍后重试。"


async def generate_annual_report_by_domains(
        result_list: dict,
        user_info: dict,
        analysis_scope: str,
        question: str,
        full_analysis_report: dict,
        xingzhi: str,
        analysis_data_result: Dict[str, Any],
        relevant_palaces_for_report: List[str],  # New parameter
        month_data_raw: list = None,
) -> tuple:
    """
    分领域生成年报
    """
    logger.info("开始分领域生成年报...")
    # 现在核心领域是 relevant_palaces_for_report，所有请求的宫位都应被分析
    core_domains = relevant_palaces_for_report

    domain_tasks = []
    for domain_name in core_domains:
        # result_list[domain_name] 已经是由 analyze_lunar_data 聚合好的字典
        domain_monthly_data = result_list.get(domain_name, {
            "time_period": "全年",
            "monthly_analysis": {"好月份": [], "坏月份": []}
        })

        domain_core_data = {
            "specific_domain_name": domain_name,  # 用于模板中识别当前是哪个宫位
            "monthly_analysis": domain_monthly_data  # 传递完整的月度分析结果
        }
        task = analyze_single_domain(
            domain_name=domain_name,
            domain_data=domain_core_data,  # 传递给模板的领域特定数据
            monthly_data=domain_monthly_data,  # 传递给模板的月度亮点数据 (现在是字典)
            user_info=user_info,
            analysis_scope=analysis_scope,
            question=question,
            full_analysis_report=full_analysis_report,
            xingzhi=xingzhi,
            analysis_data_result=analysis_data_result
        )
        domain_tasks.append((domain_name, task))

    logger.info(f"开始并发分析 {len(domain_tasks)} 个领域...")

    # 提取所有task和对应的领域名称
    tasks_only = [task for _, task in domain_tasks]
    domain_names = [name for name, _ in domain_tasks]

    # 真正并发执行所有任务
    results = await asyncio.gather(*tasks_only, return_exceptions=True)

    # 处理结果
    domain_results = {}
    for domain_name, result in zip(domain_names, results):
        if isinstance(result, Exception):
            logger.error(f"{domain_name}领域分析失败: {result}", exc_info=True)
            domain_results[domain_name] = {
                "error": f"{domain_name}分析失败"} if domain_name in DIMENSION_PALACES else f"**标题：{domain_name}**\n\n分析过程中出现问题。"
        else:
            domain_results[domain_name] = result
            logger.info(f"{domain_name}领域分析完成")

    # 分离JSON数据（维度宫位）和文本数据（其他宫位）
    dimension_json_list = []
    text_report_parts = []

    for domain in relevant_palaces_for_report:
        if domain in domain_results:
            result = domain_results[domain]
            if domain in DIMENSION_PALACES and isinstance(result, dict):
                # 这是维度宫位，返回的是JSON
                dimension_json_list.append(result)
            elif isinstance(result, str):
                # 这是其他宫位，返回的是文本
                text_report_parts.append(result)

    # 生成年度概述
    year_overview = await generate_year_overview(dimension_json_list, user_info)

    # 生成年度关键词
    annual_keywords = await generate_annual_keywords(domain_results, user_info)

    # 生成每月分数
    monthly_scores = generate_monthly_scores(month_data_raw or [], core_domains)

    # 生成追问建议
    follow_up_questions = await generate_follow_up_questions(user_info.get('gender', '男'), domain_results)

    logger.info("年报分领域生成完成")

    # 返回结构化数据
    return dimension_json_list, year_overview, annual_keywords, monthly_scores, follow_up_questions, text_report_parts


async def generate_year_overview(dimension_json_list: list, user_info: dict) -> str:
    """
    基于所有维度的JSON数据生成年度概述
    """
    # 构建所有维度的概述摘要
    dimensions_summary = ""
    for dim_data in dimension_json_list:
        if isinstance(dim_data, dict) and "dimensionType" in dim_data:
            dim_type = dim_data.get("dimensionType", "")
            dim_overview = dim_data.get("dimensionOverview", "")
            if dim_overview:
                # 取前150字作为摘要
                summary = dim_overview[:150] + "..." if len(dim_overview) > 150 else dim_overview
                dimensions_summary += f"{dim_type}：{summary}\n\n"

    system_prompt = """
    你是一个专业的紫微斗数分析师。请根据用户各个维度的运势分析，生成一份200-300字的年度整体概述。

    要求：
    1. 概述要综合各维度的情况，突出整体趋势和关键特征
    2. 语言要通俗易懂，避免使用专业术语
    3. 重点描述这一年的整体特点、主要机遇和挑战
    4. 字数控制在200-300字之间
    5. 不要使用"您"、"用户"等称呼，直接用"你"
    6. 直接输出概述文本，不要添加任何标题或格式
    """

    user_prompt = f"请根据以下各维度的运势分析，生成年度整体概述：\n\n{dimensions_summary}"

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": True
    }

    try:
        # 使用stream模式获取纯文本
        content_parts = []
        async for chunk_data in aiohttp_vllm_stream(payload):
            if isinstance(chunk_data, tuple) and len(chunk_data) == 3:
                chunk, input_tokens, output_tokens = chunk_data
                if chunk and not chunk.startswith("[错误:"):
                    content_parts.append(chunk)

        if content_parts:
            overview_text = "".join(content_parts).strip()
            if overview_text:
                return overview_text

        logger.warning("生成年度概述失败，返回默认概述")
        return "本年度运势呈现多元化特征，各维度发展不均衡，需要你在不同领域灵活调整策略，抓住机遇，应对挑战。"
    except Exception as e:
        logger.error(f"生成年度概述失败: {e}", exc_info=True)
        return "本年度运势呈现多元化特征，各维度发展不均衡，需要你在不同领域灵活调整策略，抓住机遇，应对挑战。"


async def generate_annual_keywords(domain_results: dict, user_info: dict) -> dict:
    """
    基于各领域分析结果生成年度主题词
    """
    analysis_summary = ""
    # 遍历所有宫位的分析结果，构建摘要
    for domain, content in domain_results.items():
        # 处理JSON格式的维度宫位
        if isinstance(content, dict):
            overview = content.get("dimensionOverview", "")
            if overview and len(overview) > 50:
                summary = overview[:200] + "..." if len(overview) > 200 else overview
                analysis_summary += f"{domain}：{summary}\n\n"
        # 处理文本格式的其他宫位
        elif isinstance(content, str) and len(content) > 50:
            summary = content[:200] + "..." if len(content) > 200 else content
            analysis_summary += f"{domain}：{summary}\n\n"

    system_prompt = """
    你是一个专业的紫微斗数分析师。请根据用户的年度运势分析结果，提取1个最能代表其年度运势特点的主题词，以及对于这个主题词的解释说明。

    要求：
    1. 每个主题词必须是4个字
    2. 主题词要积极正面，体现运势特点
    3. 主题词要简洁有力，朗朗上口
    4. 主题词必须个性化，不能使用"稳中求进"、"稳扎稳打"、"稳步发展"、"稳中有进"等常见通用词汇
    5. 主题词必须基于分析摘要中的具体内容生成，要能体现该用户独特的运势特点
    6. 解释说明必须是50字以上200字以下
    7. 严禁使用复杂的紫微斗数术语，必须用通俗语言解释，例如不能出现类似于：太阳巨门，天同太阴，天机激活这些紫微斗数专业术语词
    8. 输出格式必须为JSON：{"keyword": "四字词语", "explain": "解释说明内容"}

    示例输出（展示不同风格，避免使用通用词汇）：
    示例1：{"keyword": "平凡之年", "explain": "今年是你的平凡之年，生活不会有大的起伏，工作、学习、感情方面进展都比较平稳，不会有什么大的转变或突破，维持着和往常一样的状态，或许缺少了一些惊喜的体验，但也不至于有突发的状况出现，打乱你的节奏。你可以掌控生活的节奏，对凡事不要抱有过高的期待，把握当下，好好沉淀自己，当机遇来临的时候你就可以把握住。"}
    示例2：{"keyword": "破茧成蝶", "explain": "这一年你将经历重要的转变和突破，就像破茧成蝶一样，虽然过程可能充满挑战和不适，但最终会迎来全新的自己。你需要在变化中寻找机会，勇敢地走出舒适区，尝试新的方向和方法。这个转变不仅体现在外在环境上，更重要的是内心的成长和认知的提升。"}
    示例3：{"keyword": "厚积薄发", "explain": "今年是你积累和沉淀的重要一年，虽然表面上可能看不到明显的突破，但你在各个方面都在默默积累能量和资源。就像竹子在地下扎根一样，你需要耐心等待，持续努力，为未来的爆发做好准备。不要急于求成，专注于打好基础，时机成熟时自然会迎来转机。"}
    """

    user_prompt = f"请根据以下年度运势分析，生成1个年度主题词，以及对于这个主题词的解释说明：\n\n{analysis_summary}"

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.85,
        "max_tokens": 200,
        "response_format": {"type": "json_object"}
    }

    try:
        result, _, _ = await aiohttp_vllm_invoke(payload)
        if result and isinstance(result, dict):
            return result
    except Exception as e:
        logger.error(f"生成年度主题词和解释说明失败: {e}", exc_info=True)

    return {"keyword": "平凡之年",
            "explain": "今年是你的平凡之年，生活不会有大的起伏，工作、学习、感情方面进展都比较平稳，不会有什么大的转变或突破，维持着和往常一样的状态，或许缺少了一些惊喜的体验，但也不至于有突发的状况出现，打乱你的节奏。你可以掌控生活的节奏，对凡事不要抱有过高的期待，把握当下，好好沉淀自己，当机遇来临的时候你就可以把握住。"}


def generate_monthly_scores(month_data_raw: list, relevant_palaces: List[str]) -> dict:
    """
    生成每月运势分值
    """
    monthly_scores = {}
    month_names = ["", "一月", "二月", "三月", "四月", "五月", "六月",
                   "七月", "八月", "九月", "十月", "十一月", "十二月"]
    all_domains_with_average = relevant_palaces + ["月度平均分"]  # Only consider relevant palaces and average

    # Initialize all domains and months with default score
    for domain in all_domains_with_average:
        monthly_scores[domain] = {}
        for month_idx in range(1, 13):
            monthly_scores[domain][month_names[month_idx]] = 67.5  # Default score

    # Domain to its main representative palace mapping (for score extraction from scores_data)
    domain_palace_mapping = {
        "命宫": ["命宫"],
        "兄弟宫": ["兄弟宫"],
        "夫妻宫": ["夫妻宫"],
        "子女宫": ["子女宫"],
        "财帛宫": ["财帛宫"],
        "疾厄宫": ["疾厄宫"],
        "迁移宫": ["迁移宫"],
        "交友宫": ["交友宫", "仆役宫"],
        "事业宫": ["事业宫", "官禄宫"],
        "田宅宫": ["田宅宫"],
        "福德宫": ["福德宫"],
        "父母宫": ["父母宫"]
    }

    # Process raw monthly data
    for i, month_score_json in enumerate(month_data_raw):
        if i >= 12: break  # Process at most 12 months

        month_name = month_names[i + 1]

        try:
            scores_data = json.loads(month_score_json) if isinstance(month_score_json, str) else month_score_json

            # Assign scores for each relevant domain
            for domain in relevant_palaces:
                possible_palace_names = domain_palace_mapping.get(domain, [domain])

                score_found = False
                for palace_name in possible_palace_names:
                    if palace_name in scores_data:
                        raw_score = float(scores_data[palace_name])
                        normalized_score = normalize_score(raw_score)
                        monthly_scores[domain][month_name] = normalized_score
                        score_found = True
                        break

                if not score_found:
                    logger.warning(
                        f"月份 {month_name}，宫位 '{domain}' 及其候选宫位 '{possible_palace_names}' 未找到分数，使用默认值。")

            # Calculate monthly average score ONLY for the relevant palaces
            total_score_current_month = 0.0
            num_scored_domains = 0
            for domain in relevant_palaces:
                # Use the score already set (either normalized or default)
                total_score_current_month += monthly_scores[domain][month_name]
                num_scored_domains += 1

            if num_scored_domains > 0:
                monthly_scores["月度平均分"][month_name] = round(total_score_current_month / num_scored_domains, 1)
            else:
                monthly_scores["月度平均分"][month_name] = 67.5  # Fallback if no domains were scored

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"解析第{i + 1}月分数数据时出错: {e}")
            continue

    # Calculate annual average score for each relevant domain
    annual_averages = {}
    for domain in relevant_palaces:  # Only for relevant palaces
        total_score_for_year = 0.0
        for month_idx in range(1, 13):
            total_score_for_year += monthly_scores[domain][month_names[month_idx]]

        annual_average = total_score_for_year / 12
        annual_averages[domain] = round(annual_average, 1)

    # Add annual average for "月度平均分"
    total_avg_score_for_year = 0.0
    for month_idx in range(1, 13):
        total_avg_score_for_year += monthly_scores["月度平均分"][month_names[month_idx]]
    annual_averages["月度平均分"] = round(total_avg_score_for_year / 12, 1)

    # Add annual average scores as a separate field
    monthly_scores["各领域年度平均分"] = annual_averages

    return monthly_scores


async def generate_follow_up_questions(gender: str, domain_results: dict) -> list:
    """
    根据性别和运势分析生成10个追问问题
    """
    analysis_summary = ""
    # 遍历所有宫位的分析结果，构建摘要
    for domain, content in domain_results.items():
        # 处理JSON格式的维度宫位
        if isinstance(content, dict):
            overview = content.get("dimensionOverview", "")
            if overview and len(overview) > 50:
                summary = overview[:300] + "..." if len(overview) > 300 else overview
                analysis_summary += f"{domain}：{summary}\n\n"
        # 处理文本格式的其他宫位
        elif isinstance(content, str) and len(content) > 50:
            summary = content[:300] + "..." if len(content) > 300 else content
            analysis_summary += f"{domain}：{summary}\n\n"

    if gender == "男":
        priority_focus = "事业发展和财富积累"
        secondary_focus = "感情、健康等其他方面"
    else:
        priority_focus = "感情生活和财富管理"
        secondary_focus = "事业、健康等其他方面"

    system_prompt = f"""
    你是一个专业的紫微斗数分析师。请根据用户的年度运势分析结果，生成10个具有针对性的追问问题。

    用户性别：{gender}
    重点关注：{priority_focus}
    次要关注：{secondary_focus}

    要求：
    1. 生成10个问题，JSON格式：{{"questions": ["问题1", "问题2", ...]}}
    2. 问题要结合具体的运势分析内容
    3. 根据性别特点，{gender}性用户应优先关注{priority_focus}相关问题
    4. 问题要具体、实用，能帮助用户更好地理解和应用运势信息
    5. 避免过于宽泛的问题，要有针对性
    6. 问题应该是用户可能真正想了解的内容
    7. **重要**：问题中不要包含专业术语如"天同化禄"、"命宫"、"财帛宫"等宫位名称，要用通俗语言表达
    8. 问题应该从用户的角度出发，用"我"开头，关注实际生活场景

    示例格式：
    {{"questions": ["我在哪个季度的事业运势最佳？", "我在感情中总是想等'稳妥了再推进'，这会不会错过最佳表达心意的时机？", ...]}}
    """

    user_prompt = f"请根据以下运势分析，为{gender}性用户生成10个追问问题：\n\n{analysis_summary}"

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.8,
        "max_tokens": 500,
        "response_format": {"type": "json_object"}
    }

    try:
        result, _, _ = await aiohttp_vllm_invoke(payload)
        if result and isinstance(result, dict) and "questions" in result:
            questions = result["questions"]
            if len(questions) >= 10:
                return questions[:10]
            else:
                return questions + get_default_questions(gender)[len(questions):10]
    except Exception as e:
        logger.error(f"生成追问问题失败: {e}", exc_info=True)

    return get_default_questions(gender)


def get_default_questions(gender: str) -> list:
    """获取默认的追问问题"""
    if gender == "男":
        return [
            "我在事业发展上应该重点关注哪些方面？",
            "今年的财运高峰期在什么时候？",
            "如何把握事业上的关键机遇？",
            "投资理财方面有什么需要注意的？",
            "事业转型的最佳时机是什么时候？",
            "如何提升自己的财富积累能力？",
            "在职场人际关系上有什么建议？",
            "感情方面需要特别留意什么？",
            "健康养生应该从哪些方面入手？",
            "今年适合进行哪些重要决策？"
        ]
    else:
        return [
            "我的感情运势在今年会有什么变化？",
            "财运方面有哪些需要把握的机会？",
            "如何处理感情中的挑战和困扰？",
            "理财投资上有什么好的建议？",
            "桃花运最旺的时期是什么时候？",
            "如何提升自己的财富管理能力？",
            "在人际关系上需要注意什么？",
            "事业发展有什么新的契机？",
            "健康方面需要重点关注哪些？",
            "今年最适合做出的重要改变是什么？"
        ]


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
    global async_aiohttp_client
    logger.info("应用启动，正在初始化全局客户端...")

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


@app.post("/chat_year_V11_11", summary="生成年度运势报告 (固定格式输入)")
async def chat_year(request_body: dict = Body(...)):
    """
    处理用户请求，从JSON格式中直接获取出生信息和目标年份，
    并生成指定年份的整体运势年度报告。
    """
    REQUESTS_RECEIVED.inc()

    try:
        request = SignableAPIRequest.model_validate(request_body)
    except ValidationError as e:
        logger.error(f"请求体验证失败: {e}")
        raise HTTPException(status_code=422, detail=f"请求体验证失败: {e}")

    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"--- [会话: {session_id}] 收到新年度报告请求 ---")

    try:
        # --- 步骤 1: 解析出生信息 ---
        birth_info_to_use = parse_birthday_string(request.birthday)
        birth_info_to_use["gender"] = request.gender  # 添加性别信息
        logger.info(f"解析出生信息成功: {birth_info_to_use}")

        # --- 步骤 2: 固定分析维度（年度报告默认分析主要宫位）---
        target_year = request.year
        relevant_palaces_for_query = DIMENSION_PALACES  # 使用固定的维度宫位：["命宫", "事业宫", "财帛宫", "夫妻宫", "疾厄宫", "交友宫", "迁移宫"]
        logger.info(f"目标年份: {target_year}，分析宫位: {relevant_palaces_for_query}")
        horoscope_date_for_api = f"{target_year}-12-31 12:00:00"

        # 添加缺失的变量定义（在 target_year 定义之后）
        base_query = f"{target_year}年年度运势分析"  # 用于生成报告的默认查询文本
        app_id = getattr(request, 'appid', 'default_app')  # 如果请求中有appid则使用，否则使用默认值

        query_intent_data = {
            "intent_type": "horoscope_analysis",
            "analysis_level": "yearly",
            "target_year": target_year,
            "target_month": 12,  # 对应12月31日
            "target_day": 31,  # 对应12月31日
            "target_hour": 12,
            "target_minute": 0,
            "relative_time_indicator": f"{target_year}年",
            "relevant_palaces": relevant_palaces_for_query,  # Use dynamically determined palaces
            "is_query_about_other_person": False,
        }
        logger.info(f"最终意图数据 (包含动态宫位): {query_intent_data}")

        # --- 步骤 3: 调用外部紫微API获取数据 (本命盘和流年盘) ---
        time_index_for_api_payload = calculate_time_index(
            birth_info_to_use['hour'],
            birth_info_to_use['minute']
        )
        if time_index_for_api_payload is None:
            raise ValueError(
                f"无法根据提供的时辰信息计算 timeIndex。输入: Hour={birth_info_to_use['hour']}, Minute={birth_info_to_use['minute']}")

        payload_api_yearly = {
            "dateStr": f"{birth_info_to_use['year']}-{birth_info_to_use['month']:02d}-{birth_info_to_use['day']:02d}",
            "type": "solar",  # 固定为公历
            "timeIndex": time_index_for_api_payload,  # 使用计算出的 timeIndex
            "gender": birth_info_to_use.get("gender"),
            "horoscopeDate": horoscope_date_for_api
        }
        logger.info(
            f"即将调用外部紫微API获取年度数据，Payload: {json.dumps(payload_api_yearly, indent=2, ensure_ascii=False)}")

        api_response_data_yearly = await robust_api_call_with_retry(
            session=async_aiohttp_client,
            url=ZIWEI_API_URL,
            payload=payload_api_yearly,
            timeout=ZIWEI_API_TIMEOUT_SECONDS
        )
        logger.info("成功从外部紫微API获取年度数据。")

        # --- 步骤 4: 获取流月数据 for monthly scores ---
        client_astr = AstroAPIClient(
            api_url=ZIWEI_API_URL,
            birth_info={
                "dateStr": f"{birth_info_to_use['year']}-{birth_info_to_use['month']:02d}-{birth_info_to_use['day']:02d}",
                "type": "solar",
                "timeIndex": time_index_for_api_payload,  # 使用计算出的 timeIndex
                "gender": birth_info_to_use.get("gender")},
            astro_type="heaven",
            query_year=target_year,  # 查询明年的月份数据
            save_results=False  # 不需要保存到文件
        )

        results_astr_list = client_astr.run_query(target_year)
        logger.info(f"成功从 AstroAPIClient 获取 {target_year} 年的 {len(results_astr_list)} 个月份数据。")

        month_data = []
        month_fin_sum = []
        analysis_data_result_month_data = []

        for month_api_result_item in results_astr_list:
            month_api_data = month_api_result_item['api_result']['data']

            # Pass the dynamically determined relevant_palaces to generate_horoscope_analysis for monthly data
            monthly_query_intent_data = query_intent_data.copy()
            monthly_query_intent_data[
                "relevant_palaces"] = relevant_palaces_for_query  # Ensure monthly data generation also respects filtering

            analysis_data_result_month, selected_prompt_key_for_horoscope_month, final_score_json_str_month, full_analysis_report_month, summary_output_month, xingzhi_month, final_sum_month = \
                generate_horoscope_analysis(
                    birth_info_to_use,
                    None,
                    "monthly",
                    month_api_data,
                    monthly_query_intent_data,  # Use monthly specific intent data
                    base_query  # Use base_query for monthly generation
                )

            month_data.append(final_score_json_str_month)
            month_fin_sum.append(final_sum_month)
            analysis_data_result_month_data.append(analysis_data_result_month)
        logger.info("所有流月数据处理完成。")

        # --- 步骤 5: 生成主报告数据 (年度) ---
        analysis_data_result, selected_prompt_key_for_horoscope_yearly, final_score_json_str, full_analysis_report, summary_output_yearly, xingzhi, final_sum = \
            generate_horoscope_analysis(
                birth_info_to_use,  # 传入包含 hour/minute 的 birth_info
                horoscope_date_for_api,
                "yearly",
                api_response_data_yearly,
                query_intent_data,  # Use original query_intent_data with relevant_palaces
                base_query  # Use base_query for yearly generation
            )
        logger.info("紫微斗数核心年度分析数据生成完成。")
        # --- 增强日志结束 ---

        # --- 步骤 6: 聚合月度数据 (为年报) ---
        result_list_aggregated = analyze_lunar_data(
            month_data, month_fin_sum, analysis_data_result_month_data, relevant_palaces_for_query
        )
        logger.info("月度数据聚合完成。")
        monthly_scores_for_output = generate_monthly_scores(month_data, relevant_palaces_for_query)
        # --- 增强日志：打印月度分数 (`generate_monthly_scores` 的结果) ---
        # logger.info(f"--- Monthly Scores Data ---\n{json.dumps(monthly_scores_for_output, ensure_ascii=False, indent=2)}\n--- END Monthly Scores Data ---")
        # --- 增强日志结束 ---

        # --- 步骤 7: 生成年度报告内容 ---
        astrolabe_data_raw = api_response_data_yearly.get('data', {}).get('astrolabe', {})
        astrolabe_data = astrolabe_data_raw if isinstance(astrolabe_data_raw, dict) else {}

        chinese_date_obj_raw = astrolabe_data.get('chineseDate', {})
        chinese_date_obj = chinese_date_obj_raw if isinstance(chinese_date_obj_raw, dict) else {}

        user_info_for_domains = {
            "solar_date": astrolabe_data.get('solarDate', ''),
            "chinese_date": chinese_date_obj.get('full', ''),
            "gender": birth_info_to_use.get('gender')
        }
        analysis_time_scope_str = f"针对{target_year}年的流年运势分析"

        dimension_json_list, year_overview, annual_keywords, monthly_scores_final, follow_up_questions, text_report_parts = await generate_annual_report_by_domains(
            result_list=result_list_aggregated,
            user_info=user_info_for_domains,
            analysis_scope=analysis_time_scope_str,
            question=base_query,  # Use base_query for report generation
            full_analysis_report=full_analysis_report,
            month_data_raw=month_data,
            xingzhi=xingzhi,
            analysis_data_result=analysis_data_result,
            relevant_palaces_for_report=relevant_palaces_for_query  # Pass relevant palaces to filter report
        )
        logger.info("年度报告内容生成完成。")

        # --- 步骤 8: 构建最终JSON响应 ---
        # 构建最终JSON结构
        final_json_response = {
            "yearKeyword": annual_keywords.get("keyword", "平凡之年"),
            "yearOverview": year_overview,
            "dimensionDetails": dimension_json_list
        }

        # 替换所有"明年"为具体年份
        final_json_response = replace_mingnian_with_year(final_json_response, target_year)
        final_json_response = convert_and_sort_months_in_dimensions(final_json_response)

        # 返回JSON响应（非流式）
        from fastapi.responses import JSONResponse
        return JSONResponse(content=final_json_response, media_type="application/json; charset=utf-8")

    except ValueError as e:
        logger.error(f"ValueError: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"输入格式错误: {str(e)}")
    except HTTPException as e:
        logger.error(f"HTTPException: {e}", exc_info=True)
        raise
    except aiohttp.ClientError as e:
        logger.error(f"aiohttp.ClientError: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"外部API请求失败: {str(e)}")
    except Exception as e:
        logger.error(f"未知异常: {e}", exc_info=True)
        logger.error(f"详细追溯: \n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")
    finally:
        # --- 任务: 更新API使用统计 ---
        try:
            await db_manager.upsert_api_usage_stats(session_id, app_id)
            logger.info(f"用量统计已更新: session_id={session_id}, app_id={app_id}")
        except Exception as e:
            logger.error(f"更新用量统计时发生错误: {e}", exc_info=True)


if __name__ == "__main__":
    print("\n--- 启动说明 ---")
    print("这是一个生成年度紫微斗数报告的FastAPI应用，已优化为固定输入格式和意图。")
    print("\n--- 输入格式要求 ---")
    print("请严格遵循以下格式提交查询：'YYYY年MM月DD日 HH:MM:SS 性别，[基础报告问题][；追问问题]'")
    print("基础报告问题示例：'明年运势如何' 或 '明年主要运势如何' 或 '明年财运如何' 等")
    print("追问问题为可选，例如: '2001年6月8日 14:00:00 男，明年财运如何；我何时能发财？'")
    print("如果没有追问，则不带分号。")
    print("\n--- 环境变量要求 ---")
    print("\n--- 运行命令 (开发模式) ---")
    print("uvicorn api_main:app --host 0.0.0.0 --port 8009 --reload")

    uvicorn.run("api_main:app", host="0.0.0.0", port=8009, reload=True)
