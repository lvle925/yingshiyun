# services/ziwei_analyzer.py
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from database import db_manager # 假设 db_manager 在项目根目录或 PYTHONPATH 中
import copy
import json
# 导入您本地的辅助函数
from tiangan_function import parse_bazi_components, replace_key_names, key_mapping, zhuxing1, ziweids2, describe_ziwei_chart
from liushifunction import key_translation_map, value_translation_map, translate_json, sihua, zhuxing, ziweids_conzayao
from ziwei_ai_function import transform_horoscope_scope_data, transform_palace_data
from config import ALL_PALACES, TRADITIONAL_HOUR_TO_TIME_INDEX, DIZHI_DUIGONG_MAP, MISSING_BIRTH_INFO_MESSAGE
from utils import parse_chart_description_block,calculate_score_by_rules, parse_palace_data

from ziwei_ai_function import transform_horoscope_scope_data, ALL_EARTHLY_BRANCHES, FIXED_PALACE_ORDER_FOR_SCOPES, \
    HEAVENLY_STEM_MUTAGEN_MAP, get_ordered_palace_branches, get_mutagen_for_stem, transform_palace_data, \
    _parse_lenient_json, \
    _extract_first_json_object, _close_open_brackets, simple_clean_birth_info, simple_clean_query_intent, \
    validate_branch_in_prompt, \
    is_valid_datetime_string, validate_ziwei_payload, validate_birth_info_logic, get_lunar_month_range_string, \
    YANG_GAN, YIN_GAN, calculate_all_decadal_periods, save_string_to_file, VALID_XINGXI_DIZHI_COMBOS


logger = logging.getLogger(__name__)

async def generate_ziwei_analysis(birth_info: Dict[str, Any], response_data: Dict[str, Any]) -> Dict[
    str, Any]:  # Added response_data
    """
    通过调用外部API和处理本地CSV数据来生成紫微斗数分析。
    根据小时和分钟精确判断时辰。
    返回结构化的宫位数据字典。
    """
    # 检查是否所有必需的出生信息都已提供
    print(f"【generate_ziwei_analysis】开始检查出生信息完整性: {birth_info}")
    if not (birth_info.get("year") and birth_info.get("month") and birth_info.get("day") and
            (birth_info.get("hour") is not None or birth_info.get("traditional_hour_branch")) and
            birth_info.get("gender") in ["男", "女"]):
        print(
            f"【generate_ziwei_analysis】出生信息不完整或无效。返回错误。详情: 年={birth_info.get('year')}, 月={birth_info.get('month')}, 日={birth_info.get('day')}, 小时={birth_info.get('hour')}, 传统时辰={birth_info.get('traditional_hour_branch')}, 性别={birth_info.get('gender')}")
        return {"error": "gza" + MISSING_BIRTH_INFO_MESSAGE}

    # 将日期格式化为API所需的YYYY-MM-DD
    date_str = f"{birth_info['year']}-{birth_info['month']:02d}-{birth_info['day']:02d}"

    hour = birth_info.get('hour')
    minute = birth_info.get('minute', 0)
    traditional_hour_branch = birth_info.get('traditional_hour_branch')

    time_index = None

    # **修复：优先处理传统时辰，直接映射到time_index**
    if traditional_hour_branch and traditional_hour_branch in TRADITIONAL_HOUR_TO_TIME_INDEX:
        time_index = TRADITIONAL_HOUR_TO_TIME_INDEX[traditional_hour_branch]
        print(f"【generate_ziwei_analysis】通过传统时辰映射获取时辰索引: {traditional_hour_branch} -> {time_index}")
    elif hour is not None:  # 如果没有传统时辰，才通过24小时制的小时和分钟计算
        # 将小时和分钟转换为总分钟数，方便计算
        total_minutes = hour * 60 + minute
        print(f"【generate_ziwei_analysis】通过24小时制小时计算总分钟数: {total_minutes}")

        # 根据总分钟数判断时辰索引
        if 0 <= total_minutes <= 60:  # 00:00 - 00:59
            time_index = 0  # 早子时
        elif 60 < total_minutes <= 180:  # 01:00 - 02:59
            time_index = 1  # 丑时
        elif 180 < total_minutes <= 300:  # 03:00 - 04:59
            time_index = 2  # 寅时
        elif 300 < total_minutes <= 420:  # 05:00 - 06:59
            time_index = 3  # 卯时
        elif 420 < total_minutes <= 540:  # 07:00 - 08:59
            time_index = 4  # 辰时
        elif 540 < total_minutes <= 660:  # 09:00 - 10:59
            time_index = 5  # 巳时
        elif 660 < total_minutes <= 780:  # 11:00 - 12:59
            time_index = 6  # 午时
        elif 780 < total_minutes <= 900:  # 13:00 - 14:59
            time_index = 7  # 未时
        elif 900 < total_minutes <= 1020:  # 15:00 - 16:59
            time_index = 8  # 申时
        elif 1020 < total_minutes <= 1140:  # 17:00 - 18:59
            time_index = 9  # 酉时
        elif 1140 < total_minutes <= 1260:  # 19:00 - 20:59
            time_index = 10  # 戌时
        elif 1260 < total_minutes <= 1380:  # 21:00 - 22:59
            time_index = 11  # 亥时
        elif 1380 < total_minutes <= 1440:  # 23:00 - 23:59
            time_index = 12  # 晚子时
        print(f"【generate_ziwei_analysis】通过24小时制小时计算时辰索引: {time_index}")

    # 如果 time_index 仍然是 None，说明无法解析时辰信息
    if time_index is None:
        print(f"【generate_ziwei_analysis】未能解析时辰信息。")
        return {
            "error": "抱歉，无法解析您提供的时辰信息，请确保时辰在0-23点59分范围内，或提供正确的传统时辰（如子时、丑时等）。"}

    print(
        f"【generate_ziwei_analysis】调用外部API参数: 日期={date_str}, 时辰索引={time_index}, 性别={birth_info['gender']}")

    # NOTE: The 'response_data' parameter is now directly passed to this function.
    # The API call itself should happen BEFORE calling generate_ziwei_analysis/generate_horoscope_analysis.
    # This function now focuses on processing the *already fetched* response_data.

    # --- 关键数据结构校验 ---
    if not (response_data and response_data.get("data") and response_data["data"].get("astrolabe")):
        print("【generate_ziwei_analysis】API响应数据结构不符合预期：缺少'data'或'astrolabe'。",
              response_data)  # Added response_data to print
        return {"error": "抱歉，API返回了意外的数据结构，无法继续分析。"}

    natal_astrolabe = response_data["data"]["astrolabe"]

    # 验证 palaces 列表是否存在且包含 12 个元素
    if not (natal_astrolabe.get("palaces") and isinstance(natal_astrolabe["palaces"], list) and len(
            natal_astrolabe["palaces"]) == 12):
        print(
            f"【generate_ziwei_analysis】API响应中的 'palaces' 数据无效或数量不足12个。实际类型: {type(natal_astrolabe.get('palaces'))}, 长度: {len(natal_astrolabe.get('palaces', []))}")
        return {"error": "抱歉，API返回的命盘宫位数据不完整或格式不正确。"}

    # 获取中文日期数据
    data_ch = natal_astrolabe["rawDates"]["chineseDate"]
    parsed_data = parse_bazi_components(data_ch)  # 调用 tiangan_function.py 中的函数
    print(f"【generate_ziwei_analysis】解析的八字组件: {parsed_data}")

    # 映射键名
    original_data = response_data
    converted_data = replace_key_names(original_data, key_mapping)  # 调用 tiangan_function.py 中的函数

    ziwei_data2 = ziweids2(converted_data)  # 调用 tiangan_function.py 中的函数

    translated_data = translate_json(original_data, key_translation_map,
                                     value_translation_map)  # 调用 liangan_function.py 中的函数

    ziwei_data_原局 = ziweids_conzayao(translated_data['数据'], "命盘", '命主星', '身主星', '五行局', '宫位',
                                       '主星', '地支', '化曜', '辅星煞星', '杂曜', '名称',
                                       zhuxing)  # 调用 liangan_function.py 中的函数
    cleaned_chart_array = []
    for item in ziwei_data_原局:
        processed_item = ["", "", "", "", ""]  # 初始化为5个空字符串

        # 确保有足够元素来处理前4个核心信息
        if len(item) > 0:
            processed_item[0] = item[0]  # 主星
        if len(item) > 1:
            processed_item[1] = item[1]  # 地支
        if len(item) > 2:
            processed_item[2] = item[2]  # 宫位名称
        if len(item) > 3:
            processed_item[3] = item[3]  # 化忌/化权等信息

        # 合并辅星和杂曜 (从第5个元素开始的所有内容)
        all_other_stars_parts = []
        if len(item) > 4:
            for i in range(4, len(item)):
                if isinstance(item[i], str) and item[i].strip():
                    # 清理并加入所有星曜
                    all_other_stars_parts.append(item[i].replace("，", ",").strip())

        # 确保每个部分都是有效星曜列表，然后合并，并清理多余逗号
        processed_item[4] = ",".join(filter(None, all_other_stars_parts)).replace(',,', ',').strip(',')

        cleaned_chart_array.append(processed_item)
    chart_description = describe_ziwei_chart(cleaned_chart_array)  # 调用 tiangan_function.py 中的函数

    palace_key_info_summaries = {}  # 用于存储每个宫位的关键信息汇总

    # 遍历 ziwei_data2 并读取 CSV 文件来填充 palace_key_info_summaries

    queries_to_run = []

    niangan = parsed_data.get('year_stem')
    nianzhi = parsed_data.get('year_branch')
    yuezhi = parsed_data.get('month_branch')
    shichen = parsed_data.get('hour_branch')

    if not all([niangan, nianzhi, yuezhi, shichen]):
        logger.warning("本命盘分析：八字信息不完整，跳过数据库查询。")
    else:
        for i in ziwei_data2:
            if len(i) < 3: continue

            x = re.sub(r'[和,，]\s*', '_', i[0])
            y = i[2].replace("宫宫", "宫").replace("官禄宫", "事业宫").replace("仆役宫", "交友宫")
            original_dizhi = i[1]

            # 注意：传递给数据库查询的 xingxi 不应包含"_"，而是原始的、带中文逗号的
            query_params = {
                "xingxi": i[0].replace("_", "，"),  # 使用原始星系名
                "dizhi": original_dizhi, "gongwei": y,
                "niangan": niangan, "nianzhi": nianzhi, "yuezhi": yuezhi, "shichen": shichen
            }
            print(query_params)
            queries_to_run.append(query_params)

    # --- 步骤 3: 一次性调用批量查询函数 ---
    if queries_to_run:
        logger.info(f"本命盘分析：准备批量执行 {len(queries_to_run)} 个查询条件...")
        batch_results = await db_manager.query_natal_chart_info_batch(queries_to_run)
        logger.info("本命盘分析：批量查询执行完毕。")
    else:
        batch_results = {}

    # --- 步骤 4: 【核心改造】处理批量查询的结果 ---
    # 这个处理逻辑现在变得非常简单
    for query in queries_to_run:
        # 直接用本宫的参数构建ID
        bengong_id = f"{query['xingxi']}|{query['dizhi']}|{query['gongwei']}"

        # 从 batch_results 中获取结果，它已经是处理好本宫对宫关系之后的结果了
        summary_texts = batch_results.get(bengong_id)

        if summary_texts:
            gongwei = query['gongwei']
            if gongwei not in palace_key_info_summaries:
                palace_key_info_summaries[gongwei] = {}
            if "原局盘" not in palace_key_info_summaries[gongwei]:
                palace_key_info_summaries[gongwei]["原局盘"] = []
            palace_key_info_summaries[gongwei]["原局盘"].extend(summary_texts)

    # --- 步骤 5: 最终组装 structured_palace_details_for_llm (不变) ---
    parsed_palace_descriptions_for_birth_chart = parse_chart_description_block(chart_description)
    structured_palace_details_for_llm = {palace: {} for palace in ALL_PALACES}

    for palace_name in ALL_PALACES:
        description = parsed_palace_descriptions_for_birth_chart.get(palace_name, "")
        summaries = palace_key_info_summaries.get(palace_name, {}).get("原局盘", [])

        if "原局盘" not in structured_palace_details_for_llm[palace_name]:
            structured_palace_details_for_llm[palace_name]["原局盘"] = {}

        structured_palace_details_for_llm[palace_name]["原局盘"]["宫位信息"] = description
        structured_palace_details_for_llm[palace_name]["原局盘"]["关键信息汇总"] = "; ".join(
            summaries[:10]) if summaries else ""

    return structured_palace_details_for_llm

async def generate_horoscope_analysis(birth_info: Dict[str, Any], horoscope_date_str: Optional[str],
                                      analysis_level: str,
                                      response_data: Dict[str, Any], query_intent_data, prompt_input) -> (
        Dict[str, Any], str):
    """
    通过调用外部API生成指定运势日期的紫微斗数分析。
    根据 LLM 确定的 analysis_level 来选择要包含的盘和 Prompt 键。
    返回结构化的宫位数据字典和用于LLM的Prompt键。
    """
    # 检查是否所有必需的出生信息都已提供
    if not (birth_info.get("year") and birth_info.get("month") and birth_info.get("day") and
            (birth_info.get("hour") is not None or birth_info.get("traditional_hour_branch")) and
            birth_info.get("gender") in ["男", "女"]):
        print(
            f"【generate_horoscope_analysis】出生信息不完整或无效。返回错误。详情: 年={birth_info.get('year')}, 月={birth_info.get('month')}, 日={birth_info.get('day')}, 小时={birth_info.get('hour')}, 传统时辰={birth_info.get('traditional_hour_branch')}, 性别={birth_info.get('gender')}")
        return {"error": "gha" + MISSING_BIRTH_INFO_MESSAGE}, "missing_birth_info"  # 返回错误信息和对应的Prompt键

    """
    # 特殊处理：如果用户问的是“下一个大限”且没有提供具体年份
    specific_decadal_query_message = "请问您想查询哪一个大限的运势？请提供具体的年份，例如：'2030年大限运势'。"
    if analysis_level == "decadal" and horoscope_date_str is None:
        print(
            f"【generate_horoscope_analysis】请求大限分析但未提供具体年份。返回提示消息: {specific_decadal_query_message}")
        return {"error": specific_decadal_query_message}, "general_question"
    """

    birth_date_str = f"{birth_info['year']}-{birth_info['month']:02d}-{birth_info['day']:02d}"

    hour = birth_info.get('hour')
    minute = birth_info.get('minute', 0)
    traditional_hour_branch = birth_info.get('traditional_hour_branch')
    print("traditional_hour_branch:", traditional_hour_branch)

    time_index = None

    # **修复：优先处理传统时辰，直接映射到time_index**
    if traditional_hour_branch and traditional_hour_branch in TRADITIONAL_HOUR_TO_TIME_INDEX:
        time_index = TRADITIONAL_HOUR_TO_TIME_INDEX[traditional_hour_branch]
        print(f"【generate_horoscope_analysis】通过传统时辰映射获取时辰索引: {traditional_hour_branch} -> {time_index}")
    elif hour is not None:  # 如果没有传统时辰，才通过24小时制的小时和分钟计算
        # 将小时和分钟转换为总分钟数，方便计算
        total_minutes = hour * 60 + minute
        print(f"【generate_horoscope_analysis】通过24小时制小时计算总分钟数: {total_minutes}")

        # 根据总分钟数判断时辰索引
        if 0 <= total_minutes <= 60:  # 00:00 - 00:59
            time_index = 0  # 早子时
        elif 60 < total_minutes <= 180:  # 01:00 - 02:59
            time_index = 1  # 丑时
        elif 180 < total_minutes <= 300:  # 03:00 - 04:59
            time_index = 2  # 寅时
        elif 300 < total_minutes <= 420:  # 05:00 - 06:59
            time_index = 3  # 卯时
        elif 420 < total_minutes <= 540:  # 07:00 - 08:59
            time_index = 4  # 辰时
        elif 540 < total_minutes <= 660:  # 09:00 - 10:59
            time_index = 5  # 巳时
        elif 660 < total_minutes <= 780:  # 11:00 - 12:59
            time_index = 6  # 午时
        elif 780 < total_minutes <= 900:  # 13:00 - 14:59
            time_index = 7  # 未时
        elif 900 < total_minutes <= 1020:  # 15:00 - 16:59
            time_index = 8  # 申时
        elif 1020 < total_minutes <= 1140:  # 17:00 - 18:59
            time_index = 9  # 酉时
        elif 1140 < total_minutes <= 1260:  # 19:00 - 20:59
            time_index = 10  # 戌时
        elif 1260 < total_minutes <= 1380:  # 19:00 - 20:59
            time_index = 11  # 亥时 (注意，这里原代码有一个小的逻辑错误，21-22点59分是亥时，23-0点59分是晚子时)
        elif 1380 < total_minutes <= 1440:  # 23:00 - 23:59
            time_index = 12  # 晚子时
        print(f"【generate_horoscope_analysis】通过24小时制小时计算时辰索引: {time_index}")

    if time_index is None:
        print(f"【generate_horoscope_analysis】未能解析时辰信息。")
        return {
            "error": "抱歉，无法解析您的出生时辰信息，请确保时辰在0-23点59分范围内，或提供正确的传统时辰（如子时、丑时等）。"}, "general_question"

    print(
        f"【generate_horoscope_analysis】调用外部API参数: 日期={birth_date_str}, 运势日期={horoscope_date_str}, 时辰索引={time_index}, 性别={birth_info['gender']}")

    # NOTE: The 'response_data' parameter is now directly passed to this function.
    # The API call itself should happen BEFORE calling generate_ziwei_analysis/generate_horoscope_analysis.
    # This function now focuses on processing the *already fetched* response_data.
    print("response_data", response_data.keys())
    # --- 关键数据结构校验 ---
    # Always check for astrolabe data (natal chart)
    if not (response_data and response_data.get("data") and response_data["data"].get("astrolabe") and
            response_data["data"]["astrolabe"].get("palaces") and isinstance(
                response_data["data"]["astrolabe"]["palaces"], list) and len(
                response_data["data"]["astrolabe"]["palaces"]) == 12):
        print("【generate_horoscope_analysis】API响应数据结构不符合预期：缺少'data'或'astrolabe'。", response_data)
        return {"error": "抱歉，API返回了意外的数据结构，无法继续分析。"}, "general_question"

    # Only check for horoscope data if it's not a birth_chart analysis or unpredictable future
    if analysis_level not in ["birth_chart", "unpredictable_future"] and not (
            response_data["data"].get("horoscope") and isinstance(response_data["data"]["horoscope"], dict)):
        print("【generate_horoscope_analysis】运势API响应数据结构不符合预期：缺少关键运势数据或格式不正确。")
        return {
            "error": f"{analysis_level},{prompt_input}，{query_intent_data},抱歉，运势API返回的运势数据不完整或格式不正确，无法继续分析。"}, "general_question"

    # 获取基础命盘的宫位数据，用于后续转换
    astrolabe_palaces = response_data['data']['astrolabe']['palaces']

    # **新增：如果为大限分析，提取大运干支信息**
    decadal_stem_branch_str = ""
    if analysis_level == "decadal":
        decadal_horoscope_data = response_data['data']['horoscope'].get('decadal', {})
        if decadal_horoscope_data and "heavenlyStem" in decadal_horoscope_data and "earthlyBranch" in decadal_horoscope_data:
            decadal_stem_branch_str = f"{decadal_horoscope_data['heavenlyStem']}{decadal_horoscope_data['earthlyBranch']}大运"

    # 执行转换，将 API 响应数据转换为统一的格式
    transformed_horoscope_scopes = {
        "原局盘": transform_palace_data(response_data['data']['astrolabe']),  # 原始命盘宫位
        # Conditional addition of horoscope data, only if available and expected for the analysis level
        "大限盘": transform_horoscope_scope_data(
            response_data['data']['horoscope'].get('decadal', {}) if analysis_level != "birth_chart" else {},
            astrolabe_palaces),
        "流年盘": transform_horoscope_scope_data(
            response_data['data']['horoscope'].get('yearly', {}) if analysis_level != "birth_chart" else {},
            astrolabe_palaces),
        "流月盘": transform_horoscope_scope_data(
            response_data['data']['horoscope'].get('monthly', {}) if analysis_level != "birth_chart" else {},
            astrolabe_palaces),
        "流日盘": transform_horoscope_scope_data(
            response_data['data']['horoscope'].get('daily', {}) if analysis_level != "birth_chart" else {},
            astrolabe_palaces),
        "流时盘": transform_horoscope_scope_data(
            response_data['data']['horoscope'].get('hourly', {}) if analysis_level != "birth_chart" else {},
            astrolabe_palaces),
    }
    # 根据 LLM 确定的 analysis_level 确定要处理的运势范围和 Prompt 键
    relevant_chart_keys_in_order = []
    scope_display_name = ""
    selected_prompt_key_for_horoscope = "general_question"  # 默认值

    # 根据用户要求修改这里的逻辑，限制 relevant_chart_keys_in_order
    if analysis_level == "hourly":
        # 流时分析，包含流时、流日，以提供完整上下文
        relevant_chart_keys_in_order = ["流时盘", "流日盘"]
        scope_display_name = "流时和流日"
        selected_prompt_key_for_horoscope = "hourly"
    elif analysis_level == "daily":
        # 流日分析，包含流日、流月
        relevant_chart_keys_in_order = ["流日盘", "流月盘"]
        scope_display_name = "流日和流月"
        selected_prompt_key_for_horoscope = "daily"
    elif analysis_level == "monthly":
        # 流月分析，包含流月、流年、大限、原局盘
        relevant_chart_keys_in_order = ["流月盘", "流年盘"]
        scope_display_name = "流月和流年"
        selected_prompt_key_for_horoscope = "monthly"
    elif analysis_level == "yearly":
        # 流年分析，包含流年、大限、原局盘
        relevant_chart_keys_in_order = ["流年盘", "大限盘"]
        scope_display_name = "流年和大限"
        selected_prompt_key_for_horoscope = "yearly"
    elif analysis_level == "decadal":
        # 大限分析，包含大限、原局盘
        relevant_chart_keys_in_order = ["大限盘", "原局盘"]
        scope_display_name = "大限和原局"
        selected_prompt_key_for_horoscope = "decadal"
    elif analysis_level == "birth_chart":
        # 命盘分析，只包含原局盘
        relevant_chart_keys_in_order = ["原局盘"]
        scope_display_name = "原局"
        selected_prompt_key_for_horoscope = "birth_chart_analysis"
    elif analysis_level == "unpredictable_future":  # Handle unpredictable future explicitly
        relevant_chart_keys_in_order = []
        scope_display_name = "无法预测的未来"
        selected_prompt_key_for_horoscope = "unpredictable_future"
    else:
        # Fallback：当前analysis_level为general_question或其他未显式处理的情况。
        # 这里不能什么盘都不看，否则structured_palace_details_for_llm中的每个宫位都是空dict，
        # 会导致后续像命宫信息这样的内容完全缺失。
        #
        # 历史行为：即使是“泛问”(general_question)，也至少会用【原局盘】给出一份基础命宫/宫位描述。
        # 因此这里退回到只看原局盘，相当于做一层“轻量级本命盘分析”，保证每个用户都有命宫信息。
        relevant_chart_keys_in_order = ["原局盘"]
        scope_display_name = "原局"
        selected_prompt_key_for_horoscope = "birth_chart_analysis"

    # New structure to hold all palace details, organized by palace and then by chart type
    # { "命宫": { "流年盘": { "宫位信息": "...", "关键信息汇总": "..." }, "大限盘": {...} }, ... }
    structured_palace_details_for_llm = {palace: {} for palace in ALL_PALACES}

    queries_to_run = []
    metadata_for_queries = []  # 用一个列表来保持顺序

    local_transformed_scopes = copy.deepcopy(transformed_horoscope_scopes)

    tasks = []
    task_metadata = []

    queries_to_run = []

    # 【新】创建一个反向映射，用于通过参数找到 chart_key
    query_to_chart_key_map = {}

    task_metadata_map = {}

    collected_relevant_charts = []

    for chart_key in relevant_chart_keys_in_order:
        if chart_key in transformed_horoscope_scopes and transformed_horoscope_scopes[chart_key]:
            chart_data_list = transformed_horoscope_scopes[chart_key]

            collected_relevant_charts.append(chart_data_list)

            full_chart_description_block = describe_ziwei_chart(chart_data_list)
            parsed_palace_descriptions = parse_chart_description_block(full_chart_description_block)
            for palace_name, description in parsed_palace_descriptions.items():

                if palace_name not in structured_palace_details_for_llm:
                    structured_palace_details_for_llm[palace_name] = {}
                if chart_key not in structured_palace_details_for_llm[palace_name]:
                    structured_palace_details_for_llm[palace_name][chart_key] = {}
                structured_palace_details_for_llm[palace_name][chart_key]["宫位信息"] = description
                # print(palace_name, palace_name,structured_palace_details_for_llm)

            # Populate palace_key_info_summaries for this chart_key
            for j in chart_data_list:
                if len(j) < 6:
                    continue
                x = re.sub(r'[和,，]\s*', '_', j[0])
                y_display = j[2]
                y_file = y_display.replace("宫宫", "宫").replace("官禄宫", "事业宫").replace("仆役宫", "交友宫")

                original_dizhi = j[1]
                hua_star_val = j[3].split(',')[-1]
                minor_stars_val = j[4]
                adjective_stars_val = j[5]
                match_stars_val = ",".join(filter(None, [minor_stars_val]))
                match_stars_val = match_stars_val.replace(" ", "")

                file_read_successful = False
                data = pd.DataFrame()

                # print("xingxi", x.replace("_", "，"), "dizhi", original_dizhi, "gongwei", y_file,
                #       "pipeixingyao", match_stars_val, "huaxing", hua_star_val)

                query_params = {
                    "xingxi": x.replace("_", "，"), "dizhi": original_dizhi, "gongwei": y_file,
                    "pipeixingyao": match_stars_val, "huaxing": hua_star_val
                }
                queries_to_run.append(query_params)

                metadata_for_queries.append({'gongwei': y_file, 'chart_key': chart_key})

                query_tuple = (x, original_dizhi, y_file, match_stars_val, hua_star_val)
                query_to_chart_key_map[query_tuple] = chart_key

    # 2. 一次性调用批量查询函数
    if queries_to_run:
        logger.info(f"运势分析：准备批量执行 {len(queries_to_run)} 个查询条件...")
        # 【核心】: 调用新的 _batch 函数
        batch_results = await db_manager.query_horoscope_info_batch(queries_to_run)
        # print("batch_results","**********************************",batch_results)

        logger.info("运势分析：批量查询执行完毕。")
    else:
        batch_results = {}

    save_string_to_file("batch_results.txt", str(batch_results))

    for result_id, summary_texts in batch_results.items():
        if not summary_texts:
            continue

        # 解析 result_id 来获取各个部分
        parts = result_id.split('|')
        if len(parts) != 5:
            continue

        xingxi, dizhi, gongwei, pipeixingyao, huaxing = parts
        # print(xingxi, dizhi, gongwei, pipeixingyao, huaxing)

        # 现在，我们需要找到这个结果属于哪个 chart_key
        # 我们需要遍历最初的 chart_data_list 来找到匹配项
        found = False
        for chart_key in relevant_chart_keys_in_order:
            if chart_key in transformed_horoscope_scopes and transformed_horoscope_scopes[chart_key]:
                chart_data_list = transformed_horoscope_scopes[chart_key]
                for j in chart_data_list:
                    if len(j) < 6:
                        continue
                    x = re.sub(r'[和,，]\s*', '_', j[0])
                    x = x.replace("_", "，")
                    y_display = j[2]
                    y_file = y_display.replace("宫宫", "宫").replace("官禄宫", "事业宫").replace("仆役宫", "交友宫")

                    original_dizhi = j[1]
                    hua_star_val = j[3].split(',')[-1]
                    minor_stars_val = j[4]
                    adjective_stars_val = j[5]
                    match_stars_val = ",".join(filter(None, [minor_stars_val]))
                    match_stars_val = match_stars_val.replace(" ", "")

                    # 检查是否是这个组合（本宫或对宫）产生了当前的结果
                    current_dizhi = j[1]
                    duigong_dizhi = DIZHI_DUIGONG_MAP.get(current_dizhi)

                    # print("="*60)
                    # print(x,current_dizhi,duigong_dizhi,y_file,match_stars_val,hua_star_val)

                    if (dizhi == current_dizhi or dizhi == duigong_dizhi) and \
                            xingxi == x and gongwei == y_file and \
                            pipeixingyao == match_stars_val and huaxing == hua_star_val:

                        # 找到了！这个结果属于当前的 palace_name 和 chart_key
                        palace_name = y_file  # 用 y_file 作为宫位名

                        if palace_name not in structured_palace_details_for_llm: structured_palace_details_for_llm[
                            palace_name] = {}
                        if chart_key not in structured_palace_details_for_llm[palace_name]:
                            structured_palace_details_for_llm[palace_name][chart_key] = {}
                        if "关键信息汇总_raw_list" not in structured_palace_details_for_llm[palace_name][chart_key]:
                            structured_palace_details_for_llm[palace_name][chart_key]["关键信息汇总_raw_list"] = []

                        structured_palace_details_for_llm[palace_name][chart_key]["关键信息汇总_raw_list"].extend(
                            summary_texts)

                        found = True
                        break  # 内层循环中断
            if found:
                break  # 外层循环也中断

    # print("structured_palace_details_for_llm",structured_palace_details_for_llm)

    # Convert lists of summaries into single, deduplicated strings for LLM consumption
    for palace_name, chart_data in structured_palace_details_for_llm.items():
        for chart_key, details in chart_data.items():
            if "关键信息汇总_raw_list" in details and isinstance(details["关键信息汇总_raw_list"], list):
                raw_summaries = details["关键信息汇总_raw_list"]
                unique_points = set()
                for s in raw_summaries:
                    # Split by common delimiters (semicolon, Chinese period, newline)
                    points = re.split(r'[;；。\n]', s)
                    for p in points:
                        p_stripped = p.strip()
                        if p_stripped:
                            unique_points.add(p_stripped)

                # Join the unique points.
                details["关键信息汇总"] = "; ".join(sorted(list(unique_points)))  # Sort for consistent output
                # Remove the raw list after processing
                del details["关键信息汇总_raw_list"]
            else:
                details["关键信息汇总"] = ""  # Ensure it's an empty string if no summaries

    # **新增：将 decadal_stem_branch_str 附加到返回的结构化数据中**
    # 使用一个不会与宫位名称冲突的特殊键
    structured_palace_details_for_llm["_decadal_info"] = decadal_stem_branch_str

    # **修复：评分逻辑只在运势分析时执行（本命盘分析不需要评分）**
    if analysis_level not in ["birth_chart", "unpredictable_future"] and len(collected_relevant_charts) >= 2:
        liu_nian_parsed = parse_palace_data(collected_relevant_charts[0])
        da_xian_parsed = parse_palace_data(collected_relevant_charts[1])

        final_results_list = []
        for palace_info_dict in liu_nian_parsed:
            palace_name = palace_info_dict.get('palace_name', '未知宫位')
            di_zhi = palace_info_dict.get('di_zhi', '未知地支')
            score, source_scores, summary_dict = calculate_score_by_rules(palace_info_dict, liu_nian_parsed, da_xian_parsed)
            palace_result = {
                "palace_name": palace_name,
                "di_zhi": di_zhi,
                "final_score": round(score, 1),
                "combination_source_scores": source_scores,
                "summary": summary_dict
            }
            final_results_list.append(palace_result)
        sorted_results = sorted(final_results_list, key=lambda x: x['final_score'], reverse=True)

        final_results_list = []
        for palace_info_dict in liu_nian_parsed:
            palace_name = palace_info_dict.get('palace_name', '未知宫位')
            di_zhi = palace_info_dict.get('di_zhi', '未知地支')
            score, source_scores, summary_dict = calculate_score_by_rules(
                palace_info_dict,
                liu_nian_parsed,
                da_xian_parsed
            )
            # 将完整的原始结果（包含所有信息）添加到列表中
            final_results_list.append({
                "palace_name": palace_name,
                "di_zhi": di_zhi,
                "final_score": round(score, 1),
                "combination_source_scores": source_scores,
                "summary": summary_dict
            })

        # --- 按您的要求，准备最终的简洁输出 ---
        final_output = {}
        for result in final_results_list:
            final_output[result['palace_name']] = result['final_score']

        # --- 以您指定的字典格式输出 ---
        # print("## 最终得分 (字典格式)")
        print("最终得分",json.dumps(final_output, indent=2, ensure_ascii=False))
        final_score = json.dumps(final_output, indent=2, ensure_ascii=False)

    # print({"palace_details": structured_palace_details_for_llm}, selected_prompt_key_for_horoscope)

    # Return the structured data and the selected prompt key
    return structured_palace_details_for_llm, selected_prompt_key_for_horoscope
