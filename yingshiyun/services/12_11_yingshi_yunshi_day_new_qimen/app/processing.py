import asyncio
import copy
import json
import logging
import re
import requests
import time
from typing import Dict, Any, List
import math
import numpy as np

# 导入配置和辅助函数
from app.config import ASTRO_API_URL
from app.llm_calls import get_llm_daily_advice, get_llm_palace_narrative,_validate_llm_advice_format
from app.utils import scale_score_to_100
from app.monitor import StepMonitor
import pandas as pd 
# 导入您自己的库
from helper_libs import db_manager_yingshi as db_manager
from helper_libs.tiangan_function import describe_ziwei_chart
from app.utils import transform_palace_data, transform_horoscope_scope_data, parse_chart_description_block, parse_palace_data, calculate_score_by_rules1 as calculate_score_by_rules,save_string_to_file 

logger = logging.getLogger(__name__)

ALL_PALACES = ["命宫", "夫妻宫",  "财帛宫", "事业宫", "官禄宫","疾厄宫","迁移宫","交友宫","仆役宫"]
DIZHI_DUIGONG_MAP = {
    "子": "午", "丑": "未", "寅": "申", "卯": "酉", "辰": "戌", "巳": "亥",
    "午": "子", "未": "丑", "申": "寅", "酉": "卯", "戌": "辰", "亥": "巳"
}



async def get_astro_data(dateStr: str, type: str, timeIndex: int, gender: str, horoscopeDate: str) -> dict:
    """
    通过 API 获取星盘数据。

    参数:
        dateStr (str): 出生日期，格式 YYYY-MM-DD
        type (str): 类型
        timeIndex (int): 时辰索引
        gender (str): 性别
        horoscopeDate (str): 预测日期

    返回:
        dict: API 返回的 JSON 数据
    """


    start_time = time.time()
    logger.info(f"开始处理日期: {horoscopeDate}")
    # 从配置文件获取 API URL
    api_url = ASTRO_API_URL

    # 构建请求的 payload
    payload = {
        "dateStr": dateStr,
        "type": type,
        "timeIndex": timeIndex,
        "gender": gender,
        "horoscopeDate": horoscopeDate
    }

    print(payload)

    # 设置请求头，指定发送 JSON 数据
    headers = {
        'Content-Type': 'application/json'
    }

    try:
        # 发送 POST 请求
        with StepMonitor("调用紫薇API", extra_data={"api_url": api_url, "horoscope_date": horoscopeDate}):
            response = await asyncio.to_thread(
                requests.post, api_url, data=json.dumps(payload), headers=headers, timeout=6000
            )
            response.raise_for_status()
            response_data = response.json()


        analysis_level = "daily"

        astrolabe_palaces = response_data['data']['astrolabe']['palaces']

        # **新增：如果为大限分析，提取大运干支信息**
        decadal_stem_branch_str = ""
        if analysis_level == "decadal":
            decadal_horoscope_data = response_data['data']['horoscope'].get('decadal', {})
            if decadal_horoscope_data and "heavenlyStem" in decadal_horoscope_data and "earthlyBranch" in decadal_horoscope_data:
                decadal_stem_branch_str = f"{decadal_horoscope_data['heavenlyStem']}{decadal_horoscope_data['earthlyBranch']}大运"
                #print(f"【generate_horoscope_analysis】生成大运信息: {decadal_stem_branch_str}")

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
        #print(f"【generate_horoscope_analysis】转换后的运势范围数据: {transformed_horoscope_scopes}")

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
        else:  # Fallback for general_question or missing_birth_info
            relevant_chart_keys_in_order = []
            scope_display_name = "未知分析范围"
            selected_prompt_key_for_horoscope = "general_question"


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
                #print(chart_data_list)

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

                    print("xingxi", x.replace("_", "，"), "dizhi", original_dizhi, "gongwei", y_file,
                        "pipeixingyao", match_stars_val, "huaxing", hua_star_val)

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
            with StepMonitor("数据库批量查询", extra_data={"query_count": len(queries_to_run)}):
                batch_results = await db_manager.query_horoscope_info_batch(queries_to_run)
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

        # 开始调用 LLM
        #structured_palace_details_for_llm = structured_palace_details_for_llm.get('命宫').get("流日盘")
        #print("structured_palace_details_for_llm",structured_palace_details_for_llm)

        # 评分计算步骤
        with StepMonitor("评分计算", extra_data={"palace_count": len(collected_relevant_charts)}):
            liu_nian_parsed = parse_palace_data(collected_relevant_charts[0])
            da_xian_parsed = parse_palace_data(collected_relevant_charts[1])
             

            final_results_list = []
            for palace_info_dict in liu_nian_parsed:
                palace_name = palace_info_dict.get('palace_name', '未知宫位')
                di_zhi = palace_info_dict.get('di_zhi', '未知地支')
                score, source_scores, summary_dict,negative_total = calculate_score_by_rules(
                    palace_info_dict,
                    liu_nian_parsed,
                    da_xian_parsed
                )
                # 将完整的原始结果（包含所有信息）添加到列表中
                final_results_list.append({
                    "palace_name": palace_name,
                    "di_zhi": di_zhi,
                    "final_score": round(score, 1),
                    "negative_score_sum": round(negative_total, 1),  # <-- 新增的负分总和
                    "combination_source_scores": source_scores,
                    "summary": summary_dict
                })  
                # print("summary_dict",summary_dict)

        # --- 按您的要求，准备最终的简洁输出 ---
        final_output = {}
        for result in final_results_list:
            final_output[result['palace_name']] = result['final_score']

        # 定义参与综合评分计算的宫位及其权重
        # 综合评分 =（官禄宫 + 财帛宫 + 夫妻宫 + 迁移宫*0.5 + 疾厄宫*0.5 + 交友宫 *0.5）/6
        palace_weights = {
            "官禄宫": 1.0,
            "财帛宫": 1.0,
            "夫妻宫": 1.0,
            "迁移宫": 0.5,
            "疾厄宫": 0.5,
            "仆役宫": 0.5,
        }
        
        scaled_scores_map = {}
        weighted_total_score = 0
        raw_scores_map = {item['palace_name']: item['final_score'] for item in final_results_list}

        # 计算各宫位的标准化分数并应用权重
        for palace in palace_weights.keys():
            raw_score = raw_scores_map.get(palace, 0)
            scaled_score = scale_score_to_100(raw_score)
            scaled_scores_map[f"{palace}_100"] = scaled_score  # Todo
            weight = palace_weights[palace]
            if palace in ["官禄宫","财帛宫","夫妻宫"]:
                weighted_total_score += scaled_score

        composite_score = int(weighted_total_score / 3)
        # composite_score = int(50 + 49 / (1 + math.exp(-0.15 * (weighted_total_score - 7.5))) + 0.3 * math.sin(weighted_total_score * 0.4))

        print("composite_score", composite_score)

        # --- 安全地构建LLM输入并调用（并行处理） ---
        ming_gong_daily_details = structured_palace_details_for_llm.get('命宫', {}).get("流日盘", "今日请保持平常心。")
        guanlu_gong_details = structured_palace_details_for_llm.get('官禄宫', {}).get("流日盘", "事业方面建议稳步前行。")
        caibo_gong_details = structured_palace_details_for_llm.get('财帛宫', {}).get("流日盘", "财富方面建议谨慎理财。")
        fuqi_gong_details = structured_palace_details_for_llm.get('夫妻宫', {}).get("流日盘", "感情方面建议用心沟通。")

        jie_gong_details = structured_palace_details_for_llm.get('疾厄宫', {}).get("流日盘", "感情方面建议用心沟通。")
        qianyi_gong_details = structured_palace_details_for_llm.get('迁移宫', {}).get("流日盘", "感情方面建议用心沟通。")
        puyi_gong_details = structured_palace_details_for_llm.get('仆役宫', {}).get("流日盘", "感情方面建议用心沟通。")

        # 创建所有 LLM 任务
        llm_tasks = [
            get_llm_daily_advice(str(ming_gong_daily_details), composite_score),
            get_llm_palace_narrative('官禄宫', str(guanlu_gong_details), scaled_scores_map.get("官禄宫_100", 50)),
            get_llm_palace_narrative('财帛宫', str(caibo_gong_details), scaled_scores_map.get("财帛宫_100", 50)),
            get_llm_palace_narrative('夫妻宫', str(fuqi_gong_details), scaled_scores_map.get("夫妻宫_100", 50)),

            get_llm_palace_narrative('疾厄宫', str(jie_gong_details), scaled_scores_map.get("疾厄宫_100", 50)),
            get_llm_palace_narrative('迁移宫', str(qianyi_gong_details), scaled_scores_map.get("迁移宫_100", 50)),
            get_llm_palace_narrative('仆役宫', str(puyi_gong_details), scaled_scores_map.get("仆役宫_100", 50)),
        ]

        # # 并行执行所有 LLM 任务
        with StepMonitor("LLM并行调用", extra_data={"task_count": len(llm_tasks)}):
            logger.info("开始并行执行 LLM 任务...")
            llm_results = await asyncio.gather(*llm_tasks)
            logger.info("LLM 并行处理完成")

        # 解构结果
        llm_advice, llm_advice_guanlu, llm_advice_caibo, llm_advice_fuqi,llm_advice_jie,llm_advice_qianyi,llm_advice_puyi = llm_results

        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"get_astro_data 总耗时: {total_time:.2f}秒")

        # print("llm_advice",llm_advice)

        # --- 返回一个包含所有需要的信息的字典 ---
        return {
            "all_palace_scores": final_results_list,
            "scaled_scores": scaled_scores_map,
            "composite_score": composite_score,
            "llm_advice_综合": llm_advice,
            "llm_advice_事业": llm_advice_guanlu,
            "llm_advice_财富": llm_advice_caibo,
            "llm_advice_感情": llm_advice_fuqi,

            "llm_advice_健康": llm_advice_jie,
            "llm_advice_出行": llm_advice_qianyi,
            "llm_advice_人际": llm_advice_puyi,

        }

    except requests.exceptions.RequestException as e:
        logger.error(f"请求发生错误: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"处理数据时发生意外错误: {e}", exc_info=True)
        return {"error": f"内部错误: {e}"}
