# app/api.py
import asyncio
import json
import logging
from datetime import datetime, timedelta, date, time
from typing import Optional
from fastapi import HTTPException, status, APIRouter
from pydantic import BaseModel, Field, validator
import random

# 导入重构后的模块
from app.config import DB_CONFIG # 用于启动时打印
from app.processing import get_astro_data
from app.utils import convert_time_to_time_index
from helper_libs import db_manager_yingshi as db_manager
from app.llm_calls import get_llm_period_summary
from app.monitor import StepMonitor, log_step
import pandas as pd
import os
import time
import re
import uuid


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


KEY_TRANSLATION_MAP_CAMEL_CASE = {
    # --- 周期性总结部分 ---
    "周期性总结": "periodicSummary",
    "分析周期": "analysisPeriod",
    "周期平均得分": "periodAverageScores",
    "周期平均_综合评分": "averageOverallScore",
    "周期平均_感情": "averageLoveScore",
    "周期平均_事业": "averageCareerScore",
    "周期平均_财富": "averageWealthScore",
    "周期核心运势总结": "periodCoreFortuneSummary",
    "周期事业运势总结": "periodCareerFortuneSummary",
    "周期财富运势总结": "periodWealthFortuneSummary",
    "周期感情运势总结": "periodLoveFortuneSummary",
    "周期综合行动指南": "periodComprehensiveActionGuide",
    
    # --- 每日详情部分 ---
    "每日详情": "dailyDetails",
    "日期": "date",
    "综合评分": "overallScore",
    "感情": "loveScore",
    "事业": "careerScore",
    "财富": "wealthScore",
    "运势判断": "fortuneAssessment",
    "建议": "suggestions",
    "避免": "avoidances",
    "事业运势详解": "careerFortuneDetails",
    "财富运势详解": "wealthFortuneDetails",
    "感情运势详解": "loveFortuneDetails"
}

def normalize_llm_text(text: str) -> str:
    """
    规范化 LLM 返回的文本，在 JSON 序列化之前必须调用。
    1. 把字面的 "\\n"（两个字符：反斜杠+n）转换成真正的换行符 "\n"
    2. 清理末尾的换行符和空白字符
    """
    if not text:
        return text
    if not isinstance(text, str):
        return text
    text = text.replace("\\n", "\n")   # 把"写出来的 \n"变成真正换行
    text = text.rstrip()               # 清理末尾幽灵换行
    return text

def extract_suggestions_from_fortune_detail(fortune_detail: str) -> list:
    """
    从运势详解文本中提取3条建议。
    格式为：建议：\n1. [第一条详细建议]\n2. [第二条详细建议]\n3. [第三条详细建议]
    """
    if not fortune_detail or not isinstance(fortune_detail, str):
        return []
    
    suggestions = []
    # 匹配 "建议：" 或 "建议:" 后面的内容
    pattern = r'建议[：:]\s*\n?\s*1\.\s*([^\n]+)\s*\n?\s*2\.\s*([^\n]+)\s*\n?\s*3\.\s*([^\n]+)'
    match = re.search(pattern, fortune_detail)
    
    if match:
        suggestions = [match.group(1).strip(), match.group(2).strip(), match.group(3).strip()]
    else:
        # 尝试更宽松的匹配模式，匹配数字开头的建议
        pattern2 = r'1\.\s*([^\n]+)\s*\n?\s*2\.\s*([^\n]+)\s*\n?\s*3\.\s*([^\n]+)'
        match2 = re.search(pattern2, fortune_detail)
        if match2:
            suggestions = [match2.group(1).strip(), match2.group(2).strip(), match2.group(3).strip()]
    
    return suggestions


def get_randomized_score(score_value, default_range=(65, 75)):
    """
    根据给定的分数，在对应的分数区间内生成一个随机整数。
    分数范围已调整为40~95分。
    如果分数无效或超出常规范围，则返回默认区间内的随机数。

    Args:
        score_value: 原始分数，可以是 float 或 int。
        default_range: 当分数无效时使用的默认随机范围 (min, max)。

    Returns:
        一个随机整数（40~95之间）。
    """
    try:
        score = float(score_value)
        if 90 <= score <= 100:
            return random.randint(90, 95)
        elif 85 <= score < 90:
            return random.randint(85, 89)
        elif 80 <= score < 85:
            return random.randint(80, 84)
        elif 75 <= score < 80:
            return random.randint(75, 79)
        elif 70 <= score < 75:
            return random.randint(70, 74)
        elif 65 <= score < 70:
            return random.randint(65, 69)
        elif 60 <= score < 65:
            return random.randint(60, 64)
        elif 55 <= score < 60:
            return random.randint(55, 59)
        elif 50 <= score < 55:
            return random.randint(50, 54)
        elif 45 <= score < 50:
            return random.randint(45, 49)
        elif 40 <= score < 45:
            return random.randint(40, 44)
        elif score < 40:
            return random.randint(40, 44)  # 低于40的也返回40-44之间
        else: # 分数超出常规范围（例如大于100）
            return random.randint(90, 95)  # 超过100的返回最高档
    except (ValueError, TypeError):
        # 如果 score_value 不是有效的数字
        return random.randint(default_range[0], default_range[1])


def translate_json_keys(data, translation_map):
    """
    递归地将字典或列表中的中文字典键转换为英文。

    参数:
    data (dict or list): 需要转换的原始数据。
    translation_map (dict): 中文键到英文键的映射字典。

    返回:
    dict or list: 键被翻译成英文后的新数据。
    """
    # 如果数据是列表，则对列表中的每个元素进行递归处理
    if isinstance(data, list):
        return [translate_json_keys(item, translation_map) for item in data]

    # 如果数据是字典，则创建一个新字典，并对键值对进行处理
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            # 1. 翻译键名：如果键在映射表中，则使用英文名；否则，保留原键名以避免错误。
            new_key = translation_map.get(key, key)
            
            # 2. 递归处理值：值本身也可能是字典或列表，需要继续转换。
            new_dict[new_key] = translate_json_keys(value, translation_map)
        return new_dict

    # 如果数据既不是列表也不是字典（如字符串、数字等），直接返回
    return data



router = APIRouter()
db_pool = None


class DailyAnalysisRequest(BaseModel):
    birth_date_str: str = Field(..., description="出生日期", example="1995-05-06")
    birth_time_str: str = Field(..., description="出生时间 (HH:MM:SS)", example="14:30:00")
    gender: str = Field(..., description="性别 (男/女)", example="男")
    start_date: date = Field(..., description="分析开始日期", example="2025-08-11")
    end_date: date = Field(..., description="分析结束日期", example="2025-08-20")

    include_summary: Optional[bool] = Field(
        False, # 默认值为 False
        description="是否在报告中包含周期性总结。设置为 true 将执行额外的LLM分析，响应时间会更长。"
    )


    @validator('gender')
    def gender_must_be_valid(cls, v):
        if v not in ['男', '女']:
            raise ValueError('性别必须是 "男" 或 "女"')
        return v
        
    @validator('birth_time_str')
    def time_format_must_be_valid(cls, v):
        try:
            datetime.strptime(v, '%H:%M:%S')
        except ValueError:
            raise ValueError('出生时间格式必须是 "HH:MM:SS"')
        return v

    @validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('结束日期不能早于开始日期')
        return v


async def init_daily_analysis_service():
    global db_pool
    if db_pool:
        logger.info("数据库连接池已存在，跳过初始化。")
        return db_pool
    logger.info("应用启动，正在初始化数据库连接池...")
    # 假设 init_db_pool 返回连接池对象
    db_pool = await db_manager.init_db_pool()
    if not db_pool:
        logger.error("数据库连接池初始化失败！")
    else:
        logger.info("数据库连接池初始化成功。")

async def shutdown_daily_analysis_service():
    global db_pool
    if db_pool:
        logger.info("应用关闭，正在关闭数据库连接池...")
        # 假设 close_db_pool 需要连接池对象作为参数
        await db_manager.close_db_pool(db_pool)
        logger.info("数据库连接池已关闭。")
        db_pool = None

        
@router.post("/generate_daily_analysis_V2", summary="生成每日运势分析报告")
async def generate_analysis(request: DailyAnalysisRequest):
    """
    根据指定的出生年月日时、性别和时间范围，生成每日的详细运势分析报告。
    使用异步并发处理多天数据。
    """
    # 生成8位uuid作为request_id，贯穿整个HTTP请求的监控日志
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    try:
        # 参数解析步骤
        with StepMonitor("参数解析", request_id=request_id, extra_data={
            "birth_date": request.birth_date_str,
            "start_date": str(request.start_date),
            "end_date": str(request.end_date)
        }):
            birth_date_str = request.birth_date_str
            start_date_dt = request.start_date
            end_date_dt = request.end_date

            birth_time_obj = datetime.strptime(request.birth_time_str, '%H:%M:%S').time()
            time_index = convert_time_to_time_index(birth_time_obj)

            base_payload_params = {
                "type": "solar",
                "timeIndex": time_index,
                "gender": request.gender,
            }

            logger.info(f"请求参数解析成功: timeIndex={time_index}, gender={request.gender}")

        # 创建所有日期的任务列表
        with StepMonitor("创建任务列表", request_id=request_id, extra_data={
            "date_count": (end_date_dt - start_date_dt).days + 1
        }):
            tasks = []
            current_date_dt = start_date_dt
            while current_date_dt <= end_date_dt:
                horoscope_date_str = current_date_dt.strftime("%Y-%m-%d 1:00:00")
                logger.info(f"创建任务: 为 {birth_date_str} 分析 {current_date_dt.strftime('%Y-%m-%d')} 的运势...")
                
                payload = {
                    "dateStr": birth_date_str,
                    **base_payload_params,
                    "horoscopeDate": horoscope_date_str
                }
                
                # 创建异步任务
                task = asyncio.create_task(get_astro_data(**payload))
                tasks.append((current_date_dt, task))
                current_date_dt += timedelta(days=1)

        # 并发执行所有任务
        tasks_execution_start = time.time()
        all_daily_data = []
        with StepMonitor("执行任务", request_id=request_id, extra_data={
            "task_count": len(tasks)
        }):
            for date_dt, task in tasks:
                try:
                    astro_data_response = await task
                
                    if isinstance(astro_data_response, dict) and "error" in astro_data_response:
                        logger.error(f"获取 {date_dt.strftime('%Y-%m-%d')} 数据失败: {astro_data_response['error']}")
                        error_row = {"日期": date_dt.strftime('%Y-%m-%d'), "综合评分": f"处理错误: {astro_data_response['error']}"}
                        all_daily_data.append(error_row)
                    elif isinstance(astro_data_response, dict) and 'composite_score' in astro_data_response:
                        daily_row = {"日期": date_dt.strftime('%Y-%m-%d')}
                        
                        daily_row.update(astro_data_response.get("scaled_scores", {}))
                        daily_row["综合评分"] = astro_data_response.get("composite_score", "N/A")
                        
                        # 将 llm_advice_综合 的内容也直接更新到 daily_row
                        # 这样 '今日关键词'、'运势判断'、'运势判断详情'、'建议'、'避免' 都会被包含
                        llm_advice_general = astro_data_response.get("llm_advice_综合", {})
                        daily_row.update(llm_advice_general)

                        print("daily_row",daily_row)

                        # 处理各项运势数据
                        for category in ['事业', '财富', '感情','健康','出行','人际']:
                            llm_advice = astro_data_response.get(f"llm_advice_{category}", {})
                            daily_row[f"{category}标题"] = llm_advice.get(f"{category}标题")
                            daily_row[f"{category}运势详解"] = llm_advice.get("运势详解")
                            daily_row[f"{category}分享话语"] = llm_advice.get("分享话语")
                            daily_row[f"{category}引导问题"] = llm_advice.get("引导问题")
                            daily_row[f"{category}引导问题"] = llm_advice.get("引导问题")

                        all_daily_data.append(daily_row)
                    else:
                        logger.warning(f"在 {date_dt.strftime('%Y-%m-%d')} 的响应中返回了未知的数据格式。")
                        error_row = {"日期": date_dt.strftime('%Y-%m-%d'), "综合评分": "格式未知"}
                        all_daily_data.append(error_row)
                except Exception as e:
                    logger.error(f"处理 {date_dt.strftime('%Y-%m-%d')} 数据时出错: {e}")
                    error_row = {"日期": date_dt.strftime('%Y-%m-%d'), "综合评分": f"处理异常: {str(e)}"}
                    all_daily_data.append(error_row)

        tasks_execution_end = time.time()
        logger.info(f"执行所有任务耗时: {tasks_execution_end - tasks_execution_start:.2f}秒")

        if not all_daily_data:
            return []

        # --- 【核心修改】: 将所有中文键转换为英文驼峰式命名 ---
        with StepMonitor("数据转换", request_id=request_id, extra_data={
            "data_count": len(all_daily_data)
        }):
            final_camel_case_data = []
            default_suggestions = ["保持耐心", "寻求支持"]
            default_avoidances = ["过度忧虑", "冲动行事"]
            default_today_keyword = "平顺" # 默认关键词
            default_today_detail = "运势呈现积蓄力量的态势。"
            default_share_keyword = "眼光独到，决策精准"
            default_share_detail = "今天的我，看什么项目都像潜力股。"
            for original_dict in all_daily_data:
                camel_dict = {}
                individual_scores = []
                print("original_dict",original_dict)
                # --- 步骤 1: 基础字段转换 (正确) ---
                camel_dict['date'] = original_dict.get("日期")
                
                # --- 【新增】: 提取今日关键词并转换为 camelCase ---
                # todayKeyword 使用 ShareKeyword 的值
                share_keyword_value = original_dict.get("分享关键词", default_share_keyword)
                if isinstance(share_keyword_value, list):
                    camel_dict['todayKeyword'] = "，".join(str(item) for item in share_keyword_value)
                else:
                    camel_dict['todayKeyword'] = str(share_keyword_value) if share_keyword_value else default_share_keyword
                camel_dict['todayDetail'] = normalize_llm_text(original_dict.get("运势概况", default_today_detail))
                camel_dict['shareKeyword'] = original_dict.get("分享关键词", default_share_keyword)
                camel_dict['shareDetail'] = normalize_llm_text(original_dict.get("分享话语", default_share_detail))


                # --- 步骤 2: 所有分数字段的检查、转换和随机数填充 (保留您的新功能) ---
                score_keys = {
                    "夫妻宫_100": "relationshipScore",
                    "官禄宫_100": "careerScore",
                    "事业宫_100": "careerScore", 
                    "财帛宫_100": "wealthScore",

                    "疾厄宫_100": "healthyScore",
                    "迁移宫_100": "tripScore",
                    "仆役宫_100": "interpersonalScore", 
                }

                # print("camel_dict",camel_dict)

                for cn_key, en_key in score_keys.items():
                    if cn_key in original_dict:
                        score_value = original_dict[cn_key]
                        try:
                            score_numeric = float(score_value)
                            camel_dict[en_key] = int(round(score_numeric))
                            individual_scores.append(score_numeric)
                        except (ValueError, TypeError):
                            camel_dict[en_key] = random.randint(65, 75)  # 解析失败随机兜底
                    elif en_key not in camel_dict:
                         camel_dict[en_key] = random.randint(65, 75)  # 缺失随机兜底
                
                # 综合评分的计算逻辑
                composite_score_val = original_dict.get("综合评分")
                if isinstance(composite_score_val, (int, float)):
                    camel_dict['compositeScore'] = int(round(float(composite_score_val)))
                elif individual_scores:
                    try:
                        camel_dict['compositeScore'] = int(round(sum(individual_scores) / len(individual_scores)))
                    except (ValueError, TypeError):
                        camel_dict['compositeScore'] = random.randint(65, 75)  # 解析失败随机兜底
                else:
                    camel_dict['compositeScore'] = random.randint(65, 75)  # 缺失随机兜底


                # --- 步骤 3 & 4: 建议和避免字段的严格约束 (正确) ---
                suggestions = original_dict.get("建议")
                if isinstance(suggestions, list) and len(suggestions) >= 2:
                    camel_dict['suggestions'] = suggestions[:2]
                else:
                    camel_dict['suggestions'] = default_suggestions

                avoidances = original_dict.get("避免")
                if isinstance(avoidances, list) and len(avoidances) >= 2:
                    camel_dict['avoidances'] = avoidances[:2]
                else:
                    camel_dict['avoidances'] = default_avoidances

                print("original_dict",original_dict)

                # --- 步骤 4: 运势详情字段 ---
                # 定义领域映射
                category_mapping = [
                    ("感情", "relationship"),
                    ("事业", "career"),
                    ("财富", "wealth"),
                    ("健康", "healthy"),
                    ("出行", "trip"),
                    ("人际", "interpersonal")
                ]
                
                # 存储各领域的分数和运势详解，用于后续选择建议
                domain_fortune_details = {}
                
                for cn_category, en_prefix in category_mapping:
                    camel_dict[f'{en_prefix}Title'] = normalize_llm_text(original_dict.get(f"{cn_category}标题"))
                    fortune_detail = normalize_llm_text(original_dict.get(f"{cn_category}运势详解"))
                    camel_dict[f'{en_prefix}FortuneDetail'] = fortune_detail
                    camel_dict[f'{en_prefix}ShareDetail'] = normalize_llm_text(original_dict.get(f"{cn_category}分享话语"))
                    camel_dict[f'{en_prefix}GuideQuestions'] = original_dict.get(f"{cn_category}引导问题")
                    
                    # 存储运势详解，用于后续选择建议
                    domain_fortune_details[en_prefix] = fortune_detail

                # --- 【新增】: 根据 compositeScore 和各领域分数生成 fortuneJudgment 和 fortuneJudgmentDetail ---
                composite_score = camel_dict.get('compositeScore', 0)
                
                # 定义6个领域的映射
                domain_list = ["relationship", "career", "wealth", "healthy", "trip", "interpersonal"]
                
                # 获取各领域的分数
                domain_score_map = {}
                for domain in domain_list:
                    score_key = f"{domain}Score"
                    if score_key in camel_dict:
                        domain_score_map[domain] = camel_dict[score_key]
                
                fortune_judgment_value = None
                if domain_score_map:
                    # 找到最高分和最低分的领域
                    max_score = max(domain_score_map.values())
                    min_score = min(domain_score_map.values())
                    
                    # 获取所有最高分的领域
                    max_score_domains = [domain for domain, score in domain_score_map.items() if score == max_score]
                    # 获取所有最低分的领域
                    min_score_domains = [domain for domain, score in domain_score_map.items() if score == min_score]
                    
                    # 为各领域预先抽取一条建议（供详情与首页共用）
                    domain_single_suggestion = {}
                    for domain in domain_score_map.keys():
                        fortune_detail = domain_fortune_details.get(domain, "")
                        suggestions = extract_suggestions_from_fortune_detail(fortune_detail)
                        # 放宽为至少 1 条建议即可使用；避免最低分域无建议导致详情缺失
                        if suggestions:
                            domain_single_suggestion[domain] = normalize_llm_text(random.choice(suggestions))
                    
                    # 生成 fortuneJudgmentDetail（最高分域 + 最低分域，分值并列时全部域都展示；同域不重复）
                    detail_parts = []
                    added_domains = set()
                    
                    for domain in max_score_domains:
                        suggestion = domain_single_suggestion.get(domain)
                        if suggestion:
                            detail_parts.append(suggestion)  # 已在存储时规范化
                            added_domains.add(domain)
                    
                    for domain in min_score_domains:
                        if domain in added_domains:
                            continue
                        suggestion = domain_single_suggestion.get(domain)
                        if suggestion:
                            detail_parts.append(suggestion)  # 已在存储时规范化
                            added_domains.add(domain)
                    
                    # 生成 fortuneJudgment（从详情可找到的同一建议；高于等于80选最高分域，其余选最低分域；并列时只取一个域）
                    if composite_score >= 80:
                        selected_domain = random.choice(max_score_domains) if max_score_domains else domain_list[0]
                    else:
                        selected_domain = random.choice(min_score_domains) if min_score_domains else domain_list[0]
                    
                    fortune_judgment_value = domain_single_suggestion.get(selected_domain)
                    if fortune_judgment_value:
                        camel_dict['fortuneJudgment'] = normalize_llm_text(fortune_judgment_value)
                    else:
                        default_judgment = original_dict.get("运势判断", "今日运势平稳，保持耐心。")
                        camel_dict['fortuneJudgment'] = normalize_llm_text(default_judgment)
                        fortune_judgment_value = normalize_llm_text(default_judgment)
                    
                    if detail_parts:
                        camel_dict['fortuneJudgmentDetail'] = normalize_llm_text("\n".join(detail_parts))
                    else:
                        # 如果没有提取到建议，使用默认值
                        default_detail = original_dict.get("运势判断详情", "今日运势平稳，保持耐心，寻求支持。")
                        camel_dict['fortuneJudgmentDetail'] = normalize_llm_text(default_detail)
                else:
                    # 如果没有领域分数，使用原来的值
                    default_judgment = original_dict.get("运势判断", "今日运势平稳，保持耐心。")
                    camel_dict['fortuneJudgment'] = normalize_llm_text(default_judgment)
                    default_detail = original_dict.get("运势判断详情", "今日运势平稳，保持耐心，寻求支持。")
                    
                    # 详情里也包含 fortuneJudgment
                    detail_parts = []
                    if default_judgment:
                        detail_parts.append(normalize_llm_text(default_judgment))
                    if default_detail:
                        detail_parts.append(normalize_llm_text(default_detail))
                    camel_dict['fortuneJudgmentDetail'] = normalize_llm_text("\n".join(detail_parts))

                # 过滤空值

                # print("camel_dict",camel_dict)
                final_dict = {k: v for k, v in camel_dict.items() if v is not None}
                final_camel_case_data.append(final_dict)

        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"API 总耗时: {total_time:.2f}秒")
        
        # 记录总耗时
        log_step("输出给客户端", request_id=request_id, extra_data={
            "total_time": f"{total_time:.2f}秒",
            "result_count": len(final_camel_case_data)
        })

        return final_camel_case_data

    except Exception as e:
        logger.critical(f"处理请求时发生未预料的严重错误: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"服务器内部错误: {e}"
        )


