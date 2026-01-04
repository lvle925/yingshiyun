# app/api.py
import asyncio
import json
import logging
from datetime import datetime, timedelta, date, time
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
import random
import uvicorn

# 导入重构后的模块
from .processing import get_yearly_scores_only, get_monthly_scores
from .utils import convert_time_to_time_index
import os
import time


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

PALACE_TO_DIMENSION = {
    "官禄宫_100": "careerScore",
    "财帛宫_100": "wealthScore",
    "夫妻宫_100": "relationshipScore",
    "疾厄宫_100": "healthyScore",
    "迁移宫_100": "tripScore",
    "仆役宫_100": "interpersonalScore",
}

DIMENSION_LABELS = {
    "compositeScore": "整体",
    "careerScore": "事业",
    "wealthScore": "财富",
    "relationshipScore": "感情",
    "healthyScore": "健康",
    "interpersonalScore": "人际",
    "tripScore": "出行",
}


def map_palace_scores_to_dimensions(scaled_scores_map: dict, composite_score: float) -> dict:
    dimension_scores = {}
    for palace_key, dimension_key in PALACE_TO_DIMENSION.items():
        score = scaled_scores_map.get(palace_key)
        if score is None:
            score = get_randomized_score(composite_score * 0.9 + random.randint(-10, 10))
        else:
            score = int(round(score))
        dimension_scores[dimension_key] = score
    return dimension_scores


def _generate_random_monthly_scores(dimension_scores: dict, composite_score: float) -> list:
    dimensions = [
        (DIMENSION_LABELS["compositeScore"], "compositeScore", composite_score),
        (DIMENSION_LABELS["careerScore"], "careerScore", dimension_scores.get("careerScore", composite_score)),
        (DIMENSION_LABELS["wealthScore"], "wealthScore", dimension_scores.get("wealthScore", composite_score)),
        (DIMENSION_LABELS["relationshipScore"], "relationshipScore", dimension_scores.get("relationshipScore", composite_score)),
        (DIMENSION_LABELS["healthyScore"], "healthyScore", dimension_scores.get("healthyScore", composite_score)),
        (DIMENSION_LABELS["interpersonalScore"], "interpersonalScore", dimension_scores.get("interpersonalScore", composite_score)),
        (DIMENSION_LABELS["tripScore"], "tripScore", dimension_scores.get("tripScore", composite_score)),
    ]
    
    monthly_scores = []
    
    for dimension_type, dimension_key, base_score in dimensions:
        score_info = []
        base_int = int(round(base_score))
        monthly_variations = [random.randint(-12, 12) for _ in range(12)]
        total_variation = sum(monthly_variations)
        avg_adjustment = total_variation // 12
        monthly_variations = [v - avg_adjustment for v in monthly_variations]
        
        for month in range(1, 13):
            month_variation = monthly_variations[month - 1]
            month_score = base_int + month_variation
            month_score = max(40, min(95, month_score))
            score_info.append({
                "month": month,
                "score": month_score
            })
        
        monthly_scores.append({
            "dimensionType": dimension_type,
            "scoreInfo": score_info
        })
    
    return monthly_scores


def generate_monthly_scores(monthly_results: list, fallback_dimension_scores: dict, fallback_composite_score: float) -> list:
    if not monthly_results:
        return _generate_random_monthly_scores(fallback_dimension_scores, fallback_composite_score)
    
    dimension_month_map = {key: [] for key in DIMENSION_LABELS.keys()}
    
    for result in monthly_results:
        month = result.get("month")
        composite = result.get("composite_score", fallback_composite_score)
        scaled_map = result.get("scaled_scores_map", {})
        dimension_scores = map_palace_scores_to_dimensions(scaled_map, composite)
        
        dimension_month_map["compositeScore"].append({"month": month, "score": int(round(composite))})
        for dimension_key, score in dimension_scores.items():
            dimension_month_map.setdefault(dimension_key, []).append({
                "month": month,
                "score": score
            })
    
    formatted = []
    for dimension_key, label in DIMENSION_LABELS.items():
        month_entries = sorted(dimension_month_map.get(dimension_key, []), key=lambda x: x["month"])
        formatted.append({
            "dimensionType": label,
            "scoreInfo": month_entries
        })
    return formatted


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



app = FastAPI(
    title="紫微斗数年运势评分 API",
    description="输入出生日期和年份范围，生成各年的宫位分数和综合评分。"
)


class YearlyAnalysisRequest(BaseModel):
    gender: str = Field(..., description="性别 (男/女)", example="男")
    birthday: str = Field(..., description="生辰 (yyyy-MM-dd HH:mm:ss格式)", example="1995-05-06 14:30:00")
    year: int = Field(..., description="年份 (例如2026)", example=2026)

    @validator('gender')
    def gender_must_be_valid(cls, v):
        if v not in ['男', '女']:
            raise ValueError('性别必须是 "男" 或 "女"')
        return v
        
    @validator('birthday')
    def birthday_format_must_be_valid(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            raise ValueError('生辰格式必须是 "yyyy-MM-dd HH:mm:ss"')
        return v


# --- 应用生命周期事件 ---
@app.on_event("startup")
async def startup_event():
    logger.info("应用启动...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("应用关闭...")



@app.post("/yearFortuneScore", summary="生成年度运势评分")
async def generate_yearly_scores(request: YearlyAnalysisRequest):
    """ 
    根据指定的出生年月日时、性别和年份，生成各维度的年度分数和月度分数。
    只返回分数数据，不包含LLM建议。
    """
    start_time = time.time()
    try:
        # 解析birthday字段
        birthday_dt = datetime.strptime(request.birthday, '%Y-%m-%d %H:%M:%S')
        birth_date_str = birthday_dt.strftime('%Y-%m-%d')
        birth_time_obj = birthday_dt.time()
        time_index = convert_time_to_time_index(birth_time_obj)
        year = request.year

        base_payload_params = {
            "type": "solar",
            "timeIndex": time_index,
            "gender": request.gender,
        }

        logger.info(f"请求参数解析成功: timeIndex={time_index}, gender={request.gender}, 年份: {year}")

        # 处理单一年份
        horoscope_date_str = f"{year}-12-31 23:59:00"  # 使用年底作为预测日期，避免农历跨年误差
        
        payload = {
            "dateStr": birth_date_str,
            **base_payload_params,
            "horoscopeDate": horoscope_date_str
        }
        
        # 获取年度分数
        result = await get_yearly_scores_only(**payload)
        
        if isinstance(result, dict) and "error" in result:
            logger.error(f"获取 {year} 年数据失败: {result['error']}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"获取数据失败: {result['error']}"
            )
        
        if not isinstance(result, dict) or 'composite_score' not in result:
            logger.warning(f"在 {year} 年的响应中返回了未知的数据格式。")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="返回数据格式错误"
            )
        
        # 提取宫位分数
        scaled_scores_map = result.get("scaled_scores_map", {})
        composite_score = result.get("composite_score", 0)
        
        dimension_scores = map_palace_scores_to_dimensions(scaled_scores_map, composite_score)

        monthly_results = await get_monthly_scores(
            birth_date_str,
            base_payload_params["type"],
            time_index,
            request.gender,
            year
        )
        monthly_scores = generate_monthly_scores(
            monthly_results,
            dimension_scores,
            composite_score
        )
        
        # 构建响应
        response_data = {
            "compositeScore": int(round(composite_score)),
            "careerScore": dimension_scores["careerScore"],
            "wealthScore": dimension_scores["wealthScore"],
            "relationshipScore": dimension_scores["relationshipScore"],
            "healthyScore": dimension_scores["healthyScore"],
            "interpersonalScore": dimension_scores["interpersonalScore"],
            "tripScore": dimension_scores["tripScore"],
            "monthlyScore": monthly_scores
        }
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"API 总耗时: {total_time:.2f}秒")

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.critical(f"处理请求时发生未预料的严重错误: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"服务器内部错误: {e}"
        )