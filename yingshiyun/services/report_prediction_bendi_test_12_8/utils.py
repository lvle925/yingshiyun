import json
import re
import hmac
import hashlib
from typing import Dict, Any, List, Optional
from zhdate import ZhDate

def convert_chinese_month_to_number(month_str: str) -> str:
    """
    将中文月份转换为数字月份
    例如："正月" -> "1月", "二月" -> "2月", "十二月" -> "12月"
    """
    month_map = {
        "正月": "1月", "一月": "1月",
        "二月": "2月",
        "三月": "3月",
        "四月": "4月",
        "五月": "5月",
        "六月": "6月",
        "七月": "7月",
        "八月": "8月",
        "九月": "9月",
        "十月": "10月",
        "十一月": "11月",
        "十二月": "12月"
    }
    return month_map.get(month_str, month_str)


def replace_chinese_months_in_text(text: str) -> str:
    """
    替换文本中所有的中文月份为数字月份
    例如："三月" -> "3月", "十二月" -> "12月"
    """
    if not isinstance(text, str):
        return text
    
    # 按长度从长到短排序，避免"十一月"被"一月"先匹配
    month_replacements = [
        ("十一月", "11月"),
        ("十二月", "12月"),
        ("十月", "10月"),
        ("正月", "1月"),
        ("一月", "1月"),
        ("二月", "2月"),
        ("三月", "3月"),
        ("四月", "4月"),
        ("五月", "5月"),
        ("六月", "6月"),
        ("七月", "7月"),
        ("八月", "8月"),
        ("九月", "9月"),
    ]
    
    result = text
    for chinese_month, number_month in month_replacements:
        result = result.replace(chinese_month, number_month)
    
    return result

def sort_months_by_number(months: list) -> list:
    """
    对月份列表按数字顺序排序
    例如：["五月", "八月", "十二月"] -> ["5月", "8月", "12月"] -> ["5月", "8月", "12月"]
    """
    if not months:
        return months
    
    # 转换为数字月份
    number_months = [convert_chinese_month_to_number(m) for m in months]
    
    # 提取数字并排序
    def get_month_number(month_str: str) -> int:
        # 从"3月"中提取数字3
        try:
            return int(month_str.replace("月", ""))
        except:
            return 0
    
    # 按数字排序
    sorted_months = sorted(number_months, key=get_month_number)
    return sorted_months


def convert_and_sort_months_in_dimensions(data):
    """
    递归处理JSON数据，将dimensionDetails中的goodMonths和badMonths转换为数字月份并排序
    同时替换所有文本字段中的中文月份为数字月份
    """
    if isinstance(data, dict):
        # 如果是dimensionDetails中的维度对象
        if "goodMonths" in data and isinstance(data["goodMonths"], list):
            data["goodMonths"] = sort_months_by_number(data["goodMonths"])
        if "badMonths" in data and isinstance(data["badMonths"], list):
            data["badMonths"] = sort_months_by_number(data["badMonths"])
        
        # 递归处理所有值，并对字符串值进行月份替换
        for key, value in data.items():
            if isinstance(value, str):
                # 对字符串字段进行月份替换
                data[key] = replace_chinese_months_in_text(value)
            else:
                # 递归处理非字符串值
                convert_and_sort_months_in_dimensions(value)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, str):
                # 对列表中的字符串进行月份替换
                data[i] = replace_chinese_months_in_text(item)
            else:
                # 递归处理非字符串项
                convert_and_sort_months_in_dimensions(item)
    
    return data
# 农历月份名称转换
def convert_lunar_month_name(month_str: str) -> str:
    """将'农历X月'转换为'X月'格式"""
    month_map = {
        "一月": "正月", "农历一月": "正月",
        "二月": "二月", "农历二月": "二月",
        "三月": "三月", "农历三月": "三月",
        "四月": "四月", "农历四月": "四月",
        "五月": "五月", "农历五月": "五月",
        "六月": "六月", "农历六月": "六月",
        "七月": "七月", "农历七月": "七月",
        "八月": "八月", "农历八月": "八月",
        "九月": "九月", "农历九月": "九月",
        "十月": "十月", "农历十月": "十月",
        "十一月": "十一月", "农历十一月": "十一月",
        "十二月": "十二月", "农历十二月": "十二月"
    }
    return month_map.get(month_str, month_str)

def replace_mingnian_with_year(data, target_year: int):
    """
    递归替换JSON数据中所有的"明年"为指定年份（如"2026年"）
    
    Args:
        data: 要处理的数据（可以是dict、list或str）
        target_year: 目标年份
    
    Returns:
        处理后的数据
    """
    year_str = f"{target_year}年"
    
    if isinstance(data, dict):
        return {key: replace_mingnian_with_year(value, target_year) for key, value in data.items()}
    elif isinstance(data, list):
        return [replace_mingnian_with_year(item, target_year) for item in data]
    elif isinstance(data, str):
        return data.replace("明年", year_str)
    else:
        return data


def get_lunar_first_days(year):
    lunar_first_days = []
    for month in range(1, 13):
        try:
            lunar_date = ZhDate(year, month, 1)
            gregorian_date = lunar_date.to_datetime().replace(hour=12, minute=0, second=0)
            lunar_first_days.append((month, gregorian_date))
        except Exception as e:
            print(f"警告：无法获取农历{year}年{month}月初一的公历日期：{e}")
            lunar_first_days.append((month, None))
    return lunar_first_days


def get_time_period_description(top_months):
    if not top_months:
        return "全年"
    months = [month_info["month_index"] for month_info in top_months if "month_index" in month_info]  # 确保month_info是字典
    if not months:  # 如果没有有效月份数据，返回全年
        return "全年"

    # 转换为1-12的月份数字
    months_numeric = [m for m in months]

    q1 = [m for m in months_numeric if 1 <= m <= 3]
    q2 = [m for m in months_numeric if 4 <= m <= 6]
    q3 = [m for m in months_numeric if 7 <= m <= 9]
    q4 = [m for m in months_numeric if 10 <= m <= 12]
    quarters = [
        ("第一季度", len(q1)), ("第二季度", len(q2)), ("第三季度", len(q3)), ("第四季度", len(q4))
    ]
    max_count = max(count for _, count in quarters)
    dominant_quarters = [name for name, count in quarters if count == max_count and count > 0]
    if len(dominant_quarters) == 1:
        return dominant_quarters[0]
    elif len(dominant_quarters) == 2:
        return f"{dominant_quarters[0]}和{dominant_quarters[1]}"
    elif len(dominant_quarters) >= 3:
        return "全年分散"
    else:
        return "全年"


def calculate_time_index(hour: int, minute: int) -> Optional[int]:
    """
    根据小时和分钟计算时辰索引。
    这是您提供的逻辑，并进行了微调以确保明确的区间。
    """
    total_minutes = hour * 60 + minute

    # 标准时辰区间，精确到分钟
    if 0 <= total_minutes < 60:
        return 0  # 00:00 - 00:59 (早子)
    elif 60 <= total_minutes < 180:
        return 1  # 01:00 - 02:59 (丑时)
    elif 180 <= total_minutes < 300:
        return 2  # 03:00 - 04:59 (寅时)
    elif 300 <= total_minutes < 420:
        return 3  # 05:00 - 06:59 (卯时)
    elif 420 <= total_minutes < 540:
        return 4  # 07:00 - 08:59 (辰时)
    elif 540 <= total_minutes < 660:
        return 5  # 09:00 - 10:59 (巳时)
    elif 660 <= total_minutes < 780:
        return 6  # 11:00 - 12:59 (午时)
    elif 780 <= total_minutes < 900:
        return 7  # 13:00 - 14:59 (未时)
    elif 900 <= total_minutes < 1020:
        return 8  # 15:00 - 16:59 (申时)
    elif 1020 <= total_minutes < 1140:
        return 9  # 17:00 - 18:59 (酉时)
    elif 1140 <= total_minutes < 1260:
        return 10  # 19:00 - 20:59 (戌时)
    elif 1260 <= total_minutes < 1380:
        return 11  # 21:00 - 22:59 (亥时)
    elif 1380 <= total_minutes < 1440:
        return 12  # 23:00 - 23:59 (晚子)

    return None  # Invalid time


def aggregate_monthly_data_by_period(monthly_data_list_of_dicts, time_period):
    """
    根据月度数据分析好坏月份（修改版）

    Args:
        monthly_data_list_of_dicts (list): 原始月度数据列表，每个元素是字典，格式如 {"month_index": 4, "total_score": 27.0, "combined_info": {...}}
        time_period (str): 时间段描述

    Returns:
        dict: 包含好坏月份分析的数据字典，格式为 {"好月份": [...], "坏月份": [...]}
    """
    if not monthly_data_list_of_dicts:
        return {"好月份": [], "坏月份": []}

    all_monthly_data = monthly_data_list_of_dicts  # 直接使用传入的字典列表

    # 按分数排序
    all_monthly_data.sort(key=lambda x: x["total_score"], reverse=True)

    # 获取好的月份（分数较高的前3个）和坏的月份（分数较低的后3个）
    good_months_candidates = all_monthly_data[:3]
    bad_months_candidates = all_monthly_data[-3:]

    good_months_final = []
    bad_months_final = []

    seen_months_indices = set()

    for month_info in good_months_candidates:
        if month_info["month_index"] not in seen_months_indices:
            good_months_final.append(month_info)
            seen_months_indices.add(month_info["month_index"])

    for month_info in bad_months_candidates:
        if month_info["month_index"] not in seen_months_indices:
            bad_months_final.append(month_info)

    month_names_chinese = ["", "一月", "二月", "三月", "四月", "五月", "六月",
                           "七月", "八月", "九月", "十月", "十一月", "十二月"]

    good_months_data = []
    for month_info in good_months_final:
        month_num = month_info["month_index"]
        score = month_info["total_score"]
        combined_data_json = json.dumps(month_info["combined_info"], ensure_ascii=False)  # 确保是字符串
        good_months_data.append(f"农历{month_names_chinese[month_num]}: [分数:{score}, {combined_data_json}]")

    bad_months_data = []
    for month_info in bad_months_final:
        month_num = month_info["month_index"]
        score = month_info["total_score"]
        combined_data_json = json.dumps(month_info["combined_info"], ensure_ascii=False)  # 确保是字符串
        bad_months_data.append(f"农历{month_names_chinese[month_num]}: [分数:{score}, {combined_data_json}]")

    return {
        "好月份": good_months_data,
        "坏月份": bad_months_data
    }


def extract_score_from_data(data_str):
    import re
    score_match = re.search(r'\[(\d+\.?\d*),', data_str)
    if score_match:
        return float(score_match.group(1))
    return 0.0


def normalize_score(raw_score: float, min_output: float = 40.0, max_output: float = 95.0) -> float:
    """
    将原始分数归一化到指定范围内，使用固定的映射范围确保所有用户结果一致
    """
    min_input = -10.0
    max_input = 15.0

    clamped_score = max(min_input, min(raw_score, max_input))

    input_range = max_input - min_input
    output_range = max_output - min_output

    normalized_score = min_output + (clamped_score - min_input) * output_range / input_range

    return round(normalized_score, 1)


def verify_signature(params: Dict[str, Any], app_secret: str) -> bool:
    if 'sign' not in params:
        return False
    client_sign = params['sign']
    sorted_params = dict(sorted({k: str(v) for k, v in params.items() if k != 'sign'}.items()))
    string_to_sign = "".join(f"{k}{v}" for k, v in sorted_params.items())
    secret_bytes = app_secret.encode('utf-8')
    string_to_sign_bytes = string_to_sign.encode('utf-8')
    calculated_sign = hmac.new(secret_bytes, string_to_sign_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(client_sign, calculated_sign)