# utils.py
import json
import re
from datetime import datetime
from typing import Dict, Any
from lunardate import LunarDate
from datetime import datetime, timedelta
from typing import AsyncGenerator, Dict, Any, Tuple, Optional, Literal, List
import calendar
from config import YANG_GAN, YIN_GAN, ALL_PALACES, DIZHI_DUIGONG_MAP
import logging
import traceback

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# from ziwei_ai_function import transform_horoscope_scope_data, ALL_EARTHLY_BRANCHES, FIXED_PALACE_ORDER_FOR_SCOPES, \
#    HEAVENLY_STEM_MUTAGEN_MAP, get_ordered_palace_branches, get_mutagen_for_stem, transform_palace_data, \
#    _parse_lenient_json, \
#    _extract_first_json_object, _close_open_brackets, simple_clean_birth_info, simple_clean_query_intent, \
#    validate_branch_in_prompt, \
#    is_valid_datetime_string, validate_ziwei_payload, validate_birth_info_logic, get_lunar_month_range_string, \
#    YANG_GAN, YIN_GAN, calculate_all_decadal_periods, save_string_to_file, VALID_XINGXI_DIZHI_COMBOS

def parse_multiple_years(expression: str, current_year: int) -> List[int]:
    """
    解析表达式中的多个年份
    "26 27 28年" → [2026, 2027, 2028]
    "2026 2027年" → [2026, 2027]
    "2026年2027年" → [2026, 2027]
    "未来3年" → [current_year, current_year+1, current_year+2]
    "今后十年" → [current_year, ..., current_year+9]
    """
    years = []

    # 中文数字映射
    chinese_nums = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}

    # 匹配"未来N年"、"今后N年"、"往后N年"、"后N年"、"下N年"（支持中文数字）
    future_match = re.search(r'(未来|今后|往后|接下来|后|下)(十|\d+)年', expression)
    if future_match:
        num_str = future_match.group(2)
        n = chinese_nums.get(num_str, int(num_str) if num_str.isdigit() else 10)
        return list(range(current_year, current_year + n))

    # 匹配所有年份数字（2位或4位）
    year_matches = re.findall(r'\b(\d{2,4})\b', expression)
    for year_str in year_matches:
        year_num = int(year_str)
        # 2位年份补全为4位
        if year_num < 100:
            if year_num > 50:
                year_num += 1900
            else:
                year_num += 2000
        # 验证年份合理性（1900-2100）
        if 1900 <= year_num <= 2100:
            years.append(year_num)

    # 去重并排序
    return sorted(list(set(years)))


def get_decadals_in_time_span(
        birth_year: int,
        current_age: int,
        span_years: int,
        all_decadal_ages: Dict[str, Tuple[int, int]]
) -> List[Dict[str, Any]]:
    """
    计算指定年数内包含哪些大运。
    返回: [
        {"ganzhi": "丙寅", "start_age": 32, "end_age": 41, "start_year": 2022, "end_year": 2031, "covered_years": 10},
        {"ganzhi": "丁卯", "start_age": 42, "end_age": 43, "start_year": 2032, "end_year": 2033, "covered_years": 2}
    ]
    """
    result = []
    target_end_age = current_age + span_years

    for ganzhi, (start_age, end_age) in all_decadal_ages.items():
        # 检查这个大运是否与目标时间段有交集
        if start_age <= target_end_age and end_age >= current_age:
            # 计算实际覆盖的年龄范围
            actual_start = max(start_age, current_age)
            actual_end = min(end_age, target_end_age)
            covered_years = actual_end - actual_start + 1

            result.append({
                "ganzhi": ganzhi,
                "start_age": actual_start,
                "end_age": actual_end,
                "start_year": birth_year + actual_start - 1,
                "end_year": birth_year + actual_end - 1,
                "covered_years": covered_years
            })

    # 按起始年龄排序
    result.sort(key=lambda x: x['start_age'])
    return result


def get_decadals_by_count(
        birth_year: int,
        current_age: int,
        decadal_count: int,
        all_decadal_ages: Dict[str, Tuple[int, int]]
) -> List[Dict[str, Any]]:
    """
    根据大运数量计算未来N个大运。
    
    参数:
        birth_year: 出生年份
        current_age: 当前年龄（虚岁）
        decadal_count: 要查询的大运数量（如：2表示未来2个大运）
        all_decadal_ages: 所有大运的年龄范围字典
    
    返回: [
        {"ganzhi": "丙寅", "start_age": 32, "end_age": 41, "start_year": 2022, "end_year": 2031, "covered_years": 10},
        {"ganzhi": "丁卯", "start_age": 42, "end_age": 51, "start_year": 2032, "end_year": 2041, "covered_years": 10}
    ]
    """
    result = []
    
    # 找到包含当前年龄的大运
    current_decadal = None
    for ganzhi, (start_age, end_age) in all_decadal_ages.items():
        if start_age <= current_age <= end_age:
            current_decadal = {
                "ganzhi": ganzhi,
                "start_age": start_age,
                "end_age": end_age,
                "start_year": birth_year + start_age - 1,
                "end_year": birth_year + end_age - 1,
                "covered_years": end_age - start_age + 1
            }
            break
    
    if not current_decadal:
        # 如果找不到当前年龄对应的大运，从第一个大于当前年龄的大运开始
        for ganzhi, (start_age, end_age) in sorted(all_decadal_ages.items(), key=lambda x: x[1][0]):
            if start_age > current_age:
                current_decadal = {
                    "ganzhi": ganzhi,
                    "start_age": start_age,
                    "end_age": end_age,
                    "start_year": birth_year + start_age - 1,
                    "end_year": birth_year + end_age - 1,
                    "covered_years": end_age - start_age + 1
                }
                break
    
    if not current_decadal:
        return result
    
    # 按起始年龄排序所有大运
    sorted_decadals = sorted(all_decadal_ages.items(), key=lambda x: x[1][0])
    
    # 找到当前大运在排序列表中的位置
    current_index = -1
    for idx, (ganzhi, (start_age, end_age)) in enumerate(sorted_decadals):
        if ganzhi == current_decadal["ganzhi"]:
            current_index = idx
            break
    
    if current_index == -1:
        return result
    
    # 从当前大运开始，获取未来N个大运
    # 如果当前大运还未结束，从当前大运开始；如果已结束，从下一个大运开始
    start_index = current_index
    if current_age > current_decadal["end_age"]:
        start_index = current_index + 1
    
    # 获取未来N个大运
    for i in range(decadal_count):
        if start_index + i >= len(sorted_decadals):
            break
        
        ganzhi, (start_age, end_age) = sorted_decadals[start_index + i]
        
        # 如果是第一个大运且当前年龄还在其中，需要调整起始年龄
        actual_start = max(start_age, current_age) if i == 0 and start_index == current_index else start_age
        actual_end = end_age
        covered_years = actual_end - actual_start + 1
        
        result.append({
            "ganzhi": ganzhi,
            "start_age": actual_start,
            "end_age": actual_end,
            "start_year": birth_year + actual_start - 1,
            "end_year": birth_year + actual_end - 1,
            "covered_years": covered_years
        })
    
    return result


def calculate_all_decadal_periods(
        birth_year: int,
        gender: str,
        year_gan: str,  # 例如 "阳男", "阴女"
        wuxingju: str,
        palaces: List[Dict]  # API返回的完整宫位列表
) -> Optional[Dict[str, Tuple[int, int]]]:
    """
    从API返回的palaces数据中提取所有大限的干支及其对应的年龄范围。
    【关键修复】直接使用API返回的range，而不是自己计算，确保与API保持一致。
    返回一个字典，键为大运干支，值为 (起始年龄, 结束年龄) 的元组。
    例如: {'丙寅': (2, 11), '丁卯': (12, 21), ...}
    """
    if not all([birth_year, gender, year_gan, wuxingju, palaces]):
        return None

    try:
        # 【核心修正】: 只保留这一种正确的、基于年干和性别的判断逻辑
        is_forward = False
        if (gender == '男' and year_gan in YANG_GAN) or \
                (gender == '女' and year_gan in YIN_GAN):
            is_forward = True

        # 找到命宫的索引
        ming_gong_index = -1
        for i, palace in enumerate(palaces):
            if palace.get("name") == "命宫":
                ming_gong_index = i
                break

        if ming_gong_index == -1:
            return None

        decadal_periods = {}

        # 循环12个宫位来确定12个大限
        for i in range(12):
            index = 0
            if is_forward:
                index = (ming_gong_index + i) % 12
            else:
                index = (ming_gong_index - i + 12) % 12

            palace_info = palaces[index]
            decadal_info = palace_info.get("decadal", {})
            decadal_stem = decadal_info.get("heavenlyStem")
            decadal_branch = decadal_info.get("earthlyBranch")
            decadal_range = decadal_info.get("range", [])

            if decadal_stem and decadal_branch:
                decadal_gan_zhi = f"{decadal_stem}{decadal_branch}"
                
                # 【关键修复】直接使用API返回的range，而不是自己计算
                if isinstance(decadal_range, list) and len(decadal_range) == 2:
                    start_age = decadal_range[0]
                    end_age = decadal_range[1]
                    decadal_periods[decadal_gan_zhi] = (start_age, end_age)
                else:
                    # 如果API没有返回range，fallback到原来的计算方式
                    start_age_first_decadal = calculate_decadal_start_age_exact(wuxingju)
                    # 计算这是第几个大运
                    decadal_index = i
                    start_age = start_age_first_decadal + decadal_index * 10
                    end_age = start_age + 9
                    decadal_periods[decadal_gan_zhi] = (start_age, end_age)

        return decadal_periods
    except Exception as e:
        print(f"Error calculating all decadal periods: {e}")
        return None


from datetime import datetime, timedelta
# from collections import Counter # V13 中未使用，可以移除
# 【核心】: 现在我们只需要 LunarDate 这一个类
from lunardate import LunarDate


def get_leap_month(lunar_year):
    """返回农历年份中的闰月（如果无闰月则返回0）"""
    try:
        for m in range(1, 13):
            try:
                _ = LunarDate(lunar_year, -m, 1)
                return m
            except ValueError:
                continue
        return 0
    except Exception:
        return 0


def to_chinese_month_name(lunar_month):
    """将农历月份数字转换为中文月名"""
    month_names = {
        1: "正月", 2: "二月", 3: "三月", 4: "四月", 5: "五月", 6: "六月",
        7: "七月", 8: "八月", 9: "九月", 10: "十月", 11: "冬月", 12: "腊月"
    }
    abs_month = abs(lunar_month)
    base_name = month_names.get(abs_month, f"{abs_month}月")
    return "闰" + base_name if lunar_month < 0 else base_name


# --- 【核心函数 V13-StrictEnhanced - 在V13基础上严格附加天数规则】 ---
def get_lunar_month_range_string(dt_obj):
    """
    【V13-StrictEnhanced-Fixed版】
    在V13基础上严格附加天数规则 + 修正闰月推算逻辑
    """
    try:
        # --- 步骤1：通过阳历获取农历锚点 ---
        anchor_lunar_date = LunarDate.fromSolarDate(dt_obj.year, dt_obj.month, dt_obj.day)
        target_lunar_year = anchor_lunar_date.year
        target_lunar_month = anchor_lunar_date.month

        # --- 步骤2：获取该农历月的起始阳历日期 ---
        base_lunar_month_name = to_chinese_month_name(target_lunar_month)
        first_day_lunar = LunarDate(target_lunar_year, target_lunar_month, 1)
        first_day_solar = first_day_lunar.toSolarDate()

        # --- 步骤3：推算下一个农历月 ---
        abs_month = abs(target_lunar_month)
        is_leap_by_astronomy = target_lunar_month < 0
        leap_month_in_year = get_leap_month(target_lunar_year)
        next_month_year = target_lunar_year

        # ✅ 修正版核心逻辑：严格控制闰月和跨年
        if is_leap_by_astronomy:
            # 当前是闰月 → 下个月是下一正常月
            next_month_num = abs_month + 1
        elif abs_month == leap_month_in_year:
            # 当前是普通月，下一月是对应闰月
            next_month_num = -abs_month
        elif abs_month == 12:
            # 年末跨年
            next_month_num = 1
            next_month_year += 1
        else:
            # 正常月份递增
            next_month_num = abs_month + 1

        # --- 步骤4：计算下个月首日的阳历时间，得到当前月的结束日 ---
        next_month_first_day_lunar = LunarDate(next_month_year, next_month_num, 1)
        next_month_first_day_solar = next_month_first_day_lunar.toSolarDate()
        last_day_solar = next_month_first_day_solar - timedelta(days=1)

        start_date_obj = first_day_solar
        end_date_obj = last_day_solar
        duration_days = (end_date_obj - start_date_obj).days + 1

        # 天数超过35天则认为是闰月（业务规则）
        is_leap_by_rule = duration_days > 35

        # --- 步骤5：确定最终月名与提示 ---
        final_lunar_month_name = base_lunar_month_name
        final_leap_month_notice = ""

        if is_leap_by_rule:
            if not base_lunar_month_name.startswith("闰"):
                final_lunar_month_name = "闰" + base_lunar_month_name
            final_leap_month_notice = "（注意：此农历月为闰月）"
        elif is_leap_by_astronomy:
            final_leap_month_notice = "（注意：本月为闰月）"

        # --- 步骤6：组装最终字符串 ---
        solar_month_str = dt_obj.strftime('%-m月')
        start_date_str = start_date_obj.strftime('%-m月%-d日')
        end_date_str = end_date_obj.strftime('%-m月%-d日')

        final_string = (
            f"针对农历“{final_lunar_month_name}”，对应阳历时间 {start_date_str} 至 {end_date_str}"
        )

        if final_leap_month_notice:
            final_string += f" {final_leap_month_notice}"

        return final_string

    except Exception as e:
        print(f"Error calculating lunar month range: {e}\n{traceback.format_exc()}")
        return f"针对公历{dt_obj.year}年{dt_obj.month}月"


def to_chinese_day_name(day: int) -> str:
    """将日转换为中文日名（初一、十五、廿一等）。"""
    day_prefix = ["初", "十", "廿", "三十"]
    numerals = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    if day == 10:
        return "初十"
    if day == 20:
        return "二十"
    if day == 30:
        return "三十"
    if day < 10:
        return f"初{numerals[day]}"
    if 10 < day < 20:
        return f"十{numerals[day-10]}"
    if 20 < day < 30:
        return f"廿{numerals[day-20]}"
    return f"三十"


def get_lunar_day_string(dt_obj: datetime) -> str:
    """
    将公历日期转换为农历日期描述，返回格式如：农历正月初五（对应阳历 02月14日）
    """
    lunar_date = LunarDate.fromSolarDate(dt_obj.year, dt_obj.month, dt_obj.day)
    lunar_month_name = to_chinese_month_name(lunar_date.month)
    lunar_day_name = to_chinese_day_name(lunar_date.day)
    solar_str = dt_obj.strftime('%m月%d日')
    return f"农历{lunar_month_name}{lunar_day_name}（对应阳历 {solar_str}）"

def get_solar_month_range_string(dt_obj: datetime) -> str:
    """
    计算公历月的完整日期范围（用于流月分析）。
    当用户询问"这个月的财运"时，应该返回公历月的完整日期范围（如10月1日-10月31日），
    而不是对应的农历月日期范围。

    Args:
        dt_obj: 公历日期对象

    Returns:
        格式化的日期范围字符串，例如：'针对阳历"10月"（对应农历"八月"），即阳历时间 10月1日 至 10月31日'
    """
    try:
        from datetime import date
        from calendar import monthrange

        year = dt_obj.year
        month = dt_obj.month

        # 计算公历月的第一天和最后一天
        first_day = date(year, month, 1)
        last_day_num = monthrange(year, month)[1]
        last_day = date(year, month, last_day_num)

        # 获取月中日期对应的农历月份信息（使用月中日期更准确）
        anchor_lunar_date = LunarDate.fromSolarDate(year, month, 15)
        lunar_month_name = to_chinese_month_name(anchor_lunar_date.month)

        # 格式化日期字符串
        solar_month_str = dt_obj.strftime('%-m月')
        start_date_str = first_day.strftime('%-m月%-d日')
        end_date_str = last_day.strftime('%-m月%-d日')

        final_string = (
            f"针对阳历\"{solar_month_str}\"（对应农历\"{lunar_month_name}\"），"
            f"即阳历时间 {start_date_str} 至 {end_date_str}"
        )

        return final_string

    except Exception as e:
        logger.error(f"Error calculating solar month range: {e}\n{traceback.format_exc()}")
        return f"针对公历{dt_obj.year}年{dt_obj.month}月"


def validate_branch_in_prompt(data_dict: dict, prompt_input: str) -> dict:
    """
    检查字典中 'traditional_hour_branch' 的值是否存在于 prompt_input 字符串中。
    如果存在，保持字典不变。
    如果不存在，则将 'traditional_hour_branch' 的值设为 None。

    Args:
        data_dict (dict): 包含个人信息的字典，例如 {'traditional_hour_branch': '戌', ...}
        prompt_input (str): 用于检查的字符串。

    Returns:
        dict: 处理后的字典。
    """
    # 使用 .get() 方法安全地获取值，如果键不存在，则返回 None
    branch_value = data_dict.get('traditional_hour_branch')

    # 如果 branch_value 本身就是 None 或空字符串，则无需检查，直接返回原字典
    if not branch_value:
        return data_dict

    # 检查 branch_value 是否在 prompt_input 字符串中
    # 如果没有出现过
    if branch_value not in prompt_input:
        # 将该键的值赋为 None
        data_dict['traditional_hour_branch'] = None

    # 如果出现过，则不执行任何操作，直接返回原字典
    return data_dict


def validate_birth_info_logic(info: dict):
    """
    对清洗后的出生信息字典进行业务逻辑验证。
    如果验证失败，会抛出 ValueError。

    Args:
        info (dict): 经过初步清洗和Pydantic验证的出生信息字典。
    """
    year = info.get('year')
    month = info.get('month')
    day = info.get('day')
    hour = info.get('hour')
    minute = info.get('minute')

    # 只有当年月日都存在时，才进行组合日期的验证
    if year is not None and month is not None and day is not None:
        # 验证年份范围
        current_year = datetime.now().year
        if not (1900 <= year <= current_year):
            raise ValueError(f"年份 '{year}' 超出合理范围 (1900-{current_year})。")

        # 验证月份范围
        if not (1 <= month <= 12):
            raise ValueError(f"月份 '{month}' 无效，必须在 1-12 之间。")

        # 验证日期对于年月是否有效
        # calendar.monthrange(year, month) 返回 (weekday, days_in_month)
        try:
            days_in_month = calendar.monthrange(year, month)[1]
            if not (1 <= day <= days_in_month):
                raise ValueError(f"日期 '{day}' 对于 {year}年{month}月 无效。该月只有 {days_in_month} 天。")
        except calendar.IllegalMonthError:
            # monthrange 会对无效月份抛出此异常，作为双重检查
            raise ValueError(f"月份 '{month}' 无效。")
        except TypeError:
            # 如果 year 或 month 不是整数，可能抛出此异常
            raise ValueError(f"验证日期时，年({year})或月({month})类型不正确。")

    # 验证小时范围
    if hour is not None and not (0 <= hour <= 23):
        raise ValueError(f"小时 '{hour}' 无效，必须在 0-23 之间。")

    # 验证分钟范围
    if minute is not None and not (0 <= minute <= 59):
        raise ValueError(f"分钟 '{minute}' 无效，必须在 0-59 之间。")


def _close_open_brackets(text: str) -> str:
    """辅助函数，计算并添加所有未闭合的括号。"""
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')

    missing_braces = open_braces - close_braces
    missing_brackets = open_brackets - close_brackets

    if missing_braces <= 0 and missing_brackets <= 0:
        return text

    # 使用一个栈来精确地确定闭合顺序
    stack = []
    for char in text:
        if char == '{' or char == '[':
            stack.append(char)
        elif char == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif char == ']' and stack and stack[-1] == '[':
            stack.pop()

    closing_sequence = ''
    for bracket in reversed(stack):
        if bracket == '{':
            closing_sequence += '}'
        elif bracket == '[':
            closing_sequence += ']'

    return text + closing_sequence


def _parse_lenient_json(json_string: str) -> dict:
    """
    (最终版 v4) 尽力从一个包含任何已知错误的字符串中解析并修复JSON对象。
    这是迄今为止最健壮的版本，专门为应对LLM的各种“创意”错误而设计。
    """
    if not json_string or not json_string.strip():
        logger.warning("传入的解析内容为空。")
        return {}

    # 1. 预处理
    match = re.search(r'```(?:json)?\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```', json_string, re.DOTALL)
    if match:
        content_to_parse = match.group(1)
    else:
        content_to_parse = json_string

    first_brace = content_to_parse.find('{')
    first_bracket = content_to_parse.find('[')

    if first_brace == -1 and first_bracket == -1:
        raise json.JSONDecodeError("在字符串中未找到JSON起始符号 '{' 或 '['", json_string, 0)

    start_pos = min(first_brace if first_brace != -1 else float('inf'),
                    first_bracket if first_bracket != -1 else float('inf'))

    current_string = content_to_parse[start_pos:]

    # 2. “诊断-修复-重试”循环
    max_repairs = 5
    for attempt in range(max_repairs + 1):
        try:
            return json.loads(current_string)
        except json.JSONDecodeError as e:
            logger.info(f"解析尝试 #{attempt + 1} 失败: {e.msg} at pos {e.pos}. 尝试修复...")

            if attempt >= max_repairs:
                logger.error(f"达到最大修复次数仍无法解析。最终失败的字符串: '{current_string}'")
                raise e

            # 3. 诊断并执行修复
            error_msg = e.msg.lower()
            original_string_for_this_attempt = current_string
            stripped = current_string.rstrip()

            # --- 修复策略工具箱 (按优先级排序) ---

            # 策略A: 修复未闭合的字符串 (最明确的错误)
            if "unterminated string" in error_msg:
                logger.info("诊断: 字符串未闭合。修复: 添加 '\"' 并尝试闭合括号。")
                current_string += '"'
                # 补上引号后，可能就是一个完整的JSON了，所以直接闭合
                current_string = _close_open_brackets(current_string)

            # 策略B: 修复悬空的逗号
            elif "expecting value" in error_msg and stripped.endswith(','):
                logger.info("诊断: 悬空的逗号。修复: 移除末尾 ',' 并尝试闭合括号。")
                current_string = stripped[:-1]
                current_string = _close_open_brackets(current_string)

            # 策略C (终极手段): 暴力回溯 - 丢弃最后一个（可能已损坏的）条目
            else:
                logger.info("诊断: 泛化的语法错误或严重截断。修复: 暴力回溯，丢弃最后一个不完整的条目。")

                # 我们寻找最后一个逗号，因为它标志着上一个完整键值对的结束。
                # 但我们需要确保这个逗号不在一个字符串内部。
                last_comma_pos = -1
                in_string = False
                for i in range(len(current_string) - 1, -1, -1):
                    char = current_string[i]
                    if char == '"' and (i == 0 or current_string[i - 1] != '\\'):
                        in_string = not in_string
                    if not in_string and char == ',':
                        last_comma_pos = i
                        break

                if last_comma_pos != -1:
                    # 截断到最后一个逗号，丢弃后面的所有内容。
                    current_string = current_string[:last_comma_pos]
                    # 强制闭合剩下的、理论上是有效的部分。
                    current_string = _close_open_brackets(current_string)
                else:
                    # 如果连一个合法的逗号都找不到，说明整个JSON结构已完全损坏。
                    # 此时无法安全修复，直接放弃。
                    logger.error("无法进行回溯修复（未找到任何不在字符串内的逗号），放弃。")
                    raise e

            # 安全检查：如果修复操作没有改变任何东西，为避免死循环，直接放弃
            if current_string == original_string_for_this_attempt:
                logger.error("修复操作未能改变字符串，为避免死循环而放弃。")
                raise e
    # 理论上不会执行到这里，但作为保障
    raise json.JSONDecodeError("无法从模型响应中解析出任何有效的JSON片段", json_string, 0)


def simple_clean_birth_info(raw_output: dict) -> dict:
    """
    对LLM的输出进行多轮、健壮的清洗，处理已知的所有常见错误。
    """
    if not isinstance(raw_output, dict):
        return {}

    cleaned = raw_output.copy()

    # 第一轮：修复“字段名作值”的错误
    # 如果一个字段的值和它的名字一样，说明模型出错了，直接设为 None
    for field in ['year', 'month', 'day', 'hour', 'minute', 'gender']:
        if cleaned.get(field) == field:
            cleaned[field] = None

    # 第二轮：修复常见的别名和英文 (大小写不敏感)
    # 性别修复
    gender_map = {
        '女孩': '女', '女性': '女', '女生': '女', 'female': '女',
        '男孩': '男', '男性': '男', '男生': '男', 'male': '男'
    }
    if isinstance(cleaned.get("gender"), str):
        gender_val = cleaned["gender"].strip().lower()
        if gender_val in gender_map:
            cleaned['gender'] = gender_map[gender_val]

    # 月份修复
    month_map = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    if isinstance(cleaned.get("month"), str):
        month_val = cleaned["month"].strip().lower()
        if month_val in month_map:
            cleaned['month'] = month_map[month_val]

    # 第三轮：确保数值字段是整数或None，处理无法解析的字符串
    for field in ['year', 'month', 'day', 'hour', 'minute']:
        value = cleaned.get(field)
        # 如果值是字符串但不是纯数字，则设为 None
        if isinstance(value, str) and not value.isdigit():
            cleaned[field] = None
        # 如果是纯数字字符串，转换为整数
        elif isinstance(value, str) and value.isdigit():
            cleaned[field] = int(value)

    return cleaned


def escape_json_for_prompt(data_dict: Dict[str, Any]) -> str:
    """将字典转换为 JSON 字符串，并转义大括号以便在 LangChain Prompt 中使用。"""
    if not data_dict:
        return "{}"
    json_str = json.dumps(data_dict, ensure_ascii=False, indent=2)
    return json_str.replace('{', '{{').replace('}', '}}')


def parse_chart_description_block(chart_description_block: str) -> Dict[str, str]:
    """
    解析 describe_ziwei_chart 函数生成的宫位描述块，
    并提取每个宫位的详细描述。
    返回一个字典，键为宫位名称，值为对应的描述字符串。
    """
    palace_descriptions = {}
    # print("chart_description_block",chart_description_block)

    # Define the separator for the main palace block
    separator = "现在，让我们逐一看看您其他主要宫位的配置，及其三方四正和夹宫情况："

    parts = chart_description_block.split(separator, 1)  # Split into at most 2 parts

    # Part 1: Main Palace (命宫)
    if len(parts) > 0:
        main_palace_block = parts[0].strip()
        if main_palace_block:
            lines = main_palace_block.split('\n')
            first_line = lines[0].strip()
            # Match "您的命宫坐落于..." or "命宫坐落于..."
            re_match_result = re.match(r'^(您的)?(\w+宫)坐落于(.*)', first_line)
            if re_match_result:
                palace_name = re_match_result.group(2)
                # Extract content starting from "坐落于" or after the palace name and "坐落于"
                desc_content = "坐落于" + re_match_result.group(3).strip()

                # Combine with remaining lines
                full_description = desc_content
                if len(lines) > 1:
                    full_description += "\n" + "\n".join(lines[1:]).strip()
                palace_descriptions[palace_name] = full_description.strip()
            else:
                # Fallback if the first line doesn't match the expected pattern
                if "命宫" in main_palace_block:
                    palace_descriptions["命宫"] = main_palace_block.strip()
                else:
                    print(f"Warning: Could not identify '命宫' in the initial block: {main_palace_block[:100]}...")

    # Part 2: Other Palaces
    if len(parts) > 1:
        other_palaces_block = parts[1].strip()
        lines = other_palaces_block.split('\n')

        current_palace_name = None
        current_palace_desc_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match "- [其他宫位]宫坐落于..."
            re_match_other_palace = re.match(r'^- (\w+宫)坐落于(.*)', line)

            if re_match_other_palace:
                # If there was a previous palace being collected, save it
                if current_palace_name and current_palace_desc_lines:
                    palace_descriptions[current_palace_name] = " ".join(current_palace_desc_lines).strip()

                current_palace_name = re_match_other_palace.group(1)  # Extract palace name
                # Extract content starting from "坐落于" or after the palace name and "坐落于"
                desc_content = "坐落于" + re_match_other_palace.group(2).strip()
                current_palace_desc_lines = [desc_content]  # Start collecting
            else:
                # Continue collecting lines for the current palace
                if current_palace_desc_lines:
                    current_palace_desc_lines.append(line)

        # Save the last collected palace description
        if current_palace_name and current_palace_desc_lines:
            palace_descriptions[current_palace_name] = " ".join(current_palace_desc_lines).strip()

    return palace_descriptions


def calculate_decadal_start_age_exact(wuxingju: str) -> int:
    """根据五行局计算大运开始的精确年龄。"""
    if "水二局" in wuxingju: return 2
    if "木三局" in wuxingju: return 3
    if "金四局" in wuxingju: return 4
    if "土五局" in wuxingju: return 5
    if "火六局" in wuxingju: return 6
    return 6


def get_current_decadal_start_year(birth_year: int, wuxingju: str, current_age: int) -> int:
    """计算包含当前年龄的大运的开始年份。"""
    start_age_first_decadal = calculate_decadal_start_age_exact(wuxingju)
    decadal_start_age = start_age_first_decadal
    while decadal_start_age <= current_age:
        decadal_start_age += 10
    return birth_year + (decadal_start_age - 10)


def calculate_next_decadal_start_year(birth_year: int, wuxingju: str, current_age: int) -> int:
    """计算下一个大运的开始年份。"""
    start_age_first_decadal = calculate_decadal_start_age_exact(wuxingju)
    current_decadal_start_age = start_age_first_decadal
    while current_decadal_start_age <= current_age:
        current_decadal_start_age += 10
    return birth_year + current_decadal_start_age - 1


def get_decadal_ganzhi_by_age(current_age: int, all_decadal_ages: Dict[str, Tuple[int, int]]) -> Optional[str]:
    """
    根据当前年龄从all_decadal_ages字典中查找对应的大运干支。
    返回包含当前年龄的大运的干支，如果找不到则返回None。
    
    参数:
        current_age: 当前年龄（虚岁）
        all_decadal_ages: 所有大运的年龄范围字典，格式如 {'丙寅': (2, 11), '丁卯': (12, 21), ...}
    
    返回:
        大运干支字符串（如 '丙寅'），如果找不到则返回None
    """
    if not all_decadal_ages:
        return None
    
    for ganzhi, (start_age, end_age) in all_decadal_ages.items():
        if start_age <= current_age <= end_age:
            return ganzhi
    
    return None


def extract_decadal_age_range_from_api_response(
    api_response_data: Dict[str, Any],
    decadal_ganzhi: str
) -> Optional[Tuple[int, int]]:
    """
    从API响应中根据大运干支提取年龄范围。
    
    参数:
        api_response_data: API返回的完整响应数据
        decadal_ganzhi: 大运干支（如"丁亥"）
    
    返回:
        年龄范围元组 (start_age, end_age)，如果找不到则返回None
    """
    if not api_response_data or not decadal_ganzhi:
        return None
    
    try:
        astrolabe_data = api_response_data.get('data', {}).get('astrolabe', {})
        palaces = astrolabe_data.get('palaces', [])
        
        if not palaces or len(decadal_ganzhi) < 2:
            return None
        
        # 提取天干和地支
        stem = decadal_ganzhi[0]  # 天干
        branch = decadal_ganzhi[1]  # 地支
        
        # 遍历所有宫位，找到匹配的decadal
        for palace in palaces:
            decadal_info = palace.get('decadal', {})
            if not decadal_info:
                continue
            
            decadal_stem = decadal_info.get('heavenlyStem', '')
            decadal_branch = decadal_info.get('earthlyBranch', '')
            decadal_range = decadal_info.get('range', [])
            
            # 检查是否匹配
            if decadal_stem == stem and decadal_branch == branch:
                if isinstance(decadal_range, list) and len(decadal_range) == 2:
                    return tuple(decadal_range)
        
        return None
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"从API响应中提取年龄范围时出错: {e}")
        return None


def _map_topics_to_palaces(topics: List[str], relationship: Optional[str] = None) -> List[str]:
    """根据主题和关系映射到相关的紫微斗数宫位。"""
    if not topics:
        return []

    relationship_map = {
        "丈夫": "夫妻宫", "老婆": "夫妻宫", "配偶": "夫妻宫", "对象": "夫妻宫", "伴侣": "夫妻宫",
        "孩子": "子女宫", "儿子": "子女宫", "女儿": "子女宫", "子女": "子女宫", "表哥": "兄弟宫",
        "父母": "父母宫", "父亲": "父母宫", "母亲": "父母宫", "长辈": "父母宫", "表弟": "兄弟宫",
        "兄弟": "兄弟宫", "姐妹": "兄弟宫", "手足": "兄弟宫", "表妹": "兄弟宫", "堂妹": "兄弟宫",
        "朋友": "交友宫", "同事": "交友宫", "合伙人": "交友宫", "表姐": "兄弟宫", "堂姐": "兄弟宫",
        "堂弟": "兄弟宫", "堂哥": "兄弟宫", "叔": "父母宫", "舅妈": "父母宫", "舅": "父母宫",
        "弟": "兄弟宫", "姐": "兄弟宫", "妹": "兄弟宫", "哥": "兄弟宫","公司": "田宅宫","上司": "事业宫","下属": "交友宫","团队": "交友宫",
        "猫":"子女宫","狗":"子女宫","宠物":"子女宫","学校":"田宅宫"
    }

    topic_map = {
        "财运": ["财帛宫", "命宫"],
        "赌运": ["财帛宫", "福德宫", "命宫"],
        "投资": ["财帛宫", "福德宫", "命宫"],
        "股票": ["财帛宫", "福德宫", "命宫"],
        "理财": ["财帛宫", "福德宫", "命宫"],
        "事业": ["事业宫", "命宫"],
        "工作": ["事业宫", "命宫"],
        "学业": ["事业宫", "福德宫"],
        "婚姻": ["夫妻宫", "命宫"],
        "感情": ["夫妻宫", "命宫"],
        "桃花": ["夫妻宫", "命宫"],
        "健康": ["疾厄宫", "命宫"],
        "疾病": ["疾厄宫", "命宫"],
        "装修": ["田宅宫", "命宫"],
        "搬家": ["田宅宫", "命宫"],
        "居家": ["田宅宫", "命宫", "福德宫"],
        "家庭": ["田宅宫", "命宫", "福德宫"],
        "田宅": ["田宅宫"],
        "讨要工资": ["命宫", "田宅宫", "财帛宫"],
        "公司": ["田宅宫","事业宫"],
        "同事": ["命宫","交友宫", "事业宫"],
        "上司": ["命宫", "交友宫","事业宫"],
        "下属": ["命宫","交友宫", "事业宫"],
        "团队": ["命宫", "交友宫", "事业宫"],
        "单位": ["命宫", "交友宫", "事业宫"],
        "企业": ["命宫", "事业宫"],

        # --- ↓↓↓ 新增的规则 ↓↓↓ ---
        "出行": ["迁移宫", "命宫"],
        "出门": ["迁移宫", "命宫"],
        "旅游": ["迁移宫", "命宫"],
        "出差": ["迁移宫", "命宫", "事业宫"],
        "迁移": ["迁移宫", "命宫"],
        "娱乐消遣": ["福德宫", "命宫"],
        # --- ↑↑↑ 新增的规则 ↑↑↑ ---

        "人际": ["交友宫", "命宫"],
        "社交": ["交友宫", "命宫"],
        "整体运势": ["命宫", "事业宫", "财帛宫", "夫妻宫"],
        "命盘": ["命宫", "事业宫", "财帛宫", "夫妻宫"],
        "性格": ["命宫"],
        "未知": [""]
    }
    palaces = set()

    if relationship:
        for key, val in relationship_map.items():
            if key in relationship:
                palaces.add(val)
                palaces.add("命宫")
                return list(palaces)

    for topic in topics:
        for keyword, mapped_palaces in topic_map.items():
            if keyword in topic:
                palaces.update(mapped_palaces)

    final_palaces = list(palaces)
    if len(final_palaces) > 3:
        return final_palaces
    if len(final_palaces) == 2 and "命宫" not in final_palaces:
        final_palaces = [final_palaces[0], "命宫"]
    if not final_palaces and topics:
        return ["命宫"]

    return final_palaces if final_palaces else ["命宫", "事业宫", "财帛宫", "夫妻宫"]


def _resolve_horoscope_date(expression: str, current_dt: datetime) -> Dict[str, Any]:
    """
    【V2版 - 增加了对农历日期的解析能力】
    将LLM提取的时间表达式解析为具体的日期和分析级别。
    """
    if not expression:
        return {}

    expression = expression.strip()

    def _chinese_num_to_int(num_str: str) -> int:
        """
        支持简单的中文数字转整数，兼容“半”“几”这种模糊表达（几→3，半→0.5）。
        """
        mapping = {
            "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
            "半": 0.5, "几": 3
        }
        if num_str.isdigit():
            return int(num_str)
        total = 0
        temp = 0
        for ch in num_str:
            if ch == "十":
                temp = max(temp, 1)
                total += temp * 10
                temp = 0
            elif ch in mapping:
                temp += mapping[ch]
        total += temp
        return int(total) if total else 0

    # --- 优先解析“未来X个月/几天/季度”等相对跨度，避免落入年级别 ---
    span_match = re.search(r'(未来|今后|接下来|后|近)\s*([一二三四五六七八九十两几半\d]+)\s*(个)?(月|个月|周|星期|礼拜|天|日|季度|季|半年|半年度)', expression)
    if span_match:
        num_raw = span_match.group(2)
        unit = span_match.group(4)
        count = _chinese_num_to_int(num_raw)
        # 半年/半年度单独处理
        if unit in ("半年", "半年度"):
            count = 6
        # 季度=3个月
        if unit in ("季度", "季"):
            count = count * 3
            unit = "个月"
        if unit in ("周", "星期", "礼拜"):
            days = max(1, int(count * 7))
            result = {
                "relative_time_indicator": expression,
                "analysis_level": "daily"
            }
            dt = current_dt + timedelta(days=days)
        elif unit in ("天", "日"):
            days = max(1, count)
            result = {
                "relative_time_indicator": expression,
                "analysis_level": "daily"
            }
            dt = current_dt + timedelta(days=days)
        else:
            # 月度跨度，统一归类为monthly，先锚定当前月的1号；若已是月末且时间早于当前，自动跳到下月1日
            months = max(1, int(count))
            result = {
                "relative_time_indicator": expression,
                "analysis_level": "monthly",
                "multi_month_span": months
            }
            dt = current_dt.replace(day=1, hour=12, minute=0, second=0, microsecond=0)
            if dt < current_dt:
                # 确保指向未来或当月中线以后也不被历史检测拦截
                next_month = dt.month + 1
                next_year = dt.year + (1 if next_month > 12 else 0)
                next_month = 1 if next_month > 12 else next_month
                dt = dt.replace(year=next_year, month=next_month, day=1)
        result["resolved_horoscope_date"] = dt.strftime('%Y-%m-%d %H:%M:%S')
        result["target_year"] = dt.year
        result["target_month"] = dt.month
        result["target_day"] = dt.day
        result.setdefault("target_hour", 12)
        result.setdefault("target_minute", 0)
        return result

    result = {
        "relative_time_indicator": expression,
        "target_hour": 12,
        "target_minute": 0,
    }
    dt = None

    # --- 步骤 1: 优先处理相对时间 ---
    # (这部分逻辑完全不变)
    if "今天" in expression or "今日" in expression:
        dt = current_dt
        result["analysis_level"] = "daily"
    elif "昨天" in expression:
        dt = current_dt - timedelta(days=1)
        result["analysis_level"] = "daily"
    elif "前天" in expression:
        dt = current_dt - timedelta(days=2)
        result["analysis_level"] = "daily"
    elif "明天" in expression:
        dt = current_dt + timedelta(days=1)
        result["analysis_level"] = "daily"
    elif "后天" in expression:
        dt = current_dt + timedelta(days=2)
        result["analysis_level"] = "daily"
    elif "本月" in expression or "这个月" in expression:
        dt = current_dt
        result["analysis_level"] = "monthly"
    elif "下月" in expression or "下个月" in expression:
        month = current_dt.month + 1
        year = current_dt.year
        if month > 12:
            month = 1
            year += 1
        dt = current_dt.replace(year=year, month=month, day=1)
        result["analysis_level"] = "monthly"
    elif "上月" in expression or "上个月" in expression:
        month = current_dt.month - 1
        year = current_dt.year
        if month > 12:
            month = 1
            year += 1
        dt = current_dt.replace(year=year, month=month, day=1)
        result["analysis_level"] = "monthly"
        # 【修复】优先检测"今年X月"模式，避免被"今年"误匹配为yearly
    elif re.search(r'今年.*?(\d{1,2})月', expression):
        month_match = re.search(r'今年.*?(\d{1,2})月', expression)
        if month_match:
            month = int(month_match.group(1))
            dt = current_dt.replace(month=month, day=1)
            result["analysis_level"] = "monthly"
            logger.info(f"检测到'今年X月'模式: {expression} -> 月份{month}, 设置为monthly级别")
    elif re.search(r'明年.*?(\d{1,2})月', expression):
        month_match = re.search(r'明年.*?(\d{1,2})月', expression)
        if month_match:
            month = int(month_match.group(1))
            dt = current_dt.replace(year=current_dt.year + 1, month=month, day=1)
            result["analysis_level"] = "monthly"
            logger.info(f"检测到'明年X月'模式: {expression} -> 年份{current_dt.year + 1}月份{month}, 设置为monthly级别")
    elif "今年" in expression:
        dt = current_dt.replace(month=12, day=31)
        result["analysis_level"] = "yearly"
    elif "明年" in expression or "下一年" in expression:
        dt = current_dt.replace(year=current_dt.year + 1, month=6, day=1)
        result["analysis_level"] = "yearly"
    elif "后年" in expression:
        dt = current_dt.replace(year=current_dt.year + 2, month=6, day=1)
        result["analysis_level"] = "yearly"
    elif "大运" in expression or "大限" in expression:
        if "当前" in expression or "现在" in expression:
            dt = current_dt.replace(month=12, day=31)
        # "下一个大限" 的 dt 留空，由后续逻辑处理
        result["analysis_level"] = "decadal"

    # --- 步骤 2: 如果不是相对时间，则尝试解析绝对日期 ---
    if dt is None:  # 使用 is None 更严谨
        # is_lunar_query = any(...) # 这行可以被后续逻辑覆盖，所以不是必须的

        # --- 预处理部分 (保持不变) ---
        processed_expression = expression

        num_map = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
                   '十': '10'}
        month_map = {'正': '1', '冬': '11', '腊': '12'}
        day_prefix_map = {'初': '', '廿': '2'}

        for k, v in month_map.items():
            processed_expression = processed_expression.replace(k, v)
        for k, v in day_prefix_map.items():
            processed_expression = processed_expression.replace(k, v)
        for k, v in num_map.items():
            # 处理 "十" 的特殊情况，如 "十一", "二十"
            if k == '十':
                processed_expression = re.sub(r'(?<!\d)十(?=\d)', '1', processed_expression)  # 十一 -> 11
                processed_expression = re.sub(r'(?<![1-9])十(?![1-9])', '10', processed_expression)  # 单独的十 -> 10
            else:
                processed_expression = processed_expression.replace(k, v)

        # 增强版农历正则，现在可以匹配 "农历5月初5", "农历5月5日", "5月5" 等
        m_lunar = re.search(r'(\d{1,2})月.*?(\d{1,2})日?', processed_expression)
        m_lunar_month_only = re.search(r'(\d{1,2})月', processed_expression)

        # 公历正则
        m_solar_full = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日?', expression)
        m_solar_year_month = re.search(r'(\d{4})年(\d{1,2})月', expression)
        m_solar_year = re.search(r'(?<!\d)(\d{4})(?!\d)', expression)
        # m_solar_month_only = re.search(r'^(\d{1,2})月$', expression.strip())
        m_solar_month_only = re.search(r'^(\d{1,2})月(份)?$', expression.strip())
        m_solar_month_day = re.search(r'^(\d{1,2})月(\d{1,2})日?$', expression.strip())

        year = current_dt.year

        # 统一处理年份
        m_year_in_exp = re.search(r'(\d{4})年', expression)
        if m_year_in_exp:
            year = int(m_year_in_exp.group(1))

        # --- 现在开始判断逻辑 ---
        if any(keyword in expression for keyword in ["农历", "阴历"]):

            # year = current_dt.year
            month, day = None, None

            # 检查表达式中是否明确指定了年份，如果有，则覆盖默认值
            # m_year_in_exp = re.search(r'(\d{4})年', expression)
            # if m_year_in_exp:
            #    year = int(m_year_in_exp.group(1))

            # 进入农历处理逻辑
            target_match = m_lunar if m_lunar else m_lunar_month_only
            if target_match:
                groups = target_match.groups()
                if groups[0]: month = int(groups[0])
                if m_lunar and len(groups) > 1 and groups[1]:
                    day = int(groups[1])

                if month:
                    try:
                        day_to_convert = day if day is not None else 1
                        lunar_obj = LunarDate(year, month, day_to_convert)
                        solar_date = lunar_obj.toSolarDate()
                        dt = datetime(solar_date.year, solar_date.month, solar_date.day)
                        result["analysis_level"] = "daily" if day is not None else "monthly"
                    except ValueError:
                        pass

        # 如果不是农历，则按公历逻辑处理
        elif m_solar_full:
            parsed_year, month, day = map(int, m_solar_full.groups())
            dt = datetime(parsed_year, month, day)
            result["analysis_level"] = "daily"


        elif m_solar_month_day:
            month, day = map(int, m_solar_month_day.groups())
            # 使用在上面已经确定好的 year (默认为当年)
            dt = datetime(year, month, day)
            result["analysis_level"] = "daily"

        elif m_solar_year_month:
            parsed_year, month = map(int, m_solar_year_month.groups())
            dt = datetime(parsed_year, month, 1)
            result["analysis_level"] = "monthly"

        # 【新增】: 将新的判断分支放在这里
        elif m_solar_month_only:
            month = int(m_solar_month_only.group(1))
            # 使用在上面已经确定好的 year (默认为当年)
            dt = datetime(year, month, 1)
            result["analysis_level"] = "monthly"

        elif m_solar_year:
            parsed_year = int(m_solar_year.group(1))
            dt = datetime(parsed_year, 12, 31)
            if result.get("analysis_level") != "decadal":
                result["analysis_level"] = "yearly"

    # --- 步骤 3: 解析小时并格式化最终结果 ---
    # (这部分逻辑完全不变)
    m_hour = re.search(r'(\d{1,2})[点时]', expression)
    if m_hour:
        hour = int(m_hour.group(1))
        if "下午" in expression and hour < 12:
            hour += 12
        result["target_hour"] = hour
        if result.get("analysis_level") == "daily":
            result["analysis_level"] = "hourly"

    if dt:
        # final_dt = dt.replace(hour=result["target_hour"], minute=result["target_minute"], second=0, microsecond=0)
        final_dt = dt
        result["resolved_horoscope_date"] = final_dt.strftime('%Y-%m-%d %H:%M:%S')
        result["target_year"] = final_dt.year
        result["target_month"] = final_dt.month
        result["target_day"] = final_dt.day

    return result


import json

ASTRO_KB_PALACE_ADAPTED = {
    "命宫": {
        "天魁/天钺": {
            "summary": "个人综合运势得贵人相助，机遇多。在面对挑战时，总有人伸出援手，或明或暗给予支持。",
            "天魁": "在学业或事业晋升上，易得到长辈、领导的公开提携，考运亨通。",
            "天钺": "人际关系融洽，私下多有异性贵人或隐形助力，处理事务更显圆融。"
        },
        "左辅/右弼": {
            "summary": "个人行为表现稳健，处事灵活。做决策时能得到多方支持，团队协作能力强，有助于克服困难。",
            "左辅": "与同辈朋友关系和睦，遇到困境时能获得他们坚定不移的帮助和建议。",
            "右弼": "能在不经意间得到意想不到的帮助，或通过非常规渠道解决问题，善于应对突发状况。"
        },
        "文昌/文曲": {
            "summary": "个人气质儒雅，才华横溢，思维敏捷。在学习、创作或表达方面有突出表现，容易获得好名声。",
            "文昌": "学习能力强，擅长书面表达和逻辑分析，考试顺利，文章出众。",
            "文曲": "口才出众，富有艺术鉴赏力，擅长在社交场合展现魅力，容易获得异性青睐。"
        },
        "禄存/天马": {
            "summary": "个人积极进取，财富运势旺盛，尤其适合在变动中求财。越是奔波，越能找到赚钱的机遇。",
            "禄存": "通过自身的努力和辛劳，能积累稳定的财富，但过程中也需要付出较多体力或精力。",
            "天马": "适应能力强，勇于尝试新环境，异地求财或从事与流动性相关的行业，财源广进。"
        },
        "火星/铃星": {
            "summary": "个人性格急躁，易冲动，可能遭遇突发性困扰或长期的隐患。行事需谨慎，避免不必要的冲突。",
            "火星": "脾气火爆，容易与人发生口角或冲突，需警惕意外伤害。",
            "铃星": "内心情绪波动大，易烦躁不安，可能长期被某些小问题困扰，影响情绪稳定。"
        },
        "擎羊/陀罗": {
            "summary": "个人性格刚硬，易陷是非，或面临拖延与纠缠。需注意言行，避免直接冲突和慢性消耗。",
            "擎羊": "行为冲动，容易因一时意气招致是非或身体上的损伤。",
            "陀罗": "思虑过多，行动迟缓，容易陷入困境无法自拔，事情进展缓慢，常有反复。"
        },
        "地空/地劫": {
            "summary": "个人思想独特，但易脱离实际，可能遭遇精神或物质上的损失。投资需格外谨慎，避免盲目。",
            "地空": "想法天马行空，不切实际，容易错过实际机会，或在精神层面感到空虚。",
            "地劫": "钱财不易守住，容易有意外的破耗或投资失利，财富来去匆匆，不易积累。"
        },
        "化禄": {"summary": "个人福气好，财运亨通，人际关系和谐，做事顺利，容易获得实质性收获。"},
        "化权": {"summary": "个人有掌控欲和决断力，行动力强，在事业或生活中表现出强大的领导和管理才能。"},
        "化科": {"summary": "个人声誉良好，学识出众，容易得到贵人提携，在公众场合或专业领域获得认可。"},
        "化忌": {"summary": "个人运势多阻碍，易生是非，情绪波动大，需警惕小人或健康问题，但也是自我转化的契机。"},
        "sihua_combo": {"禄权": {"summary": "个人名利双收，事业财富发展势头强劲，具有很强的开创和领导能力。"},
                        "禄科": {"summary": "个人通过名声或专业知识获得财富，声誉和实际利益兼得，容易在业界脱颖而出。"},
                        "权科": {"summary": "个人权威与声望并重，专业能力得到广泛认可，在特定领域拥有较高的话语权。"},
                        "禄忌": {
                            "summary": "个人在获得利益的同时，伴随潜在的损耗或麻烦，可能因财致祸，或表面风光内藏隐忧。"},
                        "权忌": {"summary": "个人行动力强但过程多阻碍，谋生辛苦，可能因滥用权力或判断失误而陷入困境。"},
                        "科忌": {"summary": "个人名誉可能受损，或在文书、考试、签约方面遇到是非，人际评价不佳，需谨言慎行。"}}
    },
    "兄弟宫": {
        "天魁/天钺": {
            "summary": "与兄弟姐妹、亲密朋友或合作伙伴互动良好，易得他们助力，共同发展。",
            "天魁": "兄弟姐妹或合作伙伴中，有年长或有权势者能提供明确的指导和帮助。",
            "天钺": "朋友或合作伙伴私下关系亲近，能提供隐形支持或在关键时刻拉一把。"
        },
        "左辅/右弼": {
            "summary": "与兄弟姐妹、朋友关系融洽，互为助力。团队协作能力强，能共同应对挑战。",
            "左辅": "与同辈朋友关系稳定，能够相互扶持，在团队项目中表现出色。",
            "右弼": "兄弟姐妹或朋友能提供灵活多变的帮助，在不同方面给予支持，共同进步。"
        },
        "文昌/文曲": {
            "summary": "与兄弟姐妹、朋友间沟通顺畅，多有共同的学习或艺术爱好。彼此交流能促进才华发挥。",
            "文昌": "与兄弟姐妹在学业或知识分享上有良好互动，共同进步。",
            "文曲": "与朋友间多有艺术、文化方面的交流，彼此启发，共同享受生活。"
        },
        "禄存/天马": {
            "summary": "兄弟姐妹、朋友或合作伙伴可能在财务上给予支持，或共同开创事业，财富在合作与变动中产生。",
            "禄存": "与兄弟姐妹、朋友共同努力可积累财富，但过程中也需共同承担辛劳。",
            "天马": "与兄弟姐妹、朋友一起外出奔波或异地合作，更容易发现新的财富机会。"
        },
        "火星/铃星": {
            "summary": "与兄弟姐妹、朋友或合作伙伴之间易有突发性冲突或长期隔阂。需避免急躁和猜疑。",
            "火星": "与兄弟姐妹或朋友之间容易因小事爆发争吵，关系紧张。",
            "铃星": "与朋友或合作伙伴之间可能存在长期积压的矛盾，不易察觉但影响深远。"
        },
        "擎羊/陀罗": {
            "summary": "与兄弟姐妹、朋友或合作伙伴间易有纠纷、拖延。需注意金钱往来和言语冲突。",
            "擎羊": "与兄弟姐妹或朋友之间可能因钱财或原则问题产生直接冲突，关系受损。",
            "陀罗": "与朋友或合作伙伴的关系可能陷入僵局，问题难以解决，拖延日久。"
        },
        "地空/地劫": {
            "summary": "与兄弟姐妹、朋友或合作伙伴之间关系可能虚浮，缺乏实质性帮助，或共同投资失利。",
            "地空": "与兄弟姐妹或朋友的思想观念差异大，难以产生共鸣，关系可能流于表面。",
            "地劫": "与朋友共同投资容易亏损，或彼此间难以给予实际支持，感情淡薄。"
        },
        "化禄": {"summary": "与兄弟姐妹、朋友关系和谐，常有聚会，能互相带来好运和财气。"},
        "化权": {"summary": "在兄弟姐妹或朋友中处于领导地位，能有效组织和管理，但也可能过于强势。"},
        "化科": {"summary": "与兄弟姐妹、朋友间互相学习，共同进步，能获得好名声，或得到他们的智力支持。"},
        "化忌": {"summary": "与兄弟姐妹、朋友关系易有矛盾，可能因误解产生隔阂，或被他们拖累，需多加留意。"},
        "sihua_combo": {"禄权": {"summary": "与兄弟姐妹、朋友合作能名利双收，共同开创事业，关系积极向上。"},
                        "禄科": {"summary": "与兄弟姐妹、朋友间能通过知识交流或共同努力，获得名利和实质性回报。"},
                        "权科": {"summary": "与兄弟姐妹、朋友间在专业或学习上互相扶持，共同提升声望和能力。"},
                        "禄忌": {"summary": "与兄弟姐妹、朋友间表面关系好，但可能因利益纠葛或隐藏问题导致关系受损。"},
                        "权忌": {"summary": "与兄弟姐妹、朋友间关系紧张，可能因权力争夺或强势态度导致冲突和麻烦。"},
                        "科忌": {"summary": "与兄弟姐妹、朋友间可能因言语、学业或名声问题产生是非，关系出现裂痕。"}}
    },
    "夫妻宫": {
        "天魁/天钺": {
            "summary": "与伴侣关系融洽，能得到对方的帮助，或伴侣自身是贵人。单身者易遇到条件不错的对象。",
            "天魁": "伴侣可能在事业或学业上给予指导和支持，或伴侣是社会地位较高之人。",
            "天钺": "伴侣在生活细节上体贴入微，或通过伴侣认识重要的异性贵人。"
        },
        "左辅/右弼": {
            "summary": "与伴侣感情稳定，能互相扶持，共同面对生活挑战。关系和谐，不易孤单。",
            "左辅": "伴侣是可靠的支持者，能在关键时刻提供稳健的帮助和建议。",
            "右弼": "伴侣能灵活应对问题，在不同方面给予协助，使得关系更加圆满。"
        },
        "文昌/文曲": {
            "summary": "与伴侣沟通顺畅，情趣相投，多有共同的文化或艺术爱好。感情生活充满浪漫与才情。",
            "文昌": "与伴侣在思想观念上契合，能够共同学习，理性沟通，增进理解。",
            "文曲": "伴侣善于表达爱意，注重情调，共同享受艺术或文学，感情生活丰富多彩。"
        },
        "禄存/天马": {
            "summary": "伴侣能带来财富，或共同为财富奔波。婚姻生活可能因求财而多变动，但财运亨通。",
            "禄存": "伴侣有聚财能力，或通过伴侣关系带来稳定的经济收入，但可能聚少离多。",
            "天马": "与伴侣共同外出发展或从事需要奔波的行业，容易获得意外之财。"
        },
        "火星/铃星": {
            "summary": "与伴侣之间易有突发争执或长期隐忧。感情生活多波动，需警惕言语冲突和内心烦躁。",
            "火星": "与伴侣容易因小事爆发激烈争吵，感情冲动，需避免口舌之争。",
            "铃星": "感情中可能存在长期未解决的矛盾，导致内心压抑和不满，不易发觉但影响深刻。"
        },
        "擎羊/陀罗": {
            "summary": "与伴侣关系易有刑克或拖延。感情中可能面临争执、分离或长期困扰。",
            "擎羊": "与伴侣可能因意见不合而产生直接冲突，甚至肢体争执，感情关系紧张。",
            "陀罗": "感情问题拖延不决，难以化解，可能陷入纠缠不清的状态，关系进展缓慢。"
        },
        "地空/地劫": {
            "summary": "与伴侣之间感情可能缺乏实质性进展，或在财务上遭遇损失。感情理想化，脱离现实。",
            "地空": "对伴侣或感情过于理想化，难以落地，容易感到精神上的空虚或失望。",
            "地劫": "感情中容易有金钱方面的损耗，或因投资失利影响夫妻关系，感情缺乏物质基础。"
        },
        "化禄": {"summary": "与伴侣关系甜蜜，感情和睦，能从对方身上获得福气和财运。"},
        "化权": {"summary": "伴侣性格强势，在关系中占据主导地位，或能带来事业上的助力。"},
        "化科": {"summary": "与伴侣互相尊重，能提升彼此的社会形象和声誉，感情中充满知性与浪漫。"},
        "化忌": {"summary": "与伴侣关系易有矛盾，沟通不畅，可能因对方而劳心破财，或感情不顺，需多加包容。"},
        "sihua_combo": {"禄权": {"summary": "与伴侣共同努力能名利双收，感情和事业双丰收，关系充满活力和掌控力。"},
                        "禄科": {"summary": "与伴侣通过知识或专业领域合作，获得名誉和实际利益，感情中充满共同成长。"},
                        "权科": {"summary": "与伴侣在家庭或事业中拥有共同的权威和声望，能够相互支持，共同管理。"},
                        "禄忌": {"summary": "与伴侣关系表面和谐，实则可能因利益纠葛或隐藏问题，带来感情上的损耗或麻烦。"},
                        "权忌": {"summary": "与伴侣之间易有权力争夺或意见不合，感情中充满挣扎和冲突，需注意沟通方式。"},
                        "科忌": {
                            "summary": "与伴侣可能因文书、合约或名声问题产生纠纷，感情中出现误解或不信任，需谨言慎行。"}}
    },
    "父母宫": {
        "天魁/天钺": {
            "summary": "与父母、长辈、领导关系良好，能得到他们的关照和提携。遇到困难时，他们是重要的贵人。",
            "天魁": "父母或领导在事业、学业上给予公开的指导和帮助，或家族中有权威人士提携。",
            "天钺": "父母或长辈在私下给予实质性支持，或通过他们结识重要的贵人。"
        },
        "左辅/右弼": {
            "summary": "与父母、长辈关系和睦，能得到他们的支持和帮助。家庭氛围稳定，有归属感。",
            "左辅": "父母或长辈是稳健的后盾，能提供可靠的建议和帮助，让你少走弯路。",
            "右弼": "父母或长辈能提供灵活多样的帮助，在不同方面给予支持，让你感受到温暖。"
        },
        "文昌/文曲": {
            "summary": "与父母、长辈沟通良好，多有共同的学习或文化兴趣。能从他们身上学到知识和智慧。",
            "文昌": "父母或长辈注重教育，能提供良好的学习环境，或在学业上给予指导。",
            "文曲": "与父母或长辈在艺术、文化方面有共同语言，能从中获得灵感和启迪。"
        },
        "禄存/天马": {
            "summary": "父母或长辈能带来财富机遇，或自身需为家庭财富奔波。家庭经济状况可能因求财而变动。",
            "禄存": "父母有聚财能力，能为家庭积累财富，或通过他们的帮助获得稳定的经济来源。",
            "天马": "为父母或家庭事务需要外出奔波，或父母长辈能从远方带来财富机遇。"
        },
        "火星/铃星": {
            "summary": "与父母、长辈、领导之间易有突发性冲突或长期隔阂。家庭氛围可能紧张，需避免言语冲撞。",
            "火星": "与父母或领导之间容易因小事爆发争吵，关系紧张，需注意控制情绪。",
            "铃星": "与父母或长辈之间可能存在长期积压的矛盾，不易察觉但影响家庭和睦。"
        },
        "擎羊/陀罗": {
            "summary": "与父母、长辈、领导间易有刑克、拖延。关系可能不睦，或需为他们付出较多。",
            "擎羊": "与父母或领导可能因意见不合而产生直接冲突，关系受损，甚至有刑伤。",
            "陀罗": "与父母或长辈的问题可能拖延不决，难以解决，或需长期照顾他们而劳心费力。"
        },
        "地空/地劫": {
            "summary": "与父母、长辈关系可能疏远，缺乏实质性支持，或在家庭财务上遭遇损失。理想与现实脱节。",
            "地空": "与父母或长辈在思想观念上差异大，难以理解，或对家庭责任感到空虚。",
            "地劫": "家庭容易有意外的财务损失，或与父母共同投资失利，家庭经济基础不稳定。"
        },
        "化禄": {"summary": "与父母、长辈关系和睦，能从他们那里获得福气和帮助，家庭生活幸福。"},
        "化权": {"summary": "父母或领导强势，在家庭或工作中占据主导地位，但也可能带来事业上的助力。"},
        "化科": {"summary": "与父母、长辈互相学习，能提升家庭声誉，或在学业、事业上得到他们的提携。"},
        "化忌": {"summary": "与父母、长辈关系易有矛盾，可能因他们而劳心费力，或自身健康受影响，需多加关心。"},
        "sihua_combo": {"禄权": {"summary": "与父母、长辈共同努力能名利双收，家庭和睦且兴旺，事业发展顺利。"},
                        "禄科": {"summary": "与父母、长辈通过知识交流或共同努力，获得名誉和实际利益，家庭关系充满智慧。"},
                        "权科": {"summary": "与父母、长辈在家庭或事业中拥有共同的权威和声望，能够相互支持，共同管理。"},
                        "禄忌": {
                            "summary": "与父母、长辈关系表面和谐，实则可能因利益纠葛或隐藏问题，带来家庭上的损耗或麻烦。"},
                        "权忌": {
                            "summary": "与父母、长辈之间易有权力争夺或意见不合，家庭中充满挣扎和冲突，需注意沟通方式。"},
                        "科忌": {
                            "summary": "与父母、长辈可能因文书、合约或名声问题产生纠纷，家庭中出现误解或不信任，需谨言慎行。"}}
    },
    "子女宫": {
        "天魁/天钺": {
            "summary": "与子女、晚辈关系良好，他们能带来贵人运，或自身是晚辈的贵人。子女聪明有出息。",
            "天魁": "子女或晚辈学业有成，能得到长辈或老师的提携，或他们自身能带来正面影响力。",
            "天钺": "子女或晚辈在生活中体贴孝顺，或通过他们结识重要的异性贵人。"
        },
        "左辅/右弼": {
            "summary": "与子女、晚辈关系融洽，能得到他们的支持和帮助，或自身能给予他们稳定的辅佐。",
            "左辅": "子女或晚辈是可靠的支持者，能在关键时刻提供帮助和建议，家庭关系和睦。",
            "右弼": "子女或晚辈能灵活应对问题，在不同方面给予协助，使得家庭更加圆满。"
        },
        "文昌/文曲": {
            "summary": "子女、晚辈聪明有才华，学习能力强，多有艺术或文学方面的天赋。",
            "文昌": "子女或晚辈学业优异，擅长书面表达，考试顺利，在学术方面有建树。",
            "文曲": "子女或晚辈富有艺术细胞，口才出众，擅长表演，在艺术领域有突出表现。"
        },
        "禄存/天马": {
            "summary": "子女能带来财富，或为子女的教育、发展而奔波。子女多有远方发展或异地求财的机遇。",
            "禄存": "子女有聚财能力，能为家庭带来稳定的经济收入，或通过他们获得财富。",
            "天马": "为子女的学业或事业需要外出奔波，或子女在异地发展能获得良好财运。"
        },
        "火星/铃星": {
            "summary": "与子女、晚辈之间易有突发性冲突或长期隔阂。需注意沟通方式，避免急躁和猜疑。",
            "火星": "与子女或晚辈容易因小事爆发争吵，关系紧张，需注意控制情绪。",
            "铃星": "与子女或晚辈之间可能存在长期积压的矛盾，不易察觉但影响亲子关系。"
        },
        "擎羊/陀罗": {
            "summary": "与子女、晚辈间易有刑克或拖延。关系可能不睦，或需为他们付出较多心力。",
            "擎羊": "与子女或晚辈可能因意见不合而产生直接冲突，甚至有管教上的困难。",
            "陀罗": "与子女或晚辈的问题可能拖延不决，难以解决，或需长期为他们操心劳力。"
        },
        "地空/地劫": {
            "summary": "与子女、晚辈关系可能疏远，或为他们的教育、发展付出较多但收效甚微，财务上易有损失。",
            "地空": "与子女或晚辈思想观念差异大，难以理解，或在教育子女上感到迷茫。",
            "地劫": "为子女的教育或发展容易有大的财务支出，但收效不佳，或子女易有金钱损耗。"
        },
        "化禄": {"summary": "与子女、晚辈关系亲密，能从他们那里获得福气和快乐，家庭生活温馨。"},
        "化权": {"summary": "子女或晚辈个性强势，在家庭中具有影响力，或能带来事业上的助力。"},
        "化科": {"summary": "与子女、晚辈互相学习，能提升彼此的社会形象和声誉，子女聪明有才。"},
        "化忌": {"summary": "与子女、晚辈关系易有矛盾，可能因他们而劳心费力，或需为他们付出较多，需多加关心。"},
        "sihua_combo": {"禄权": {"summary": "与子女、晚辈共同努力能名利双收，家庭和睦且兴旺，子女事业发展顺利。"},
                        "禄科": {"summary": "与子女、晚辈通过知识交流或共同努力，获得名誉和实际利益，亲子关系充满智慧。"},
                        "权科": {"summary": "与子女、晚辈在家庭或学业中拥有共同的权威和声望，能够相互支持，共同进步。"},
                        "禄忌": {
                            "summary": "与子女、晚辈关系表面和谐，实则可能因利益纠葛或隐藏问题，带来家庭上的损耗或麻烦。"},
                        "权忌": {
                            "summary": "与子女、晚辈之间易有权力争夺或意见不合，亲子关系充满挣扎和冲突，需注意沟通方式。"},
                        "科忌": {
                            "summary": "与子女、晚辈可能因文书、合约或名声问题产生纠纷，亲子关系出现误解或不信任，需谨言慎行。"}}
    },
    "仆役宫": {
        "天魁/天钺": {
            "summary": "与朋友、下属、合作方关系良好，能得到他们的帮助和支持，人际网络广阔。",
            "天魁": "在团队合作或社交场合中，能得到有权势的贵人相助，下属或合作伙伴表现出色。",
            "天钺": "私下能得到朋友或下属的隐形帮助，或通过他们结识重要的异性贵人。"
        },
        "左辅/右弼": {
            "summary": "与朋友、下属关系融洽，能互相扶持，共同应对挑战。团队协作能力强。",
            "左辅": "与同辈朋友或下属关系稳定，能够相互扶持，在团队项目中表现出色。",
            "右弼": "与不同渠道的朋友或下属能提供灵活多变的帮助，在不同方面给予支持。"
        },
        "文昌/文曲": {
            "summary": "与朋友、下属沟通顺畅，多有共同的学习或艺术爱好。能在社交中展现才华，获得认可。",
            "文昌": "与朋友或下属在学业或知识分享上有良好互动，共同进步。",
            "文曲": "与朋友或下属间多有艺术、文化方面的交流，彼此启发，共同享受生活。"
        },
        "禄存/天马": {
            "summary": "朋友、下属或合作方能带来财富机遇，或自身需为社交奔波。财富在合作与变动中产生。",
            "禄存": "与朋友或下属共同努力可积累财富，但过程中也需共同承担辛劳。",
            "天马": "与朋友或下属一起外出奔波或异地合作，更容易发现新的财富机会。"
        },
        "火星/铃星": {
            "summary": "与朋友、下属、合作方之间易有突发性冲突或长期隔阂。需避免急躁和猜疑。",
            "火星": "与朋友或下属之间容易因小事爆发争吵，关系紧张。",
            "铃星": "与朋友或下属之间可能存在长期积压的矛盾，不易察觉但影响合作关系。"
        },
        "擎羊/陀罗": {
            "summary": "与朋友、下属、合作方间易有纠纷、拖延。需注意金钱往来和言语冲突。",
            "擎羊": "与朋友或下属之间可能因钱财或原则问题产生直接冲突，关系受损。",
            "陀罗": "与朋友或下属的关系可能陷入僵局，问题难以解决，拖延日久。"
        },
        "地空/地劫": {
            "summary": "与朋友、下属、合作方关系可能虚浮，缺乏实质性帮助，或共同投资失利。",
            "地空": "与朋友或下属的思想观念差异大，难以产生共鸣，关系可能流于表面。",
            "地劫": "与朋友共同投资容易亏损，或彼此间难以给予实际支持，感情淡薄。"
        },
        "化禄": {"summary": "与朋友、下属关系和谐，常有聚会，能互相带来好运和财气。"},
        "化权": {"summary": "在朋友或下属中处于领导地位，能有效组织和管理，但也可能过于强势。"},
        "化科": {"summary": "与朋友、下属间互相学习，共同进步，能获得好名声，或得到他们的智力支持。"},
        "化忌": {"summary": "与朋友、下属关系易有矛盾，可能因误解产生隔阂，或被他们拖累，需多加留意。"},
        "sihua_combo": {"禄权": {"summary": "与朋友、下属合作能名利双收，共同开创事业，关系积极向上。"},
                        "禄科": {"summary": "与朋友、下属间能通过知识交流或共同努力，获得名利和实质性回报。"},
                        "权科": {"summary": "与朋友、下属间在专业或学习上互相扶持，共同提升声望和能力。"},
                        "禄忌": {"summary": "与朋友、下属间表面关系好，但可能因利益纠葛或隐藏问题导致关系受损。"},
                        "权忌": {"summary": "与朋友、下属间关系紧张，可能因权力争夺或强势态度导致冲突和麻烦。"},
                        "科忌": {"summary": "与朋友、下属间可能因言语、学业或名声问题产生是非，关系出现裂痕。"}}
    },
    "福德宫": {
        "天魁/天钺": {
            "summary": "精神愉悦，心态开朗，多有贵人相助，做事顺利，生活充满机遇。",
            "天魁": "在精神层面感到满足，有长辈或有权势者在思想上给予指引和帮助，内心安定。",
            "天钺": "私下能得到隐形贵人的精神慰藉，内心平静，对待事物积极乐观。"
        },
        "左辅/右弼": {
            "summary": "心态平和，精神状态良好，有稳定的支持力量。内心充满能量，做事有条不紊。",
            "左辅": "内心稳定，有坚实的后盾，面对困难能保持冷静和理性。",
            "右弼": "思维活跃，能灵活应对各种状况，内心充满智慧和变通力。"
        },
        "文昌/文曲": {
            "summary": "精神世界丰富，有艺术鉴赏力，爱好广泛。内心充满才情，追求高雅的生活。",
            "文昌": "思想清晰，逻辑性强，能在学习和思考中获得乐趣，精神层面得到提升。",
            "文曲": "感性丰富，热爱艺术和美，通过创作或欣赏艺术来丰富精神生活。"
        },
        "禄存/天马": {
            "summary": "精神上积极进取，追求财富与成功。心态活跃，乐于奔波，在变动中感受幸福。",
            "禄存": "通过努力和辛劳获得精神满足，虽然辛苦但内心充实。",
            "天马": "喜欢奔波忙碌，在行动中寻找快乐，越是变动越能感受到幸福和成就感。"
        },
        "火星/铃星": {
            "summary": "精神状态不稳定，易烦躁不安，内心多冲突和隐忧。需警惕情绪波动和压力积累。",
            "火星": "脾气急躁，内心容易感到焦虑和不安，冲动行事可能带来悔恨。",
            "铃星": "内心长期压抑，不易察觉但持续性的烦恼，可能导致精神疲劳或失眠。"
        },
        "擎羊/陀罗": {
            "summary": "精神上易有困扰，内心纠结，情绪波动大。需注意心理健康，避免长期压抑。",
            "擎羊": "内心冲动，容易因小事而烦躁，精神上感到痛苦和挣扎。",
            "陀罗": "思虑过多，犹豫不决，内心长期处于纠结状态，难以放松，容易积郁成疾。"
        },
        "地空/地劫": {
            "summary": "精神空虚，思想脱离现实，对事物缺乏兴趣，易感到迷茫或破耗感。需避免过度理想化。",
            "地空": "思想天马行空，不切实际，内心感到空虚和孤独，对世俗事物缺乏兴趣。",
            "地劫": "精神上容易感到无力，对未来感到迷茫，容易在精神或物质上有所损耗。"
        },
        "化禄": {"summary": "精神愉悦，心情开朗，能享受生活中的福气和乐趣，容易感到幸福满足。"},
        "化权": {"summary": "精神状态积极，有掌控欲和决断力，面对挑战能保持强大的内心力量。"},
        "化科": {"summary": "精神世界丰富，追求高雅，能通过学习或艺术提升内心修养，获得好名声。"},
        "化忌": {"summary": "精神压力大，内心烦躁，容易感到不安或困扰，需要关注心理健康和情绪管理。"},
        "sihua_combo": {"禄权": {"summary": "精神状态积极向上，内心充满力量，既能享受生活乐趣又能掌控局面。"},
                        "禄科": {"summary": "精神高雅，追求知识和智慧，能通过学习或艺术提升内心修养，获得福气和名望。"},
                        "权科": {"summary": "精神强大，有决断力，同时又注重修养和品味，内心充满权威和智慧。"},
                        "禄忌": {"summary": "精神上表面愉悦，实则可能伴随烦恼或困扰，容易因享乐过度而带来问题。"},
                        "权忌": {"summary": "精神压力大，内心挣扎，可能因过度掌控或决策失误而感到疲惫和烦躁。"},
                        "科忌": {"summary": "精神上容易因学习、名声或言语而产生困扰，内心焦虑不安，需注意修身养性。"}}
    },
    "田宅宫": {
        "天魁/天钺": {
            "summary": "家庭环境稳定，多有贵人相助，住宅或公司能带来好运。家庭氛围和睦。",
            "天魁": "家庭居住环境良好，或能得到长辈、领导在置业上的帮助，家庭有声望。",
            "天钺": "家中女性成员是贵人，或通过家中关系结识重要人士，家庭内部和睦。"
        },
        "左辅/右弼": {
            "summary": "家庭关系稳定和谐，家人互相支持，房产或公司状况良好，有强大的后盾。",
            "左辅": "家庭成员之间相互扶持，关系稳定，能共同维护家庭和睦。",
            "右弼": "家庭事务能得到灵活处理，家人能提供多方面的帮助，使家庭生活更顺畅。"
        },
        "文昌/文曲": {
            "summary": "家庭环境充满文化气息，家人多有学习或艺术爱好。住宅适宜学习和创作。",
            "文昌": "家庭注重教育，藏书丰富，家人热爱学习，家中多有文书往来。",
            "文曲": "家庭氛围浪漫，充满艺术气息，家人多有艺术天赋或对美有独到见解。"
        },
        "禄存/天马": {
            "summary": "家庭财富积累迅速，或通过房产、置业获得财富。家庭可能因求财而多变动或搬迁。",
            "禄存": "家庭有稳定的财产积累，或能通过房产获得稳定的经济来源。",
            "天马": "家庭可能因工作或求财而经常搬迁，或通过买卖房产获得利润。"
        },
        "火星/铃星": {
            "summary": "家庭内部易有突发性冲突或长期隔阂。需警惕家庭纠纷、家具损坏或公司内部矛盾。",
            "火星": "家庭成员之间容易因小事爆发争吵，家庭氛围紧张，需注意防火安全。",
            "铃星": "家庭内部可能存在长期积压的矛盾，不易察觉但影响家庭和睦或公司运营。"
        },
        "擎羊/陀罗": {
            "summary": "家庭关系易有刑克或拖延。房产可能面临纠纷，或家庭事务进展缓慢。",
            "擎羊": "家庭成员之间可能因财产或意见不合而产生直接冲突，导致家庭不睦。",
            "陀罗": "家庭事务进展缓慢，房产交易可能拖延不决，或长期被家庭问题困扰。"
        },
        "地空/地劫": {
            "summary": "家庭财务可能不稳定，容易有破耗。对居住环境或公司发展有不切实际的期望。",
            "地空": "对家庭或居住环境有不切实际的幻想，容易感到空虚或不满足。",
            "地劫": "家庭财务容易有意外的损失，或房产投资失利，家庭经济基础不稳定。"
        },
        "化禄": {"summary": "家庭生活富足和睦，能从房产或家庭中获得财富和福气，家庭氛围温馨。"},
        "化权": {"summary": "家庭中有强势的成员主导，或能在房产、置业上展现掌控力，家庭兴旺。"},
        "化科": {"summary": "家庭声誉良好，家人有学识修养，能通过房产或家族声望获得社会认可。"},
        "化忌": {"summary": "家庭关系易有矛盾，房产可能出现问题，或因家庭事务劳心费力，需多加留意。"},
        "sihua_combo": {"禄权": {"summary": "家庭财富和地位双丰收，房产增值，家庭成员共同努力，事业发展顺利。"},
                        "禄科": {"summary": "家庭通过知识或声誉获得财富，房产增值，家庭氛围充满智慧和名望。"},
                        "权科": {"summary": "家庭成员在社会上具有权威和声望，能够相互支持，共同管理家庭事务。"},
                        "禄忌": {"summary": "家庭表面富裕，实则可能伴随烦恼或损耗，容易因房产或家庭问题而导致破财。"},
                        "权忌": {"summary": "家庭内部易有权力争夺或意见不合，家庭中充满挣扎和冲突，需注意沟通方式。"},
                        "科忌": {"summary": "家庭可能因文书、合约或名声问题产生纠纷，家庭中出现误解或不信任，需谨言慎行。"}}
    },
    "迁移宫": {
        "天魁/天钺": {
            "summary": "外出遇贵人，交通顺利，外地发展有良好机遇。在外人缘好，易得帮助。",
            "天魁": "外出办事容易遇到有权势的贵人相助，或在外地得到长辈、领导的提携。",
            "天钺": "在外人际关系和谐，私下能得到异性贵人或隐形帮助，出行顺利平安。"
        },
        "左辅/右弼": {
            "summary": "外出顺利，有朋友或同事相伴，能得到多方支持。在外适应能力强，团队协作好。",
            "左辅": "外出时有同伴同行，能互相帮助，或在异地得到朋友的稳健支持。",
            "右弼": "外出时能灵活应对各种突发情况，总能得到意想不到的帮助，使行程顺利。"
        },
        "文昌/文曲": {
            "summary": "外出适宜学习、考察或文化交流。在外能展现才华，获得好名声，旅途充满知性美。",
            "文昌": "外出求学或考察顺利，能在旅途中获得知识，或在外地有文书方面的佳绩。",
            "文曲": "外出旅行充满艺术气息，能欣赏美景，结交志同道合的朋友，口才得到发挥。"
        },
        "禄存/天马": {
            "summary": "外出求财有利，越是奔波越能生财。异地发展机遇多，但过程中需付出较多辛劳。",
            "禄存": "通过外出奔波或异地发展能积累财富，但过程中也需要付出较多努力。",
            "天马": "在外地发展或从事与流动性相关的行业，财运亨通，越动越有利。"
        },
        "火星/铃星": {
            "summary": "外出易有突发性冲突或交通事故。需警惕意外，避免急躁行事，注意人身安全。",
            "火星": "外出时容易因小事爆发争吵，或遭遇突发意外，需注意交通安全。",
            "铃星": "外出旅途可能遭遇长期延误或隐形困扰，不易察觉但影响行程。"
        },
        "擎羊/陀罗": {
            "summary": "外出易有是非、纠纷或拖延。需注意交通安全，避免冲突，行程可能受阻。",
            "擎羊": "外出时可能遭遇直接冲突或意外伤害，需特别注意人身安全和交通状况。",
            "陀罗": "外出行程可能遭遇延误，或在异地陷入纠纷，事情进展缓慢，不易解决。"
        },
        "地空/地劫": {
            "summary": "外出易有财务损失或计划落空。旅途可能感到空虚，或对异地发展抱不切实际的幻想。",
            "地空": "外出旅行或工作可能感到空虚，或计划难以实现，理想与现实有差距。",
            "地劫": "外出时容易有意外的财务损失，或投资失利，旅途花费多但收获少。"
        },
        "化禄": {"summary": "外出顺利，能获得好运和财富，在外地人际关系和谐，有贵人相助。"},
        "化权": {"summary": "外出时能展现领导能力和决断力，在外地掌握主动权，事业发展迅速。"},
        "化科": {"summary": "外出时能提升名声和声誉，得到贵人提携，在外地受到尊重，学业有成。"},
        "化忌": {"summary": "外出易有阻碍，可能遭遇是非或破财，需警惕交通安全和小人，旅途不顺。"},
        "sihua_combo": {"禄权": {"summary": "外出能名利双收，在外地事业和财富双丰收，具有很强的开创和领导能力。"},
                        "禄科": {"summary": "外出通过知识或声誉获得财富，在外地受人尊敬，声誉和实际利益兼得。"},
                        "权科": {"summary": "外出时能展现权威和声望，在外地拥有行业话语权，专业能力得到认可。"},
                        "禄忌": {"summary": "外出看似得利，实则伴随损耗或麻烦，容易因财致祸，或在旅途中遭遇不测。"},
                        "权忌": {"summary": "外出时行动力强但过程多阻碍，可能因滥用权力或判断失误而陷入困境。"},
                        "科忌": {"summary": "外出时可能因文书、合约或名声问题产生纠纷，旅途不顺，需谨言慎行。"}}
    },
    "疾厄宫": {
        "天魁/天钺": {
            "summary": "身体健康，精力充沛，若有疾病易遇良医，恢复快。心态乐观，有贵人扶持。",
            "天魁": "身体强健，不易生病，若有小恙能及时发现并得到有效治疗，或有资深医生帮助。",
            "天钺": "心理健康，情绪稳定，能得到身边人的关心和支持，缓解压力。"
        },
        "左辅/右弼": {
            "summary": "身体状况稳定，抵抗力强，不易疲劳。精神状态良好，有强大的自我修复能力。",
            "左辅": "身体基础好，不易被外界因素影响，日常作息规律，身体稳定。",
            "右弼": "身体适应能力强，能灵活应对各种环境变化，精力充沛，恢复力快。"
        },
        "文昌/文曲": {
            "summary": "身体状态受情绪影响，需注意心理健康。易有小毛病，但通过调养能改善。",
            "文昌": "需注意用脑过度导致的疲劳，或因学习压力引起的身体不适。",
            "文曲": "情绪波动可能影响身体，需注意调理，或易有与艺术、声音相关的疾病。"
        },
        "禄存/天马": {
            "summary": "身体健康状况与劳碌奔波有关。精力充沛但易疲劳，需注意劳逸结合，多做运动。",
            "禄存": "身体健康与日常的辛劳付出成正比，越是努力越能保持健康体魄。",
            "天马": "身体状况受奔波影响大，易疲劳，需注意运动和休息，以保持活力。"
        },
        "火星/铃星": {
            "summary": "身体易有突发性疾病或旧疾复发。需警惕炎症、发热、意外伤害或长期慢性病。",
            "火星": "身体容易出现炎症、发热或突发性疾病，需注意急症和意外伤害。",
            "铃星": "身体可能存在长期隐患，如慢性病，不易察觉但持续困扰，或内心烦躁不安。"
        },
        "擎羊/陀罗": {
            "summary": "身体易有刑伤、手术或慢性病。需警惕意外伤害，注意筋骨、皮肤问题。",
            "擎羊": "身体容易有外伤、手术，或因冲动导致意外伤害，筋骨方面需注意。",
            "陀罗": "身体易有慢性病，康复缓慢，或皮肤、呼吸系统方面有反复发作的问题。"
        },
        "地空/地劫": {
            "summary": "身体健康可能不稳定，易有虚弱感或精神疲劳。需注意调理，避免过度劳累。",
            "地空": "身体感到虚弱，精神空虚，容易失眠，或对健康问题缺乏实际行动。",
            "地劫": "身体容易有意外的损耗，或因投资失利影响情绪，导致身体不适。"
        },
        "化禄": {"summary": "身体健康，精力充沛，不易生病，享受生活带来的舒适感和满足感。"},
        "化权": {"summary": "身体健康状况良好，抵抗力强，能承受较大压力，运动能力强。"},
        "化科": {"summary": "身体状况平稳，注重养生保健，若有不适易遇良医，能保持良好形象。"},
        "化忌": {"summary": "身体易有不适，精神疲劳，需要多加休息和调理，或易有慢性疾病。"},
        "sihua_combo": {"禄权": {"summary": "身体健康，精力充沛，能承担重任，既能享受生活又能保持健康体魄。"},
                        "禄科": {"summary": "身体健康，注重养生，通过科学方法保持良好状态，享受健康带来的福气。"},
                        "权科": {"summary": "身体健康，精力充沛，注重形象，通过自律和锻炼保持强健体魄。"},
                        "禄忌": {"summary": "身体表面健康，实则可能伴随某些隐患，容易因享乐过度或不节制而引发问题。"},
                        "权忌": {"summary": "身体健康状况受情绪影响大，可能因过度劳累或压力过大导致疾病，需注意休息。"},
                        "科忌": {"summary": "身体容易因情绪、精神压力或饮食不当引发不适，需注意调理和休养。"}}
    },
    "官禄宫": {
        "天魁/天钺": {
            "summary": "事业发展顺利，易得贵人提携，有晋升机会。学业有成，考运亨通。",
            "天魁": "在事业上能得到上级、长辈的公开提拔，或在学业上获得好的成绩。",
            "天钺": "在职场中能得到同事、下属的暗中帮助，或通过人脉获得好的工作机会。"
        },
        "左辅/右弼": {
            "summary": "事业发展稳健，有得力助手，团队协作能力强。学业进步，能得到同学帮助。",
            "左辅": "在工作中能得到同辈同事的稳定支持，共同完成项目，业绩提升。",
            "右弼": "在事业上能得到晚辈或不同渠道的灵活帮助，解决突发问题，工作顺利。"
        },
        "文昌/文曲": {
            "summary": "事业或学业表现出色，有才华，利于文书工作、创作。考试顺利，名声显赫。",
            "文昌": "学业优异，擅长文书处理和报告撰写，在考试中能取得好成绩。",
            "文曲": "在工作中展现出艺术才华或出众的口才，在行业内获得好名声。"
        },
        "禄存/天马": {
            "summary": "事业发展迅速，求财有利，尤其适合在变动中求发展。越是奔波，事业越能兴旺。",
            "禄存": "在事业上能通过努力和辛劳积累财富，职业发展稳定。",
            "天马": "事业发展需要经常外出奔波或异地发展，在变动中能抓住更多机遇，成就一番事业。"
        },
        "火星/铃星": {
            "summary": "事业发展易有突发性困扰或长期隐患。需警惕职场冲突、急躁决策或暗中阻碍。",
            "火星": "工作中容易因急躁或冲动而出现失误，或与同事上级发生冲突。",
            "铃星": "事业发展可能存在长期未解决的问题，不易察觉但影响深远，或内心烦躁不安。"
        },
        "擎羊/陀罗": {
            "summary": "事业发展易有是非、纠纷或拖延。需注意职场人际，避免冲突，项目可能受阻。",
            "擎羊": "工作中容易与同事或上级发生直接冲突，或因决策失误导致事业受损。",
            "陀罗": "事业项目进展缓慢，可能陷入僵局，或因外部因素导致长期拖延，难以推进。"
        },
        "地空/地劫": {
            "summary": "事业发展可能面临虚耗或不切实际的计划。财务上易有损失，或对职业前景感到迷茫。",
            "地空": "在事业上想法过于理想化，不切实际，容易错过实际机会，或感到空虚。",
            "地劫": "在事业投资或创业中容易遭遇损失，或工作付出多但收获少，对前景感到迷茫。"
        },
        "化禄": {"summary": "事业发展顺利，能带来丰厚财富，职场人际关系和谐，容易获得晋升机会。"},
        "化权": {"summary": "事业发展迅速，具有掌控力和决断力，在工作中表现出强大的领导和管理才能。"},
        "化科": {"summary": "事业名声显赫，学业有成，容易得到贵人提携，在行业内获得认可和声望。"},
        "化忌": {"summary": "事业发展多阻碍，易生是非，工作压力大，需警惕小人或合同纠纷，但也是转化的契机。"},
        "sihua_combo": {"禄权": {"summary": "事业名利双收，发展势头强劲，具有很强的开创和领导能力，能够掌握主动权。"},
                        "禄科": {"summary": "事业通过专业知识或良好声誉获得财富，在行业内享有盛名，实际利益和名望兼得。"},
                        "权科": {"summary": "事业中权威与声望并重，专业能力得到广泛认可，在特定领域拥有较高的话语权。"},
                        "禄忌": {"summary": "事业表面看似得利，实则伴随巨大损耗或麻烦，容易因财致祸，或在工作中埋下隐患。"},
                        "权忌": {"summary": "事业发展过程多阻碍，可能因滥用权力或判断失误而陷入困境，工作辛苦挣扎。"},
                        "科忌": {"summary": "事业中可能因文书、合约或名声问题产生是非，职场人际关系不佳，需谨言慎行。"}}
    },
    "财帛宫": {
        "天魁/天钺": {
            "summary": "财运亨通，多有贵人相助，赚钱机遇多。投资理财顺利，能获得意外之财。",
            "天魁": "在求财路上能得到长辈、领导的公开提携，或通过社会关系获得财富。",
            "天钺": "私下能得到异性贵人或隐形帮助，通过人际关系获得财富，偏财运佳。"
        },
        "左辅/右弼": {
            "summary": "财运稳定，有多个收入来源或得朋友助力。理财能力强，能有效积累财富。",
            "左辅": "能通过稳定的工作或合作获得财富，朋友或同事能在财务上提供支持。",
            "右弼": "财源广阔，能通过多渠道获得收入，或在关键时刻得到意想不到的财务帮助。"
        },
        "文昌/文曲": {
            "summary": "通过知识、才华或文书工作求财有利。财运与名声相关，适合文化、教育、艺术领域。",
            "文昌": "通过写作、咨询、教育等智力型工作获得收入，或有文书合约带来的财富。",
            "文曲": "通过艺术、表演、口才或人际交往获得财富，财运与个人魅力相关。"
        },
        "禄存/天马": {
            "summary": "求财积极，越是奔波越能生财。财运在变动和远方，投资有利，但需注意风险。",
            "禄存": "通过自身的努力和辛劳，能积累稳定的财富，但也需要不断求索和付出。",
            "天马": "适合在外地发展或从事与流动性相关的行业，财运亨通，越动越有利。"
        },
        "火星/铃星": {
            "summary": "财运波动大，易有突发性破财或长期暗耗。需警惕投资风险，避免冲动消费。",
            "火星": "容易因冲动消费或意外事件导致破财，投资需谨慎，避免急功近利。",
            "铃星": "财运可能存在长期暗耗，不易察觉但持续消耗财富，或因内心烦躁影响理财判断。"
        },
        "擎羊/陀罗": {
            "summary": "财运易有纠纷、拖延或损耗。需注意金钱往来，避免借贷纠纷或投资套牢。",
            "擎羊": "容易因金钱问题与人发生冲突，或有意外的破财，需警惕财务诈骗。",
            "陀罗": "财务问题拖延不决，资金周转缓慢，或投资被套牢，难以快速回笼。"
        },
        "地空/地劫": {
            "summary": "财运不稳定，容易财来财去，或投资失利。对金钱观念不切实际，需谨慎理财。",
            "地空": "对金钱概念模糊，容易盲目投资或消费，导致财富空耗，精神上缺乏安全感。",
            "地劫": "容易有意外的财务损失，或投资付出多但回报少，财富难以积累。"
        },
        "化禄": {"summary": "财运亨通，收入增加，容易获得意外之财，花钱也大方，但能带来福气。"},
        "化权": {"summary": "财运旺盛，有掌控财富的能力和决断力，能在投资或理财上取得成功。"},
        "化科": {"summary": "财运与名声相关，通过专业知识或良好声誉获得财富，收入稳定且有保障。"},
        "化忌": {"summary": "财运不佳，容易破财或资金周转困难，花钱劳心，需警惕财务陷阱。"},
        "sihua_combo": {"禄权": {"summary": "财运名利双收，收入丰厚且有掌控力，在财务上具有很强的开创和管理能力。"},
                        "禄科": {"summary": "财运与名声兼得，通过专业知识或良好声誉获得财富，收入稳定且有保障。"},
                        "权科": {"summary": "财运与权威声望并重，在财务领域有独到见解，能够掌控大局。"},
                        "禄忌": {"summary": "财运表面看似有利，实则伴随巨大损耗或麻烦，容易因财致祸，需谨慎理财。"},
                        "权忌": {"summary": "财运不稳定，可能因投资失误或决策不当导致损失，求财过程辛苦挣扎。"},
                        "科忌": {"summary": "财运可能因文书、合约或名声问题产生纠纷，影响收入，需谨言慎行。"}}
    }
}

STAR_PAIRS = {
    frozenset({'天魁', '天钺'}): True,
    frozenset({'左辅', '右弼'}): True,
    frozenset({'禄存', '天马'}): True,
    frozenset({'文昌', '文曲'}): True,
    frozenset({'火星', '铃星'}): False,
    frozenset({'陀罗', '擎羊'}): False,
    frozenset({'地空', '地劫'}): False
}

# 定义单个星曜的吉凶性质
GOOD_STARS = {'天魁', '天钺', '左辅', '右弼', '禄存', '天马', '文昌', '文曲'}
BAD_STARS = {'火星', '铃星', '陀罗', '擎羊', '地空', '地劫'}

# 定义四化类型吉凶
GOOD_SI_HUA_TYPES = {'化禄', '化权', '化科'}
BAD_SI_HUA_TYPES = {'化忌'}

# 地支顺序和对冲关系
DI_ZHI_ORDER = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']
OPPOSITE_DI_ZHI = {
    '子': '午', '丑': '未', '寅': '申', '卯': '酉', '辰': '戌', '巳': '亥',
    '午': '子', '未': '丑', '申': '寅', '酉': '卯', '戌': '辰', '亥': '巳'
}

ASTRO_KB_GENERIC = {
    # 吉星组合
    "pairs": {
        "天魁/天钺": {
            "name": "贵人星组合", "is_good": True,
            "summary": "形成“坐贵向贵”格局，象征公开与暗中的双重助力，一生贵人不断，事业易得提携与升迁。",
            "天魁": "主男性贵人、长辈或上级的公开助力，利于科甲功名。",
            "天钺": "主女性贵人或私下人脉的暗中帮扶，增强人际亲和力。"
        },
        "左辅/右弼": {
            "name": "辅佐星组合", "is_good": True,
            "summary": "形成强大的辅佐力量，象征稳健与灵活的助力兼备，能有效增强核心决策力与团队协作。",
            "左辅": "主同辈、同事的稳健支持，善于协调团队关系。",
            "右弼": "主晚辈或不同渠道的灵活助力，善于谋略与危机化解。"
        },
        "文昌/文曲": {
            "name": "科甲星组合", "is_good": True,
            "summary": "形成“文星拱命”之势，象征理性与感性的才华并发，极利于学术、文艺、创作与名声传播。",
            "文昌": "主理性思维、文书与考试功名，气质清雅端正。",
            "文曲": "主感性才艺、口才与艺术表现，气质风流善辩。"
        },
        "禄存/天马": {
            "name": "禄马交驰格", "is_good": True,
            "summary": "形成“禄马交驰”的暴富格局，象征动态求财能力极强，财富机遇在奔波、变动与远方。",
            "禄存": "主稳定财富的积累，但也需奔波劳碌。",
            "天马": "主动态机遇与财富流动，越迁移越能生财。"
        },
    },
    # 煞星组合
    "dark_pairs": {
        "火星/铃星": {
            "name": "火铃夹命/同宫", "is_good": False,
            "summary": "形成“火铃”的冲击格局，象征突发性与持续性的困扰交织，易有急躁的决策和暗中的波折。",
            "火星": "主突发性灾祸、冲突，破坏猛烈但去速快。",
            "铃星": "主持续性困扰、暗藏的隐患，不易察觉。"
        },
        "擎羊/陀罗": {
            "name": "羊陀夹忌/同宫", "is_good": False,
            "summary": "形成“羊陀”的刑忌组合，象征公开的伤害与暗中的拖延并存，易陷入纠纷与困局。",
            "擎羊": "主直接的伤害、手术、冲突，性格冲动刚硬。",
            "陀罗": "主拖延纠缠、慢性问题，行动滞涩多顾虑。"
        },
        "地空/地劫": {
            "name": "空劫夹命/同宫", "is_good": False,
            "summary": "形成“空劫”的虚耗格局，象征精神与物质的双重破耗，易有理想脱离现实、财来财去的现象。",
            "地空": "主精神空虚、理想主义，易错失机遇。",
            "地劫": "主物质破耗、意外损耗，投资需谨慎。"
        },
    },
    # 四化
    "sihua": {
        "化禄": {"name": "福禄之星", "summary": "主福气、财运与机遇，所在宫位易得实质性收获，人际关系和谐。"},
        "化权": {"name": "权威之星", "summary": "主权势、执行力与竞争力，赋予掌控力与决断力，利于开创与管理。"},
        "化科": {"name": "文贵之星", "summary": "主名声、学识与贵人缘，提升社会形象与专业声誉，利于科考与名望。"},
        "化忌": {"name": "困扰之星",
                 "summary": "主阻碍、是非与损耗，是需要重点关注和谨慎应对的领域，但也可能带来深刻的转化。"},
    },
    # 四化组合
    "sihua_combo": {
        "禄权": {"name": "禄权交会", "summary": "形成“名利双收”的强势格局，事业与财富两得意，发展势头强劲。"},
        "禄科": {"name": "禄科交会", "is_good": True,
                 "summary": "形成“名誉与实质利益兼得”的吉象，利于通过专业知识或良好声誉获得财富。"},
        "权科": {"name": "权科交会", "is_good": True,
                 "summary": "形成“权威与声望并重”的格局，专业能力易获认可，拥有行业话语权。"},
        "禄忌": {"name": "禄忌同宫/交会", "is_good": False,
                 "summary": "形成“吉中藏凶”的复杂局面，表面看似得利，实则伴随巨大损耗或麻烦，易因财致祸。"},
        "权忌": {"name": "权忌交战", "is_good": False,
                 "summary": "形成“辛苦谋生”的挣扎之象，行动力强但过程多阻碍，易滥用权力或判断失误。"},
        "科忌": {"name": "科忌交战", "is_good": False,
                 "summary": "形成“文书是非”或“名誉受损”的困扰，易因签约、考试、或人际评价不佳而产生麻烦。"}
    }
}

# (在原有的常量定义下方，添加以下内容)

# --- 知识库：定义性质、激烈程度和描述模板 ---

# 1. 定义不同分值对应的激烈程度
SCORE_INTENSITY_MAP = {
    2.0: "性质正面强烈，影响迅速且明显",
    -2.0: "性质负面且强烈，冲击力大且事发突然",
    1.5: "性质正面且显著，有实质性影响",
    -1.5: "性质负面且显著，有实质性阻碍",
    1.0: "性质正面，有可见的助力",
    -1.0: "性质负面，有可见的困扰",
}

da_xian_raw_data = [['太阳, 巨门', '寅', '福德宫', '', '文昌，天马', ''],
                    ['廉贞, 破军', '卯', '田宅宫', '', '天魁', '红鸾'],
                    ['天机, 天梁', '辰', '官禄宫', '', '', ''],
                    ['天府', '巳', '仆役宫', '', '天钺', ''],
                    ['天同, 太阴', '午', '迁移宫', '', '', ''],
                    ['武曲, 贪狼', '未', '疾厄宫', '武曲化忌，左辅化科', '', ''],
                    ['太阳, 巨门', '申', '财帛宫', '', '', ''],
                    ['天相', '酉', '子女宫', '', '', '天喜'],
                    ['天机, 天梁', '戌', '夫妻宫', '天梁化禄', '陀罗', ''],
                    ['紫微, 七杀', '亥', '兄弟宫', '紫微化权', '禄存', ''],
                    ['天同, 太阴', '子', '命宫', '', '擎羊，文曲', ''],
                    ['武曲, 贪狼', '丑', '父母宫', '武曲化忌，左辅化科', '', '']]
liu_nian_raw_data = [['太阳, 巨门', '寅', '子女宫', '', '陀罗', ''],
                     ['廉贞, 破军', '卯', '夫妻宫', '', '禄存', ''],
                     ['天机, 天梁', '辰', '兄弟宫', '天机化禄,天梁化权', '擎羊', '天喜'],
                     ['天府', '巳', '命宫', '', '', '年解'],
                     ['天同, 太阴', '午', '父母宫', '太阴化忌', '文昌', ''],
                     ['武曲, 贪狼', '未', '福德宫', '', '', ''],
                     ['太阳, 巨门', '申', '田宅宫', '', '天钺, 文曲', ''],
                     ['天相', '酉', '官禄宫', '', '', ''],
                     ['天机, 天梁', '戌', '仆役宫', '天机化禄,天梁化权', '', '红鸾'],
                     ['紫微, 七杀', '亥', '迁移宫', '紫微化科', '天马', ''],
                     ['天同, 太阴', '子', '疾厄宫', '太阴化忌', '天魁', ''],
                     ['武曲, 贪狼', '丑', '财帛宫', '', '', '']]

# 3. 定义不同宫位互动的场景描述
CONTEXT_DESCRIPTIONS = {
    "self_vs_opposite": "自身状态与外部环境/人际关系产生互动",
    "self_vs_trine_A": "自身状态与三方支撑A产生互动",
    "self_vs_trine_B": "自身状态与三方支撑B产生互动",
    "trine_A_vs_trine_B": "两大三方支撑之间相互影响",
    "trine_vs_opposite": "三方支撑力与外部环境产生互动"
}

INTERACTION_INTENSITY_DESC = {
    "ln_vs_dx": "此性质的发生往往非常剧烈迅猛，是当年运势的核心引爆点。",
    "dx_internal": "此性质属于大限十年运势的内部结构，影响深远但发生过程相对平缓。",
    "ln_internal": "此性质属于本流年盘的内部结构，影响直接但发生过程相对平缓。"
}


def get_san_fang_si_zheng_di_zhi(current_di_zhi):
    current_idx = DI_ZHI_ORDER.index(current_di_zhi)
    opposite_di_zhi = OPPOSITE_DI_ZHI[current_di_zhi]
    trine_A_di_zhi_idx = (current_idx + 4) % 12
    trine_A_di_zhi = DI_ZHI_ORDER[trine_A_di_zhi_idx]
    trine_B_di_zhi_idx = (current_idx + 8) % 12
    trine_B_di_zhi = DI_ZHI_ORDER[trine_B_di_zhi_idx]
    return {
        'opposite_di_zhi': opposite_di_zhi,
        'trine_A_di_zhi': trine_A_di_zhi,
        'trine_B_di_zhi': trine_B_di_zhi
    }


def record_event(reason, score, event_type, context, events_list, **kwargs):
    event = {
        "reason": reason,
        "score": score,
        "type": event_type,
        "context": context,
        "intensity": SCORE_INTENSITY_MAP.get(score, "性质一般"),
    }
    event.update(kwargs)
    events_list.append(event)


def generate_qualitative_summary(palace_name, final_score, events):
    if not events:
        return {"overall_evaluation": f"{palace_name}性质平稳...", "detailed_analysis": ["..."],
                "scoring_reasons": "..."}

    source_map = {
        "ln_vs_dx": {"stars": set(), "sihua": set(), "desc": "（大限与流年交会，性质发生剧烈迅猛）"},
        "dx_internal": {"stars": set(), "sihua": set(), "desc": "（大限盘内部结构，性质发生相对平缓）"},
        "ln_internal": {"stars": set(), "sihua": set(), "desc": "（流年盘内部结构，性质发生相对平缓）"}
    }
    for event in events:
        level, context = event.get("intensity_desc", ""), event.get("context", "")
        source_key = None
        if "剧烈迅猛" in level:
            source_key = "ln_vs_dx"
        elif "大限" in context:
            source_key = "dx_internal"
        elif "流年" in context:
            source_key = "ln_internal"

        if source_key:
            if event.get("pair"): source_map[source_key]["stars"].update(event["pair"])
            if event.get("star"): source_map[source_key]["stars"].add(event["star"])
            if event.get("sihua_pair"):
                s1_type = get_si_hua_type(event["sihua_pair"][0])
                s2_type = get_si_hua_type(event["sihua_pair"][1])
                if s1_type: source_map[source_key]["sihua"].add(s1_type)
                if s2_type: source_map[source_key]["sihua"].add(s2_type)
            if event.get("sihua_members"):
                source_map[source_key]["sihua"].update(event["sihua_members"])
            source = event.get("source")
            if source:
                if source in GOOD_STARS or source in BAD_STARS: source_map[source_key]["stars"].add(source)
                s_type = get_si_hua_type(source)
                if s_type: source_map[source_key]["sihua"].add(s_type)

    overall_evaluation_components, covered_stars, covered_sihua = [], set(), set()
    palace_kb = ASTRO_KB_PALACE_ADAPTED.get(palace_name, ASTRO_KB_GENERIC)

    for source_key in ["ln_vs_dx", "dx_internal", "ln_internal"]:
        stars = source_map[source_key]["stars"]
        sihua_types = source_map[source_key]["sihua"]
        desc_suffix = source_map[source_key]["desc"]
        if not stars and not sihua_types: continue

        sihua_combos_kb = palace_kb.get("sihua_combo", {})
        combos_to_check = [
            ({"化禄", "化权", "化科"}, "三奇加会(禄权科)"),
            ({"化禄", "化权"}, "禄权交会"), ({"化禄", "化科"}, "禄科交会"), ({"化权", "化科"}, "权科交会"),
            ({"化禄", "化忌"}, "禄忌交冲"), ({"化权", "化忌"}, "权忌交冲"), ({"化科", "化忌"}, "科忌交冲")
        ]
        for members, combo_key in combos_to_check:
            if members.issubset(sihua_types) and not members.issubset(covered_sihua):
                combo_name_from_kb = next(
                    (info.get('name', combo_key) for ck, info in sihua_combos_kb.items() if ck in combo_key), combo_key)
                summary_from_kb = next((info.get('summary') for ck, info in sihua_combos_kb.items() if
                                        ck in combo_key and info.get('summary')), None)
                if summary_from_kb:
                    overall_evaluation_components.append(f"【{combo_name_from_kb}】：{summary_from_kb} {desc_suffix}")
                    covered_sihua.update(members)

        for combo_key, combo_info in palace_kb.items():
            if "/" in combo_key:
                s1, s2 = combo_key.split('/')
                if s1 in stars and s2 in stars and not {s1, s2}.issubset(covered_stars):
                    if combo_info.get("summary"):
                        overall_evaluation_components.append(f"【{s1}/{s2}组合】：{combo_info['summary']} {desc_suffix}")
                        covered_stars.update({s1, s2})

    for source_key in ["ln_vs_dx", "dx_internal", "ln_internal"]:
        stars = source_map[source_key]["stars"]
        sihua_types = source_map[source_key]["sihua"]
        desc_suffix = source_map[source_key]["desc"]
        for sihua in sihua_types:
            if sihua not in covered_sihua and palace_kb.get(sihua):
                overall_evaluation_components.append(f"【{sihua}】：{palace_kb[sihua]['summary']} {desc_suffix}")
                covered_sihua.add(sihua)
        for star in stars:
            if star not in covered_stars:
                desc_found = False
                for combo_key, combo_info in palace_kb.items():
                    if "/" in combo_key and star in combo_key and combo_info.get(star):
                        overall_evaluation_components.append(f"【{star}单星】：{combo_info[star]} {desc_suffix}");
                        desc_found = True;
                        break
                if not desc_found and palace_kb.get(star) and palace_kb[star].get("summary"):
                    overall_evaluation_components.append(f"【{star}单星】：{palace_kb[star]['summary']} {desc_suffix}")
                covered_stars.add(star)

    unique_evaluations = list(dict.fromkeys(overall_evaluation_components))
    if unique_evaluations:
        overall_evaluation_str = f"{palace_name}的核心性质由以下几点构成：\n" + "\n".join(
            [f"- {e}" for e in unique_evaluations])
    else:
        overall_evaluation_str = f"{palace_name}在该时限内性质表现平稳，无激烈吉凶组合引动。"

    analysis_details = []
    sorted_events = sorted(events, key=lambda x: abs(x['score']), reverse=True)
    for event in sorted_events:
        if event.get("type") in ["single_good", "single_bad"]:
            source = event.get("source", "")
            if source and ((source in covered_stars) or (get_si_hua_type(source) in covered_sihua)): continue
        desc = ""
        event_type = event.get("type", "");
        loc1 = event.get('location1', '宫位1');
        loc2 = event.get('location2', '宫位2')
        if event_type == "sihua_good_combo":
            desc = f"在 {event.get('context', '特定组合')} 中, 发现吉化组合 “{event.get('combo_name', '吉化组合')}”."
        elif event_type == "sihua_overlap":
            desc = f"'{event.get('sihua_pair', ('未知', '未知'))[0]}'({loc1})与'{event.get('sihua_pair', ('未知', '未知'))[1]}'({loc2})发生重叠，强化了影响力。"
        elif event_type == "good_pair" or event_type == "bad_pair":
            desc = f"星曜 “{'-'.join(sorted(list(event.get('pair', []))))}” 分别在 {loc1} 与 {loc2}，形成{'吉星对' if event_type == 'good_pair' else '煞星对'}结构。"
        elif event_type == "star_overlap":
            desc = f"星曜 “{event.get('star', '')}” 同时出现在 {loc1} 与 {loc2}，增强了其在该场景下的作用。"
        elif event_type == "single_good" or event_type == "single_bad":
            desc = f"本宫因 “{event.get('source', '')}” 的存在，获得了基础的{'吉' if event_type == 'single_good' else '凶'}性力量。"
        if desc: desc += f" [评分强度: {event.get('intensity', '一般')}; 事件性质: {event.get('intensity_desc', '未知')}]"; analysis_details.append(
            desc)
    reasons_str = "; ".join([f"{e['reason']} (得分: {e['score']})" for e in sorted_events]) + "."
    return {"overall_evaluation": overall_evaluation_str, "detailed_analysis": analysis_details,
            "scoring_reasons": reasons_str}


def get_palace_info_by_di_zhi(palace_list_parsed, di_zhi):
    for palace_info in palace_list_parsed:
        if palace_info['di_zhi'] == di_zhi: return palace_info
    return None


def get_si_hua_type(si_hua_str):
    if not isinstance(si_hua_str, str): return None
    if '化禄' in si_hua_str: return '化禄'
    if '化权' in si_hua_str: return '化权'
    if '化科' in si_hua_str: return '化科'
    if '化忌' in si_hua_str: return '化忌'
    return None


# ==============================================================================
# 最终版评分函数 (V25 - 增加交叉盘计分来源)
# ==============================================================================
def calculate_score_by_rules(current_liu_nian_palace, liu_nian_parsed, da_xian_parsed):
    """
    (V25 - 增加交叉盘计分来源)
    - 核心修正: 在计分来源白名单中增加“交叉_大限本宫_vs_流年三方A/B”两种组合。
    - 维持V24的所有计分逻辑和数据解析修正。
    """
    score, events = 0.0, []
    current_di_zhi, current_palace_name = current_liu_nian_palace['di_zhi'], current_liu_nian_palace['palace_name']

    # --- 1. 数据准备 ---
    P = {"ln_本宫": current_liu_nian_palace, "ln_对宫": get_palace_info_by_di_zhi(liu_nian_parsed,
                                                                                  get_san_fang_si_zheng_di_zhi(
                                                                                      current_di_zhi)[
                                                                                      'opposite_di_zhi']),
         "ln_三方A": get_palace_info_by_di_zhi(liu_nian_parsed,
                                               get_san_fang_si_zheng_di_zhi(current_di_zhi)['trine_A_di_zhi']),
         "ln_三方B": get_palace_info_by_di_zhi(liu_nian_parsed,
                                               get_san_fang_si_zheng_di_zhi(current_di_zhi)['trine_B_di_zhi']),
         "dx_本宫": get_palace_info_by_di_zhi(da_xian_parsed, current_di_zhi)}
    if P["dx_本宫"]:
        dx_sfsz = get_san_fang_si_zheng_di_zhi(P["dx_本宫"]['di_zhi'])
        P["dx_对宫"] = get_palace_info_by_di_zhi(da_xian_parsed, dx_sfsz['opposite_di_zhi'])
        P["dx_三方A"] = get_palace_info_by_di_zhi(da_xian_parsed, dx_sfsz['trine_A_di_zhi'])
        P["dx_三方B"] = get_palace_info_by_di_zhi(da_xian_parsed, dx_sfsz['trine_B_di_zhi'])
    else:
        P["dx_对宫"], P["dx_三方A"], P["dx_三方B"] = None, None, None

    def get_name(k):
        p_info = P.get(k);
        if not p_info: return k
        return f"{p_info['palace_name']}({p_info['di_zhi']},{k.split('_')[0]})"

    # V25 核心修正: 增加新的计分来源
    source_scores = {cat: 0.0 for cat in
                     ["流年_本宫_vs_对宫", "流年_本宫_vs_三方A", "流年_本宫_vs_三方B", "流年_对宫_vs_三方A",
                      "流年_对宫_vs_三方B", "大限_本宫_vs_对宫", "大限_本宫_vs_三方A", "大限_本宫_vs_三方B",
                      "大限_对宫_vs_三方A", "大限_对宫_vs_三方B", "交叉_大限本宫_vs_流年本宫",
                      "交叉_大限对宫_vs_流年本宫", "交叉_大限三方A_vs_流年本宫", "交叉_大限三方B_vs_流年本宫",
                      "交叉_大限对宫_vs_流年三方A", "交叉_大限对宫_vs_流年三方B", "交叉_大限本宫_vs_流年三方A",
                      "交叉_大限本宫_vs_流年三方B"]}

    def add_source_score(p1_key, p2_key, value):
        key = get_source_key(p1_key, p2_key)
        if key in source_scores: source_scores[key] += value

    def record_event_with_level(reason, score, event_type, context, interaction_level, **kwargs):
        kwargs['intensity_desc'] = INTERACTION_INTENSITY_DESC.get(interaction_level, "强度未知")
        record_event(reason, score, event_type, context, events, **kwargs)

    def get_source_key(p1_key, p2_key):
        p_map = {"本宫": 0, "对宫": 1, "三方A": 2, "三方B": 3};
        scope1, name1_raw = p1_key.split('_', 1);
        scope2, name2_raw = p2_key.split('_', 1);
        scope_map = {'ln': '流年', 'dx': '大限'}
        if scope1 == scope2:
            scope_name = scope_map.get(scope1)
            if p_map.get(name1_raw, 99) > p_map.get(name2_raw, 99): name1_raw, name2_raw = name2_raw, name1_raw
            return f"{scope_name}_{name1_raw}_vs_{name2_raw}"
        else:
            ln_palace, dx_palace = (name1_raw, name2_raw) if scope1 == 'ln' else (name2_raw, name1_raw)
            return f"交叉_大限{dx_palace}_vs_流年{ln_palace}"

    BASE_SCORE_STAR = 2.0

    def get_interaction_coefficient(k1, k2):
        if {k1, k2} == {'ln_本宫', 'ln_对宫'} or {k1, k2} == {'dx_本宫', 'dx_对宫'}:
            return 1.0
        return 0.75

    # V25 核心修正: 增加新的计分组合到白名单
    VALID_INTERACTION_PAIRS = [('ln_本宫', 'ln_对宫'), ('ln_本宫', 'ln_三方A'), ('ln_本宫', 'ln_三方B'),
                               ('ln_对宫', 'ln_三方A'), ('ln_对宫', 'ln_三方B'), ('dx_本宫', 'dx_对宫'),
                               ('dx_本宫', 'dx_三方A'), ('dx_本宫', 'dx_三方B'), ('dx_对宫', 'dx_三方A'),
                               ('dx_对宫', 'dx_三方B'), ('dx_本宫', 'ln_本宫'), ('dx_对宫', 'ln_本宫'),
                               ('dx_三方A', 'ln_本宫'), ('dx_三方B', 'ln_本宫'), ('dx_对宫', 'ln_三方A'),
                               ('dx_对宫', 'ln_三方B'), ('dx_本宫', 'ln_三方A'), ('dx_本宫', 'ln_三方B')]
    fully_inactivated_pairs = set()
    sihua_inactivated_palaces = set()
    processed_interaction_fingerprints = set()

    for scope in ['ln', 'dx']:
        self_key, opp_key = f"{scope}_本宫", f"{scope}_对宫"
        if P.get(self_key) and P.get(opp_key):
            sihua_self = {get_si_hua_type(s) for s in P[self_key]['si_hua'] if get_si_hua_type(s)}
            sihua_opp = {get_si_hua_type(s) for s in P[opp_key]['si_hua'] if get_si_hua_type(s)}
            if sihua_self and sihua_self == sihua_opp:
                fully_inactivated_pairs.add(frozenset((self_key, opp_key)))
                if '化忌' in sihua_self:
                    sihua_inactivated_palaces.add(opp_key)

    # --- 2. 开始白名单评估规则 ---
    for k1, k2 in VALID_INTERACTION_PAIRS:
        if not P.get(k1) or not P.get(k2): continue
        if frozenset((k1, k2)) in fully_inactivated_pairs: continue

        content_k1 = (frozenset(P[k1]['all_stars']), frozenset(P[k1]['si_hua']))
        content_k2 = (frozenset(P[k2]['all_stars']), frozenset(P[k2]['si_hua']))
        fingerprint = tuple(sorted((str(content_k1), str(content_k2))))
        if fingerprint in processed_interaction_fingerprints: continue

        interaction_sihua_score = 0;
        interaction_star_score = 0
        level = "ln_vs_dx" if 'dx' in k1 or 'dx' in k2 else ("dx_internal" if 'dx' in k1 else "ln_internal")

        # --- 2.1 四化层面计分 ---
        if k1 not in sihua_inactivated_palaces and k2 not in sihua_inactivated_palaces:
            sihua_types1 = {get_si_hua_type(s) for s in P[k1].get('si_hua', []) if get_si_hua_type(s)}
            sihua_types2 = {get_si_hua_type(s) for s in P[k2].get('si_hua', []) if get_si_hua_type(s)}
            combined_sihua_types = sihua_types1.union(sihua_types2)

            combo_scored = False
            if '化忌' not in combined_sihua_types:
                SIHUA_GOOD_COMBOS = [({"化禄", "化权", "化科"}, "三奇加会(禄权科)", 2.0),
                                     ({"化禄", "化权"}, "禄权交会", 2.0), ({"化禄", "化科"}, "禄科交会", 2.0),
                                     ({"化权", "化科"}, "权科交会", 2.0)]
                for required_sihua, combo_name, combo_score in SIHUA_GOOD_COMBOS:
                    if required_sihua.issubset(combined_sihua_types) and \
                            not sihua_types1.isdisjoint(required_sihua) and \
                            not sihua_types2.isdisjoint(required_sihua):
                        interaction_sihua_score += combo_score
                        record_event_with_level(f"由 [{get_name(k1)} 与 {get_name(k2)}] 互动形成: {combo_name}",
                                                combo_score, "sihua_good_combo", f"{get_source_key(k1, k2)}", level,
                                                combo_name=combo_name, sihua_members=list(required_sihua))
                        combo_scored = True
                        break

                if not combo_scored:
                    s_val = 1.5
                    for s1_type in sihua_types1:
                        for s2_type in sihua_types2:
                            if s1_type in GOOD_SI_HUA_TYPES and s2_type in GOOD_SI_HUA_TYPES:
                                interaction_sihua_score += s_val
                                record_event_with_level(
                                    f"'{s1_type}'({get_name(k1)})与'{s2_type}'({get_name(k2)})吉化重叠", s_val,
                                    "sihua_overlap", f"{get_source_key(k1, k2)}", level, sihua_pair=(s1_type, s2_type),
                                    location1=get_name(k1), location2=get_name(k2))

            is_cross_interaction = '交叉' in get_source_key(k1, k2)
            if '化忌' in sihua_types1 and '化忌' in sihua_types2 and is_cross_interaction:
                interaction_sihua_score -= 1.5
                record_event_with_level(f"'化忌'({get_name(k1)})与'化忌'({get_name(k2)})重叠", -1.5, "sihua_overlap",
                                        f"{get_source_key(k1, k2)}", level, sihua_pair=('化忌', '化忌'),
                                        location1=get_name(k1), location2=get_name(k2))

        # --- 2.2 星曜层面计分 ---
        coefficient = get_interaction_coefficient(k1, k2)
        stars1, stars2 = set(P[k1]['all_stars']), set(P[k2]['all_stars'])
        for star in stars1.intersection(stars2):
            if star in GOOD_STARS or star in BAD_STARS:
                final_s = (BASE_SCORE_STAR if star in GOOD_STARS else -BASE_SCORE_STAR) * coefficient
                interaction_star_score += final_s
                record_event_with_level(f"星曜'{star}'同时出现在{get_name(k1)}与{get_name(k2)}", final_s,
                                        "star_overlap", f"{get_source_key(k1, k2)}", level, star=star,
                                        location1=get_name(k1), location2=get_name(k2))

        for pair, is_good in STAR_PAIRS.items():
            s1_star, s2_star = list(pair)
            if s1_star == s2_star and s1_star in stars1.intersection(stars2):
                continue
            if (s1_star in stars1 and s2_star in stars2) or (s2_star in stars1 and s1_star in stars2):
                final_s = (BASE_SCORE_STAR if is_good else -BASE_SCORE_STAR) * coefficient
                interaction_star_score += final_s
                record_event_with_level(f"星曜对“{s1_star}/{s2_star}”由{get_name(k1)}与{get_name(k2)}构成", final_s,
                                        "good_pair" if is_good else "bad_pair", f"{get_source_key(k1, k2)}", level,
                                        pair=pair, location1=get_name(k1), location2=get_name(k2))

        total_interaction_score = interaction_sihua_score + interaction_star_score
        if total_interaction_score != 0:
            score += total_interaction_score
            add_source_score(k1, k2, total_interaction_score)
            processed_interaction_fingerprints.add(fingerprint)

            # --- 3. 单独计算本宫自身性质分 ---
    processed_stars_in_self = set()
    ln_self_stars = set(P['ln_本宫']['all_stars'])
    for pair, is_good in STAR_PAIRS.items():
        if pair.issubset(ln_self_stars):
            s = BASE_SCORE_STAR
            score += s
            record_event_with_level(f"本宫内自带星曜对: {sorted(list(pair))}", s,
                                    "good_pair" if is_good else "bad_pair", "本宫自身性质", "ln_internal", pair=pair,
                                    location1=get_name("ln_本宫"))
            processed_stars_in_self.update(pair)

    ln_sihua_types = {get_si_hua_type(si) for si in P['ln_本宫']['si_hua']}
    if '化忌' not in ln_sihua_types:
        SIHUA_GOOD_COMBOS_SELF = [({"化禄", "化权", "化科"}, "三奇加会(禄权科)", 2.0),
                                  ({"化禄", "化权"}, "禄权交会", 2.0), ({"化禄", "化科"}, "禄科交会", 2.0),
                                  ({"化权", "化科"}, "权科交会", 2.0)]
        for required_sihua, combo_name, combo_score in SIHUA_GOOD_COMBOS_SELF:
            if required_sihua.issubset(ln_sihua_types):
                score += combo_score
                record_event_with_level(f"本宫内自带吉格: {combo_name}", combo_score, "sihua_good_combo",
                                        "本宫自身性质", "ln_internal", combo_name=combo_name,
                                        sihua_members=list(required_sihua))
                break
    if any(s in GOOD_SI_HUA_TYPES for s in ln_sihua_types): score += 1.0; record_event_with_level("流年本宫有吉化", 1.0,
                                                                                                  "single_good",
                                                                                                  "本宫自身性质",
                                                                                                  "ln_internal",
                                                                                                  source="吉化")
    if any(s in BAD_SI_HUA_TYPES for s in ln_sihua_types): score -= 1.0; record_event_with_level("流年本宫有化忌", -1.0,
                                                                                                 "single_bad",
                                                                                                 "本宫自身性质",
                                                                                                 "ln_internal",
                                                                                                 source="化忌")

    for star in P['ln_本宫']['aux_stars']:
        if star in processed_stars_in_self: continue
        if star in GOOD_STARS or star in BAD_STARS:
            s = 1.0 if star in GOOD_STARS else -1.0
            score += s
            record_event_with_level(f"流年本宫有{'吉' if s > 0 else '煞'}辅星'{star}'", s,
                                    "single_good" if s > 0 else "single_bad", "本宫自身性质", "ln_internal",
                                    source=star)

    summary_dict = generate_qualitative_summary(current_palace_name, score, events)
    filtered_source_scores = {k: round(v, 2) for k, v in source_scores.items() if v != 0.0}

    return score, filtered_source_scores, summary_dict


# V24 核心修正: 增强_process_field函数以兼容全角逗号
def parse_palace_data(raw_data):
    """
    解析原始宫位数据，将字符串或列表格式的星曜统一处理，并合并所有星曜。
    V24修正: 兼容全角逗号。
    """
    parsed_list = []
    for entry in raw_data:
        def _process_field(field_data):
            if isinstance(field_data, str):
                # 核心修正：在分割前，将全角逗号替换为半角逗号
                standardized_str = field_data.replace('，', ',')
                return [s.strip() for s in standardized_str.split(',') if s.strip()]
            elif isinstance(field_data, list):
                return [s.strip() for s in field_data if isinstance(s, str) and s.strip()]
            return []

        parsed_entry = {'main_stars': _process_field(entry[0]), 'di_zhi': entry[1], 'palace_name': entry[2] or '',
                        'si_hua': _process_field(entry[3]),
                        'aux_stars': _process_field(entry[4]) + _process_field(entry[5])}
        parsed_entry['all_stars'] = parsed_entry['main_stars'] + parsed_entry['aux_stars']
        parsed_list.append(parsed_entry)
    return parsed_list


SCORE_JUDGMENT_WORD_BANK = {
    # 评分区间：> 10 (极佳) - 对应“能”的非常肯定的回答
    "excellent": [
        "时机绝佳，成功的确定性非常高。", "这是一个不容错过的黄金机会。", "前景一片光明，结果将超出预期。",
        "形势大好，成功的可能性近乎必然。", "各方面条件都指向了积极的结果。", "成功的道路已经铺平，只需前行。",
        "此事的成功概率极高，值得全力以赴。", "这是一个收获满满的绝佳时机。", "结果非常乐观，可以大胆推进。",
        "机遇之门已经敞开，成功触手可及。", "得天时地利，所谋之事易成。", "能量充沛，做此事将势如破竹。",
        "成功的信号非常强烈且清晰。", "几乎没有悬念，结果会非常理想。", "这是一个可以实现跨越式发展的良机。",
        "所有迹象都表明，这是一个正确的决定。", "成功的果实已经成熟，静待采摘。", "放心去做，结果大概率会遂心如意。",
        "这是一个顺风顺水的局面，事半功倍。", "运势正当旺，做此事有如神助。"
    ],
    # 评分区间：7-10 (向好) - 对应“能”的肯定回答
    "good": [
        "成功的希望很大，前景值得期待。", "整体趋势向好，值得积极尝试。", "成功的概率较高，只需注意细节。",
        "大方向是正确的，坚持下去便有收获。", "机会大于挑战，结果倾向于乐观。", "这是一个有利的局面，可以积极行动。",
        "虽然会有些许波折，但结局是好的。", "成功的可能性很高，应抓住机会。", "付出的努力将有很大概率获得回报。",
        "这是一个充满希望和机遇的时期。", "事情正朝着积极的方向发展。", "此刻行动，获得理想结果的可能性很大。",
        "比较顺利，成功的可能性相当可观。", "根基稳固，达成目标的可能性较高。", "建议把握当下，成功的机会不小。",
        "前景看好，只需稳步推进即可。", "成功的条件基本成熟。", "这是一个值得投入并期待回报的时刻。",
        "整体运势偏向有利，可以抱有信心。", "成功的希望占据主导，波折是次要的。"
    ],
    # 评分区间：3-7 (中平) - 对应“不确定”，需要努力
    "mixed": [
        "机会与挑战并存，结果取决于努力。", "存在一定的可能性，但过程不会一帆风顺。", "情况较为微妙，成败系于一线之间。",
        "这是一个需要审慎乐观的局面。", "结果具有不确定性，需要精心规划。", "有成功的希望，但需付出加倍努力。",
        "前景尚不明朗，需要边走边看。", "关键在于能否处理好过程中的变量。", "一个需要智慧和耐心的局面。",
        "不好不坏，维持现状或小幅进展是可能的。", "需要积极争取，机会不会自动上门。", "结果的走向，一半在天意，一半在人为。",
        "可以尝试，但务必设置好止损点。", "成功的可能性存在，但并非唾手可得。", "这是一个考验能力和应变技巧的时刻。",
        "不好断言，情况的发展有多种可能。", "外部环境助力有限，更多要依靠自己。", "若准备充分，仍有希望达成目标。",
        "局面复杂，需要步步为营。", "当下状况中性，未来的走向掌握在自己手中。"
    ],
    # 评分区间：0-3 (挑战) - 对应“不能”的委婉说法
    "challenging": [
        "情况比预想的复杂，建议谨慎行事。", "成功的希望存在，但过程将充满考验。", "会遇到一些明显的阻力，需做好准备。",
        "结果的走向存在较大变数。", "这不是一个理想的行动时机。", "需要付出超常的努力才可能看到希望。",
        "建议优先考虑风险，而非潜在收益。", "过程会比较曲折，不确定性较高。", "达成目标的难度较大，需要有心理准备。",
        "外部条件不太有利，需要加倍小心。", "成功的可能性偏低，建议保守应对。", "可能会有事与愿违的倾向。",
        "投入与产出可能不成正比。", "建议暂缓行动，待时机好转再做打算。", "需要有应对意外状况的充分准备。",
        "成功的道路上布满了需要清除的障碍。", "此刻推进此事，可能会感到力不从心。", "希望的火种微弱，需要小心呵护。",
        "这是一个需要“求稳”而非“求进”的时刻。", "环境中的制约因素较多。"
    ],
    # 评分区间：< 0 (艰难) - 对应“不能”的强烈但委婉的说法
    "very_challenging": [
        "前景并不乐观，建议重新评估计划。", "情况复杂且充满挑战，成功的希望渺茫。", "投入与产出可能严重失衡，风险极高。",
        "此刻并非有利时机，强求恐有损失。", "会遇到始料未及的阻碍，建议规避。", "成功的可能性微乎其微，不宜冒险。",
        "前路挑战重重，建议将重心放在防守。", "这是一个“韬光养晦”的时期，不宜出击。", "形势对达成目标非常不利。",
        "各种条件都预示着此事困难重重。", "建议果断放弃或调整方向。", "此事的潜在风险远大于可见的机遇。",
        "强行推进，可能会导致令人失望的结果。", "成功的希望不大，更应关注如何避免损失。", "这是一个大概率会事倍功半的局面。",
        "周围环境充满了不确定性和制约。", "建议保存实力，不作无谓的尝试。", "从各方面看，这都不是一个明智的选择。",
        "如逆水行舟，每前进一步都异常艰难。", "形势如雾里看花，看不真切，极易迷失。"
    ]
}


def get_judgment_word_bank_for_score(score: Optional[float]) -> list:
    """根据分数返回对应的判断词词库。"""
    if score is None:
        return SCORE_JUDGMENT_WORD_BANK["mixed"]  # 如果没有分数，返回中性词库
    if score > 10:
        return SCORE_JUDGMENT_WORD_BANK["excellent"]
    elif 7 <= score <= 10:
        return SCORE_JUDGMENT_WORD_BANK["good"]
    elif 3 <= score < 7:
        return SCORE_JUDGMENT_WORD_BANK["mixed"]
    elif 0 <= score < 3:
        return SCORE_JUDGMENT_WORD_BANK["challenging"]
    else:  # score < 0
        return SCORE_JUDGMENT_WORD_BANK["very_challenging"]


def generate_score_based_judgment(final_score_json_str: Optional[str], relevant_palaces: List[str]) -> str:
    """
    根据 final_score 的JSON字符串和相关的宫位，生成一段定性的运势判断描述。
    【V2 - 已修复别名问题】
    """
    if not final_score_json_str:
        return "无量化评分信息。"

    try:
        scores = json.loads(final_score_json_str)
    except (json.JSONDecodeError, TypeError):
        return "量化评分信息格式错误。"

    # --- 【核心修改 1】建立一个宫位名称的别名映射表 ---
    # 目的是将所有可能的输入名 (如'事业宫') 都指向分数数据源中的官方名 (如'官禄宫')
    PALACE_SYNONYM_MAP = {
        # 别名: 官方名
        "事业宫": "官禄宫",
        "官禄宫": "官禄宫",  # 官方名也映射到自己，方便处理
        "交友宫": "仆役宫",
        "仆役宫": "仆役宫"
        # 未来如果还有其他别名，在这里继续添加即可
    }

    def get_judgment_phrase(score: float) -> str:
        if score > 10:
            return "趋势极佳，预示成功机会很大，前景非常光明"
        elif 7 <= score <= 10:
            return "趋势向好，有较大可能性成功，值得积极尝试"
        elif 3 <= score < 7:
            return "机会与挑战并存，结果有一定不确定性，需要谨慎乐观"
        elif 0 < score < 3:
            return "进展可能较为平缓，存在一些变数，需多加留意"
        elif score <= 0:
            return "趋势不甚明朗，可能遇到一定困难或阻碍，建议谨慎行事"
        return "状况平稳，无明显吉凶倾向"

    judgment_parts = []

    palaces_to_analyze = relevant_palaces if relevant_palaces else list(scores.keys())

    # --- 【核心修改 2】改造循环逻辑，使用映射表查找分数 ---
    for original_palace_name in palaces_to_analyze:
        # 1. 查找别名对应的官方名。如果不在映射表里，就用它本身。
        canonical_name = PALACE_SYNONYM_MAP.get(original_palace_name, original_palace_name)

        # 2. 使用官方名(canonical_name)去分数词典(scores)里查找
        if canonical_name in scores:
            score = scores[canonical_name]
            if isinstance(score, (int, float)):
                phrase = get_judgment_phrase(score)
                # 3. 【重要】在最终输出时，仍然使用用户更熟悉的原始名称(original_palace_name)
                judgment_parts.append(f"【{original_palace_name}】(得分: {score:.1f})：{phrase}。")

    if not judgment_parts:
        return "未找到相关宫位的量化评分。"

    return " ".join(judgment_parts)


# 您 ziwei_ai_function.py 中的其他辅助函数也可以放在这里，如
# _parse_lenient_json, _extract_first_json_object 等
# 为了保持原样，我将它们放在 clients/vllm_client.py 和 services/chat_processor.py 中


def parse_fixed_format_birth_info(query: str) -> Tuple[Dict[str, Any], str]:
    """
    解析固定格式的出生信息。

    预期格式: "公历 1995-05-20 10:30:00 女，我最近工作压力很大，财运怎么样?"

    Args:
        query: 用户的完整查询字符串

    Returns:
        Tuple[Dict[str, Any], str]: (出生信息字典, 实际问题部分)
    """
    import re

    # 定义匹配固定格式的正则表达式
    # 格式: (公历|农历) YYYY-MM-DD HH:MM:SS (男|女)，问题内容
    # pattern = r'^(公历|农历)\s+(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{1,2}):(\d{1,2})\s+(男|女)，(.+)$'
    pattern = r'^(公历|农历)\s+(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{1,2}):(\d{1,2})\s+(男|女)[\s，]*(.*)$'

    match = re.match(pattern, query.strip())

    if not match:
        # 如果不匹配固定格式，返回空的出生信息和原始查询
        return {}, query

    calendar_type = match.group(1)  # 公历 或 农历
    year = int(match.group(2))
    month = int(match.group(3))
    day = int(match.group(4))
    hour = int(match.group(5))
    minute = int(match.group(6))
    second = int(match.group(7))  # 秒数，虽然不用于紫微斗数，但需要解析
    gender = match.group(8)
    question = match.group(9)

    # 构建出生信息字典
    birth_info = {
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "gender": gender,
        "is_lunar": calendar_type == "农历",
        "traditional_hour_branch": None
    }

    # 基本验证
    if not (1900 <= year <= 2100):
        raise ValueError(f"年份 {year} 超出合理范围 (1900-2100)")
    if not (1 <= month <= 12):
        raise ValueError(f"月份 {month} 超出合理范围 (1-12)")
    if not (1 <= day <= 31):
        raise ValueError(f"日期 {day} 超出合理范围 (1-31)")
    if not (0 <= hour <= 23):
        raise ValueError(f"小时 {hour} 超出合理范围 (0-23)")
    if not (0 <= minute <= 59):
        raise ValueError(f"分钟 {minute} 超出合理范围 (0-59)")
    if gender not in ["男", "女"]:
        raise ValueError(f"性别 '{gender}' 必须是 '男' 或 '女'")

    logger.info(f"成功解析固定格式出生信息: {birth_info}")
    logger.info(f"提取的问题: {question}")

    return birth_info, question