import re
import calendar
import logging
from datetime import date, datetime, timedelta
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

try:
    from chinese_calendar import get_holiday_detail
    HAS_CHINESE_CALENDAR = True
except ImportError:
    HAS_CHINESE_CALENDAR = False
    logger.warning("chinese_calendar 模块未安装，农历年龄计算将使用fallback逻辑")


def convert_hour_to_time_index(birth_info: Dict[str, Any]) -> int:
    """将24小时制转换为传统时辰索引"""
    if birth_info.get('traditional_hour_branch'):
        # 这里的 TRADITIONAL_HOUR_TO_TIME_INDEX 在 chat_processor 中从 config 引入并用于构建 payload，
        # 在本工具函数中仅负责 hour → index 的计算逻辑，真正的映射仍在调用方完成。
        from config import TRADITIONAL_HOUR_TO_TIME_INDEX
        return TRADITIONAL_HOUR_TO_TIME_INDEX.get(birth_info.get('traditional_hour_branch', ''), 0)
    elif 'hour' in birth_info and birth_info['hour'] is not None:
        total_minutes = birth_info['hour'] * 60 + birth_info.get('minute', 0)
        if 0 <= total_minutes <= 60:
            return 0  # 早子时
        elif 60 < total_minutes <= 180:
            return 1  # 丑时
        elif 180 < total_minutes <= 300:
            return 2  # 寅时
        elif 300 < total_minutes <= 420:
            return 3  # 卯时
        elif 420 < total_minutes <= 540:
            return 4  # 辰时
        elif 540 < total_minutes <= 660:
            return 5  # 巳时
        elif 660 < total_minutes <= 780:
            return 6  # 午时
        elif 780 < total_minutes <= 900:
            return 7  # 未时
        elif 900 < total_minutes <= 1020:
            return 8  # 申时
        elif 1020 < total_minutes <= 1140:
            return 9  # 酉时
        elif 1140 < total_minutes <= 1260:
            return 10  # 戌时
        elif 1260 < total_minutes <= 1380:
            return 11  # 亥时
        elif 1380 < total_minutes <= 1440:
            return 12  # 晚子时
        else:
            return 0  # 默认值
    else:
        return 0  # 默认值


def normalize_vague_time_expressions(cleaned_question: str) -> str:
    """
    在第一层意图分类之前，将用户问题中的模糊时间表述替换为具体时间表述。
    
    需求变更：将模糊时间直接替换为"从YYYY年MM月DD日到YYYY年MM月DD日"或包含时间的具体范围，按当前日期动态计算。
    """
    now = datetime.now()
    today = now.date()

    # ---------- 全局模糊时间词数量检查：出现两个及以上直接跳过规范化 ----------
    vague_tokens = [
        "将来", "未来", "近", "这", "后", "后面",
        "近期", "这个阶段", "当前阶段", "现阶段", "当下阶段",
        "接下来", "这段时间", "最近",
        "下个阶段", "下阶段", "下一阶段", "下下阶段",
        "这两天", "马上","季度",
        "年前", "年内",
        "下个月",
        "今年", "明年", "后年", "去年", "前年",
        "下一年", "下年", "后一年",
        "下半年", "上半年",
        "年", "月", "日", "时", "号","天"
    ]

    vague_count = 0
    for token in vague_tokens:
        vague_count += len(re.findall(re.escape(token), cleaned_question))

    # 如果上述模糊/时间相关词整体出现次数 >= 2，则认为时间结构较复杂，避免误改，直接返回原问题
    if vague_count >= 2:
        return cleaned_question

    # ---------- 绝对时间复杂度检查：出现多个或多层级时间时直接跳过规范化 ----------
    # 规则：
    # 1) 如果同时出现“年+月”或“月+日”，为了避免破坏用户明确给定的时间点，直接返回原问题；
    # 2) 如果出现多个“年”或多个“月”或多个“日”（代表有时间范围或多时间点），同样直接返回原问题。

    # 年：包括 2025年 / 1999年 以及 今年/明年/后年/去年/前年
    year_matches = []
    year_matches += re.findall(r'\d{4}年', cleaned_question)
    year_matches += re.findall(r'今年|明年|后年|去年|前年', cleaned_question)

    # 月：只统计带数字的“X月”，不区分是否有“份”
    month_matches = re.findall(r'\d{1,2}月', cleaned_question)

    # 日：统计“X日/号/天”
    day_matches = re.findall(r'\d{1,2}(日|号|天)', cleaned_question)

    year_count = len(year_matches)
    month_count = len(month_matches)
    day_count = len(day_matches)

    has_year_and_month = year_count >= 1 and month_count >= 1
    has_month_and_day = month_count >= 1 and day_count >= 1
    has_multi_year = year_count > 1
    has_multi_month = month_count > 1
    has_multi_day = day_count > 1

    if has_year_and_month or has_month_and_day or has_multi_year or has_multi_month or has_multi_day:
        # 出现明确的复杂绝对时间表达（如“明年3月”“明年3月3日”“2023年到2025年”“3月到5月”等），
        # 为避免误改，直接返回原始问题。
        return cleaned_question

    def fmt_d(d: date) -> str:
        return f"{d.year}年{d.month}月{d.day}日"

    def fmt_dt(dt: datetime) -> str:
        return f"{dt.year}年{dt.month}月{dt.day}日 {dt.hour:02d}:{dt.minute:02d}"

    def fmt_y(year: int) -> str:
        """格式化为“YYYY年”"""
        return f"{year}年"

    def fmt_ym(year: int, month: int) -> str:
        """格式化为“YYYY年M月”"""
        return f"{year}年{month}月"

    def month_add(year: int, month: int, add: int) -> (int, int):
        m = month + add
        y = year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        return y, m

    def last_day(y: int, m: int) -> int:
        return calendar.monthrange(y, m)[1]

    # 解析中文或数字形式的数量，支持 1-30
    def parse_number(num_str: str) -> int:
        num_str = num_str.strip()
        # 纯数字
        if num_str.isdigit():
            try:
                return int(num_str)
            except ValueError:
                return 0

        cn_map = {
            "零": 0, "〇": 0,
            "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
        }

        # 单个数字：一、二、三...
        if num_str in cn_map:
            return cn_map[num_str]

        # 十、十一、十二...十九
        if num_str == "十":
            return 10
        if len(num_str) == 2:
            if num_str[0] == "十" and num_str[1] in cn_map:
                # 十一 ~ 十九
                return 10 + cn_map[num_str[1]]
            if num_str[1] == "十" and num_str[0] in cn_map:
                # 二十、三十
                return cn_map[num_str[0]] * 10

        # 二十一、二十二... 二十九
        if len(num_str) == 3 and num_str[1] == "十" and num_str[0] in cn_map and num_str[2] in cn_map:
            return cn_map[num_str[0]] * 10 + cn_map[num_str[2]]

        return 0

    # -------- 带“几”的模糊时间 --------

    # 未来几天 / 近几天 / 这几天 / 后几天 → 从明天到三天后
    def repl_days(match):
        start = today + timedelta(days=1)
        end = start + timedelta(days=2)
        return f"从{fmt_d(start)}到{fmt_d(end)}"
    cleaned_question = re.sub(r'(将来|未来|近|这|后|后面)几天', repl_days, cleaned_question)

    # 几小时 → 从当前时刻到3小时后
    def repl_hours(match):
        start = now
        end = now + timedelta(hours=3)
        return f"从{fmt_dt(start)}到{fmt_dt(end)}"
    cleaned_question = re.sub(r'(将来|未来|近|这|后|后面)几小时', repl_hours, cleaned_question)

    # 几周 → 从明天到21天后
    def repl_weeks(match):
        start = today + timedelta(days=1)
        end = start + timedelta(days=21 - 1)
        return f"从{fmt_d(start)}到{fmt_d(end)}"
    cleaned_question = re.sub(r'(将来|未来|近|这|后|后面)几周', repl_weeks, cleaned_question)

    # 几个月 / 几月 → 下月1日到下月起三个月的月底
    def repl_months(match):
        start_y, start_m = month_add(today.year, today.month, 1)  # 下月
        end_y, end_m = month_add(start_y, start_m, 2)  # 共3个月
        # 月相关：只精确到“年+月”，不落到日
        return f"从{fmt_ym(start_y, start_m)}到{fmt_ym(end_y, end_m)}"
    cleaned_question = re.sub(r'(将来|未来|近|这|后|后面)几个月?', repl_months, cleaned_question)

    # 几年 → 明年到往后5年的年底
    def repl_years(match):
        start_y = today.year + 1
        end_y = start_y + 5 - 1
        # 年相关：只精确到年份区间
        return f"从{fmt_y(start_y)}到{fmt_y(end_y)}"
    cleaned_question = re.sub(r'(将来|未来|近|这|后|后面|下|接下来|下个|下一|下)几年', repl_years, cleaned_question)

    # -------- 带具体数字的时间：支持 1~30（阿拉伯数字或中文数字） --------
    num_pattern = r'(?P<num>[一二三四五六七八九十零〇两\d]{1,3})'
    prefix_pattern = r'(将来|未来|近|这|后|后面|下|接下来|下个|下一|下)'

    # N天 → 从明天起 N 天
    def repl_days_num(match):
        n = parse_number(match.group("num"))
        if n <= 0:
            return match.group(0)
        n = min(n, 30)
        start = today + timedelta(days=1)
        end = start + timedelta(days=n - 1)
        return f"从{fmt_d(start)}到{fmt_d(end)}"

    cleaned_question = re.sub(
        prefix_pattern + num_pattern + r'天',
        repl_days_num,
        cleaned_question,
    )

    # N小时 → 从当前时刻起 N 小时
    def repl_hours_num(match):
        n = parse_number(match.group("num"))
        if n <= 0:
            return match.group(0)
        n = min(n, 30)
        start = now
        end = now + timedelta(hours=n)
        return f"从{fmt_dt(start)}到{fmt_dt(end)}"

    cleaned_question = re.sub(
        prefix_pattern + num_pattern + r'小时',
        repl_hours_num,
        cleaned_question,
    )

    # N周 → 从明天起 N 周
    def repl_weeks_num(match):
        n = parse_number(match.group("num"))
        if n <= 0:
            return match.group(0)
        n = min(n, 30)
        days = n * 7
        start = today + timedelta(days=1)
        end = start + timedelta(days=days - 1)
        return f"从{fmt_d(start)}到{fmt_d(end)}"

    cleaned_question = re.sub(
        prefix_pattern + num_pattern + r'周',
        repl_weeks_num,
        cleaned_question,
    )

    # N个月 / N月 → 下月起 N 个月
    def repl_months_num(match):
        n = parse_number(match.group("num"))
        if n <= 0:
            return match.group(0)
        n = min(n, 30)
        start_y, start_m = month_add(today.year, today.month, 1)  # 下月
        end_y, end_m = month_add(start_y, start_m, n - 1)
        # 只精确到“年+月”
        return f"从{fmt_ym(start_y, start_m)}到{fmt_ym(end_y, end_m)}"

    cleaned_question = re.sub(
        prefix_pattern + num_pattern + r'个月?',
        repl_months_num,
        cleaned_question,
    )

    # N年 → 明年起 N 年
    def repl_years_num(match):
        n = parse_number(match.group("num"))
        if n <= 0:
            return match.group(0)
        n = min(n, 30)
        start_y = today.year + 1
        end_y = start_y + n - 1
        # 只精确到年份区间
        return f"从{fmt_y(start_y)}到{fmt_y(end_y)}"

    cleaned_question = re.sub(
        prefix_pattern + num_pattern + r'年',
        repl_years_num,
        cleaned_question,
    )

    # 近期 / 最近 / 这段时间 / 下阶段等 → 未来三个月（下月起三个月）
    cleaned_question = re.sub(
        r'近期|这个阶段|当前阶段|现阶段|当下阶段|接下来|这段时间|最近|下个阶段|下阶段|下一阶段|下下阶段|下个季度',
        repl_months,
        cleaned_question
    )

    # 这两天 / 马上 → 今天到明天
    def repl_two_days(match):
        start = today
        end = today + timedelta(days=1)
        return f"从{fmt_d(start)}到{fmt_d(end)}"
    cleaned_question = re.sub(r'这两天|马上', repl_two_days, cleaned_question)

    # 年前 / 年内 → 当年全年
    def repl_year_inside(match):
        y = today.year
        return fmt_y(y)
    cleaned_question = re.sub(r'年前|年内', repl_year_inside, cleaned_question)

    # 下个月 → 下月1日到下月月底
    def repl_next_month(match):
        start_y, start_m = month_add(today.year, today.month, 1)
        # 单个月份：用“年+月”
        return f"从{fmt_ym(start_y, start_m)}到{fmt_ym(start_y, start_m)}"
    cleaned_question = re.sub(r'下个月', repl_next_month, cleaned_question)

    # 今年 / 当年 / 这一年 / 这年 / 本年：直接替换为具体年份（不再使用“从X年到X年”的区间形式）
    def repl_this_year(match):
        y = today.year
        # 直接输出“YYYY年”
        return fmt_y(y)
    cleaned_question = re.sub(r'今年|当年|这一年|这年|本年', repl_this_year, cleaned_question)

    # 明年 / 下一年 / 下年：直接替换为具体年份
    def repl_next_year(match):
        y = today.year + 1
        return fmt_y(y)
    cleaned_question = re.sub(r'明年|下一年|下年', repl_next_year, cleaned_question)

    # 后年 / 后一年：从后年1月1日到后年12月31日
    def repl_year_after_next(match):
        y = today.year + 2
        return fmt_y(y)
    cleaned_question = re.sub(r'后年|后一年', repl_year_after_next, cleaned_question)

    # 下半年仍保持原逻辑：当年7月1日到12月31日（如需跨年可再调整）
    current_year = today.year
    cleaned_question = re.sub(r'(?<!\\d)下半年', f'{current_year}年7月1日到12月31日', cleaned_question)

    return cleaned_question



def smart_normalize_punctuation(text: str) -> str:
    """
    智能标点符号标准化：保护出生信息格式，只对问题部分进行标准化。
    """
    birth_info_pattern = re.compile(r"(公历|农历)?\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+(男|女)")
    birth_match = birth_info_pattern.search(text)

    if not birth_match:
        return normalize_punctuation_simple(text)

    birth_info_end = birth_match.end()
    birth_part = text[:birth_info_end]  # 出生信息部分，保持原样
    question_part = text[birth_info_end:].strip()  # 问题部分，进行标准化

    normalized_question = normalize_punctuation_simple(question_part)

    # 重新组合
    result = birth_part + " " + normalized_question if question_part else birth_part

    logger.debug(f"智能符号标准化:")
    logger.debug(f"  原文: '{text}'")
    logger.debug(f"  出生信息部分: '{birth_part}'")
    logger.debug(f"  问题部分: '{question_part}' -> '{normalized_question}'")
    logger.debug(f"  结果: '{result}'")

    return result


def normalize_punctuation_simple(text: str) -> str:
    """
    简单的标点符号标准化，将英文标点转换为中文标点。
    """
    punctuation_map = {
        '?': '？',
        '!': '！',
        ',': '，',
        ';': '；',
        '(': '（',
        ')': '）',
        '[': '［',
        ']': '］',
        '{': '｛',
        '}': '｝',
        '"': '"',  # 转换为中文双引号
        "'": "'",  # 转换为中文单引号
        '.': '。'  # 句号转换（但要小心数字中的小数点）
    }

    normalized_text = text
    for en_punct, cn_punct in punctuation_map.items():
        # 对于句号，避免替换数字中的小数点
        if en_punct == '.':
            # 只替换句末的点，不替换数字中的小数点
            normalized_text = re.sub(r'\.(?!\d)', cn_punct, normalized_text)
        else:
            normalized_text = normalized_text.replace(en_punct, cn_punct)

    return normalized_text


def is_birth_info_complete(info: Dict[str, Any]) -> bool:
    """检查一个字典是否包含完整的紫微斗数排盘所需信息。"""
    if not isinstance(info, dict):
        return False

    # 1. 检查日期字段是否存在且不为 None
    has_date = all(info.get(key) is not None for key in ["year", "month", "day"])

    # 2. 检查性别字段是否有效
    has_gender = info.get("gender") in ["男", "女"]

    # 3. 检查时间字段：必须有 hour 或 traditional_hour_branch
    has_hour = info.get("hour") is not None
    has_branch = info.get("traditional_hour_branch") is not None and info.get("traditional_hour_branch") != ""
    has_time = has_hour or has_branch

    return has_date and has_gender and has_time


def calculate_lunar_age(birth_info_to_use: Dict[str, Any]) -> int:
    """
    根据出生年份和当年春节时间，计算农历年龄（虚岁）。
    """
    birth_year = birth_info_to_use['year']

    now = datetime.now()
    current_year = now.year

    # 1. 定义日期范围：从今年一月一日到三月一日，涵盖春节
    start_date = date(current_year, 1, 1)
    end_date = date(current_year, 3, 1)

    lunar_new_year_date = None

    # 遍历范围内的日期，找到春节
    if HAS_CHINESE_CALENDAR:
        date_to_check = start_date
        while date_to_check <= end_date:
            # get_holiday_detail 返回 (是否是节假日, 节假日名称)
            is_holiday_flag, holiday_name = get_holiday_detail(date_to_check)

            if is_holiday_flag and holiday_name == '春节':
                # 找到了春节的公历日期
                lunar_new_year_date = date_to_check
                break

            date_to_check += timedelta(days=1)

    if not lunar_new_year_date:
        # 极端情况 fallback：如果没找到或模块未安装，设定为二月一日
        lunar_new_year_date = date(current_year, 2, 1)

    # 3. 初始化年龄为 公历年差
    current_age = now.year - birth_year

    if now.date() >= lunar_new_year_date:
        return current_age + 1
    else:
        return current_age


def resolve_horoscope_date(time_expression: str, current_time: datetime) -> Dict[str, Any]:
    """
    解析时间表达式，返回运势日期信息
    
    Args:
        time_expression: 时间表达式，如 "28年"、"明年"、"2025年3月" 等
        current_time: 当前时间，用于计算相对时间
        
    Returns:
        包含解析后时间信息的字典，包括：
        - resolved_horoscope_date: 解析后的日期字符串 (YYYY-MM-DD HH:MM:SS)
        - target_year, target_month, target_day, target_hour, target_minute: 时间组件
        - analysis_level: 分析级别 (yearly/monthly/daily)
        - relative_time_indicator: 相对时间指示器
        - multi_month_span: 多月跨度（可选）
    """
    result = {
        "resolved_horoscope_date": None,
        "target_year": None,
        "target_month": None,
        "target_day": None,
        "target_hour": None,
        "target_minute": None,
        "analysis_level": None,
        "relative_time_indicator": time_expression,
        "multi_month_span": None
    }
    
    if not time_expression or not time_expression.strip():
        logger.warning(f"[resolve_horoscope_date] time_expression为空或None")
        return result
    
    time_expression = time_expression.strip()
    logger.info(f"[resolve_horoscope_date] 开始解析时间表达式: '{time_expression}'")
    current_year = current_time.year
    current_month = current_time.month
    current_day = current_time.day
    
    # 首先检查是否为时间范围（从X到Y），如果是，使用起始日期
    time_range_pattern = r'从\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*到\s*(\d{4})\s*年\s*(\d{1,2})\s*月'
    range_match = re.search(time_range_pattern, time_expression)
    if range_match:
        start_year = int(range_match.group(1))
        start_month = int(range_match.group(2))
        end_year = int(range_match.group(3))
        end_month = int(range_match.group(4))
        
        # 使用起始日期
        result["target_year"] = start_year
        result["target_month"] = start_month
        result["analysis_level"] = "monthly"
        result["resolved_horoscope_date"] = f"{start_year}-{start_month:02d}-15 12:00:00"
        
        # 计算月份跨度
        if start_year == end_year:
            span = end_month - start_month + 1
        else:
            span = (end_year - start_year) * 12 + (end_month - start_month) + 1
        result["multi_month_span"] = span
        
        return result
    
    # 检查是否为时间范围（X年X月到X年X月，没有"从"字）
    time_range_pattern2 = r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*到\s*(\d{4})\s*年\s*(\d{1,2})\s*月'
    range_match2 = re.search(time_range_pattern2, time_expression)
    if range_match2:
        start_year = int(range_match2.group(1))
        start_month = int(range_match2.group(2))
        end_year = int(range_match2.group(3))
        end_month = int(range_match2.group(4))
        
        # 使用起始日期
        result["target_year"] = start_year
        result["target_month"] = start_month
        result["analysis_level"] = "monthly"
        result["resolved_horoscope_date"] = f"{start_year}-{start_month:02d}-15 12:00:00"
        
        # 计算月份跨度
        if start_year == end_year:
            span = end_month - start_month + 1
        else:
            span = (end_year - start_year) * 12 + (end_month - start_month) + 1
        result["multi_month_span"] = span
        
        return result
    
    # 先检查"上半年"和"下半年"，因为它们可能包含"年"字但不应被年份匹配逻辑处理
    if "上半年" in time_expression:
        # 上半年指1月到6月，属于一年，按流年处理
        # 检查是否有年份信息（如"今年上半年"、"明年上半年"）
        target_year = current_year
        if "明年" in time_expression or "来年" in time_expression:
            target_year = current_year + 1
        elif "后年" in time_expression:
            target_year = current_year + 2
        elif "去年" in time_expression or "前年" in time_expression:
            target_year = current_year - 1
        
        result["target_year"] = target_year
        result["analysis_level"] = "yearly"
        # 使用上半年中间日期（3月15日）
        result["resolved_horoscope_date"] = f"{target_year}-03-15 12:00:00"
        return result
    elif "下半年" in time_expression:
        # 下半年指7月到12月，属于一年，按流年处理
        # 检查是否有年份信息（如"今年下半年"、"明年下半年"）
        target_year = current_year
        if "明年" in time_expression or "来年" in time_expression:
            target_year = current_year + 1
        elif "后年" in time_expression:
            target_year = current_year + 2
        elif "去年" in time_expression or "前年" in time_expression:
            target_year = current_year - 1
        
        result["target_year"] = target_year
        result["analysis_level"] = "yearly"
        # 使用下半年中间日期（9月15日）
        result["resolved_horoscope_date"] = f"{target_year}-09-15 12:00:00"
        return result
    
    # 解析年份：支持 "28年"、"2028年"、"明年"、"后年" 等
    year_match = re.search(r'(\d{2,4})年', time_expression)
    if year_match:
        year_str = year_match.group(1)
        if len(year_str) == 2:
            # 两位数字，如 "28年" -> 2028年
            year = 2000 + int(year_str)
            if year < current_year:
                year = 1900 + int(year_str)  # 如果小于当前年份，可能是19xx年
        else:
            # 四位数字，如 "2028年"
            year = int(year_str)
        
        result["target_year"] = year
        result["analysis_level"] = "yearly"
        
        # 检查是否有月份信息
        month_match = re.search(r'(\d{1,2})月', time_expression)
        if month_match:
            month = int(month_match.group(1))
            if 1 <= month <= 12:
                result["target_month"] = month
                result["analysis_level"] = "monthly"
                
                # 检查是否有日期信息
                day_match = re.search(r'(\d{1,2})(?:日|号)', time_expression)
                if day_match:
                    day = int(day_match.group(1))
                    if 1 <= day <= 31:
                        result["target_day"] = day
                        result["analysis_level"] = "daily"
        
        # 构建日期字符串
        if result["target_year"]:
            year = result["target_year"]
            month = result["target_month"] or 6  # 默认6月
            day = result["target_day"] or 15  # 默认15日
            hour = 12  # 默认中午12点
            minute = 0
            
            result["target_hour"] = hour
            result["target_minute"] = minute
            result["resolved_horoscope_date"] = f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:00"
    
    # 解析相对时间：明年、后年、来年等
    elif "明年" in time_expression or "下一年" in time_expression or "来年" in time_expression:
        result["target_year"] = current_year + 1
        result["analysis_level"] = "yearly"
        result["resolved_horoscope_date"] = f"{current_year + 1}-06-15 12:00:00"
    elif "后年" in time_expression:
        result["target_year"] = current_year + 2
        result["analysis_level"] = "yearly"
        result["resolved_horoscope_date"] = f"{current_year + 2}-06-15 12:00:00"
    elif "今年" in time_expression or "当年" in time_expression:
        result["target_year"] = current_year
        result["analysis_level"] = "yearly"
        result["resolved_horoscope_date"] = f"{current_year}-06-15 12:00:00"
    elif "年内" in time_expression:
        # 年内指当前年份内，从当前日期到年底
        result["target_year"] = current_year
        result["analysis_level"] = "yearly"
        # 使用当前日期，因为"年内"是从当前日期到年底
        result["resolved_horoscope_date"] = f"{current_year}-{current_month:02d}-{current_day:02d} 12:00:00"
    elif "下个月" in time_expression:
        next_month = current_month + 1
        next_year = current_year
        if next_month > 12:
            next_month = 1
            next_year += 1
        result["target_year"] = next_year
        result["target_month"] = next_month
        result["analysis_level"] = "monthly"
        result["resolved_horoscope_date"] = f"{next_year}-{next_month:02d}-15 12:00:00"
    elif "这个月" in time_expression or "本月" in time_expression:
        result["target_year"] = current_year
        result["target_month"] = current_month
        result["analysis_level"] = "monthly"
        result["resolved_horoscope_date"] = f"{current_year}-{current_month:02d}-15 12:00:00"
    elif "今天" in time_expression:
        result["target_year"] = current_year
        result["target_month"] = current_month
        result["target_day"] = current_day
        result["analysis_level"] = "daily"
        result["resolved_horoscope_date"] = f"{current_year}-{current_month:02d}-{current_day:02d} 12:00:00"
    
    return result

