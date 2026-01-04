import re
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


def _find_relative_past_phrases(text: str) -> List[str]:
    """
    检测明显指向“已经过去时间段”的相对时间表达。

    说明：
    - 一旦命中这些关键词，就可以直接视为“询问过去时间”，无需再进入意图识别。
    - 这里的词表可以根据业务需要自行扩展或收缩。
    """
    relative_past_keywords = [
        "昨天",
        "前天",
        "大前天",
        "上周",
        "上星期",
        "上礼拜",
        "前一周",
        "上个月",
        "上個月",
        "上上个月",
        "上上個月",
        "去年",
        "前年",
        "往年",
        "上一年",
        "上半年",  # 在当前日期接近年末时，通常也可以视为“已过”
        "过去几年",
        "过去这几年",
        "以前",
        "之前",
        "曾经",
        "当时",
        "那时候",
        "那会儿",
        "已经发生",
        "已经发生过",
    ]
    hits = {kw for kw in relative_past_keywords if kw in text}
    return sorted(hits)


def _last_day_of_month(year: int, month: int) -> date:
    """给定年份和月份，返回该月最后一天的日期。"""
    if month == 12:
        return date(year, 12, 31)
    first_next_month = date(year, month + 1, 1)
    return first_next_month - timedelta(days=1)


def _analyze_absolute_dates(text: str, today: date) -> Tuple[bool, List[str]]:
    """
    解析“YYYY年M月D日 / YYYY年M月 / 今年M月”这类绝对时间表达，
    判断是否完全早于 today。

    返回:
        (has_past, matched_phrases)
    """
    matched_phrases: List[str] = []
    has_past = False

    # 1) 解析 2025年3月15日 / 2025年3月 的形式
    abs_pattern = re.compile(r"(?P<year>\d{4})年(?P<month>\d{1,2})月(?:(?P<day>\d{1,2})日)?")
    for m in abs_pattern.finditer(text):
        year = int(m.group("year"))
        month = int(m.group("month"))
        day_str = m.group("day")
        phrase = m.group(0)

        try:
            if day_str:
                day = int(day_str)
                dt = date(year, month, day)
                if dt < today:
                    has_past = True
                    matched_phrases.append(phrase)
            else:
                # 只有“年-月”，认为询问该整月；如果整月都在 today 之前，则视为过去
                last_day = _last_day_of_month(year, month)
                if last_day < today:
                    has_past = True
                    matched_phrases.append(phrase)
        except ValueError:
            # 非法日期，直接忽略
            logger.debug(f"[时间识别] 忽略无法解析的日期表达: {phrase}")
            continue

    # 2) 解析 “今年3月” 这类相对绝对混合表达
    this_year = today.year
    this_year_month_pattern = re.compile(r"今年(?P<month>\d{1,2})月")
    for m in this_year_month_pattern.finditer(text):
        month = int(m.group("month"))
        phrase = m.group(0)
        try:
            last_day = _last_day_of_month(this_year, month)
            if last_day < today:
                has_past = True
                matched_phrases.append(phrase)
        except ValueError:
            logger.debug(f"[时间识别] 忽略无法解析的今年X月表达: {phrase}")
            continue

    return has_past, matched_phrases


def should_block_past_time_query(
    text: str,
    now: Optional[datetime] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    判断用户提问是否明显是在询问“已经过去的时间段”，用于在进入意图识别之前快速拦截。

    规则分两层：
    1. 先匹配相对时间关键词（如“上个月”“去年”“以前”“已经发生”等），命中即视为询问过去；
    2. 再解析绝对时间（如“2025年3月”“2024年5月10日”“今年3月”），
       如果这些时间段整体早于当前日期 today，也视为询问过去。

    参数:
        text:      标准化后的用户提问文本（建议用 normalized_prompt）
        now:       当前时间，默认使用 datetime.now()；可在单元测试中注入固定时间

    返回:
        (should_block, reason, debug_info)
        - should_block: 是否应该直接拒绝（不再进入意图识别）
        - reason:       简短的中文说明，可用于日志
        - debug_info:   辅助调试信息（如当前日期、命中的时间短语等）
    """
    if not text:
        return False, "", {"now": None, "matched_relative": [], "matched_absolute": []}

    if now is None:
        now = datetime.now()
    today = now.date()

    # 1. 相对时间关键词匹配（最高优先级）
    relative_hits = _find_relative_past_phrases(text)
    if relative_hits:
        reason = f"检测到指向过去时间的相对时间表达: {', '.join(relative_hits)}"
        debug_info: Dict[str, Any] = {
            "now": today.isoformat(),
            "matched_relative": relative_hits,
            "matched_absolute": [],
        }
        return True, reason, debug_info

    # 2. 绝对时间解析（年-月-日 / 年-月 / 今年X月）
    has_past_abs, abs_hits = _analyze_absolute_dates(text, today)
    if has_past_abs:
        reason = f"检测到早于当前日期 {today.isoformat()} 的绝对时间表达: {', '.join(abs_hits)}"
        debug_info = {
            "now": today.isoformat(),
            "matched_relative": [],
            "matched_absolute": abs_hits,
        }
        return True, reason, debug_info

    # 3. 未发现明确的过去时间表达，不拦截
    return False, "", {
        "now": today.isoformat(),
        "matched_relative": [],
        "matched_absolute": [],
    }


