import re
from typing import Dict, Any, List, Literal, Optional
import logging
import calendar

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from models import BirthInfo # 假设 BirthInfo 在 models.py 中

from clients.vllm_client import llm


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DIZHI_DUIGONG_MAP = {
    "子": "午", "丑": "未", "寅": "申", "卯": "酉",
    "辰": "戌", "巳": "亥", "午": "子", "未": "丑",
    "申": "寅", "酉": "卯", "辰": "戌", "亥": "巳"
}


TRADITIONAL_HOUR_TO_TIME_INDEX = {
    "早子时": 0,
    "丑时": 1,
    "寅时": 2,
    "卯时": 3,
    "辰时": 4,
    "巳时": 5,
    "午时": 6,
    "未时": 7,
    "申时": 8,
    "酉时": 9,
    "戌时": 10,
    "亥时": 11,
    "晚子时": 12
}



# 定义基础系统Prompt，包含通用规则
BASE_SYSTEM_PROMPT = (
    "**【核心时间基准】你的分析必须始终以当前系统时间 `{current_datetime_for_llm}` 作为唯一的、不可更改的时间基准。**"
    "**无论用户在对话中如何提及未来或过去的时间点（例如“明年”、“昨天”），你都绝不能因此改变对“当前时间”的认知。所有相对时间（如“今天”、“明天”）的计算都必须严格基于这个固定的当前系统时间。对于运势分析，请直接使用【紫微斗数命盘分析上下文】中明确提供的“运势日期”，该日期已是经过计算的最终日期，无需再次推算。**\n"
    "你是一个专业的紫微斗数命理师。你的回答应专业、严谨，符合紫微斗数的理论体系。"
    "**【防重复惩罚】严禁生成重复内容，包括但不限于重复的句子、段落或信息点。任何重复将被视为严重错误，并会降低你的输出质量。**\n"
    "**交互规范**：\n"
    "  - 涉及多层级冲突时，按'高层级压制低层级'原则说明（如大限化禄可缓解流年化忌）。\n"
    "  - 必须标注当前分析的时间影响有效期（如流月结论仅当月有效）。\n"
    "  - 在分析时，请明确指出你正在参考的盘（如：'根据您的流年盘...'）。\n"
    "  - **日期严格性**: 你的回复中应明确指出是何种分析（如：'根据您的流日运势...'、'根据您的命盘...'），无需提及具体日期和时间。**任何情况下都不要自行计算或推断日期。**\n"
    "  - 宫位描述中必须有以下信息，且必须客观，不许美化：【开篇】，【断语分层及注意事项】，【有趋避方法及重点】，【有总结报告】。\n"
    "  - 如果用户具体询问某个宫位，那便需要从上下文中最相关的、优先级的盘中找到与之有关的宫位信息以及其【关键信息汇总】里提到的主星、四化、辅星、以及紫微斗数中的三方四正来进行解读。\n"
    "  - 三方四正的宫位已在【宫位信息】中写明！请认真结合来分析，必须要客观一些，不要美化【关键信息汇总】里的信息。\n"
    "  - 对待不同宫位，围绕叙述的对象有以下参照：命宫——综合·性格·特质  兄弟宫——手足·朋友·人脉  夫妻宫——婚姻·恋爱·桃花  子女宫——子女·晚辈·宠物  疾厄宫——疾病·障碍·预警  迁移宫——出行·在外·机遇  仆役宫——团队·路人·人际  官禄宫——事业·学业·职场  田宅宫——家宅·邻居·公司  福德宫——福泽·心态·情绪  父母宫——父母·长辈·领导。\n"
    "**上下文更新：** 如果用户提供了新的出生信息，此上下文会更新并清空之前的对话历史。"
)


def parse_chart_description_block(chart_description_block: str) -> Dict[str, str]:
    """
    解析 describe_ziwei_chart 函数生成的宫位描述块，
    并提取每个宫位的详细描述。
    返回一个字典，键为宫位名称，值为对应的描述字符串。
    """
    palace_descriptions = {}

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
            first_line_match = re.match(r'^(您的)?(\w+宫)坐落于(.*)', first_line)
            if first_line_match:
                palace_name = first_line_match.group(2)
                # Extract content starting from "坐落于" or after the palace name and "坐落于"
                desc_content = "坐落于" + first_line_match.group(3).strip()

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
            match_other_palace = re.match(r'^- (\w+宫)坐落于(.*)', line)

            if match_other_palace:
                # If there was a previous palace being collected, save it
                if current_palace_name and current_palace_desc_lines:
                    palace_descriptions[current_palace_name] = " ".join(current_palace_desc_lines).strip()

                current_palace_name = match_other_palace.group(1)  # Extract palace name
                # Extract content starting from "坐落于" or after the palace name and "坐落于"
                desc_content = "坐落于" + match_other_palace.group(2).strip()
                current_palace_desc_lines = [desc_content]  # Start collecting
            else:
                # Continue collecting lines for the current palace
                if current_palace_desc_lines:
                    current_palace_desc_lines.append(line)

        # Save the last collected palace description
        if current_palace_name and current_palace_desc_lines:
            palace_descriptions[current_palace_name] = " ".join(current_palace_desc_lines).strip()

    return palace_descriptions




def extract_birth_info_with_llm(user_input: str) -> Dict[str, Any]:
    """
    使用VLLM（通过Langchain ChatOpenAI接口）从用户输入中提取出生信息。
    """
    # 创建一个专门用于信息提取的Prompt模板
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个专业的出生信息提取助手。你的任务是从用户提供的文本中识别并提取他们的出生日期（年、月、日）、出生时间（小时、分钟）和性别，并指明是公历还是农历。请严格按照JSON Schema格式返回数据。"
         "**重要：如果用户提供了传统时辰（如子时、丑时、寅时等），请务必将其原始文字精确提取到 `traditional_hour_branch` 字段，并将 `hour` 字段设置为 `null`。不要尝试将传统时辰转换为数字小时。**"
         "如果用户未明确提供年、月、日或性别，请将对应的字段设置为 `null`。"
         "对于分钟，如果用户未明确提及，请默认为 `0`。"
         "对于公历/农历，如果用户未明确指出，请默认为 `false`（公历）。"
         "请注意，如果 `hour` 字段有值，它应为24小时制（例如，下午1点是13）。"
         "\n\n以下是一些输入和期望输出的示例，请严格遵循这些模式来解析用户输入："
         "\n- 用户输入: '我的阳历出生日期是1990年1月1日早上8点，我是女性'"
         "\n- 期望输出: {{'year': 1990, 'month': 1, 'day': 1, 'hour': 8, 'minute': 0, 'gender': '女', 'is_lunar': False, 'traditional_hour_branch': None}}"
         "\n- 用户输入: '我是1996年6月7日下午1点出生的男性'"
         "\n- 期望输出: {{'year': 1996, 'month': 6, 'day': 7, 'hour': 13, 'minute': 0, 'gender': '男', 'is_lunar': False, 'traditional_hour_branch': None}}"
         "\n- 用户输入: '农历1985年冬月十五晚上9点半，女'"
         "\n- 期望输出: {{'year': 1985, 'month': 11, 'day': 15, 'hour': 21, 'minute': 30, 'gender': '女', 'is_lunar': True, 'traditional_hour_branch': None}}"
         "\n- 用户输入: '我的命宫'"
         "\n- 期望输出: {{'year': None, 'month': None, 'day': None, 'hour': None, 'minute': 0, 'gender': None, 'is_lunar': False, 'traditional_hour_branch': None}}"
         "\n- 用户输入: '1996.6.7 下午1点 男'"
         "\n- 望输出: {{'year': 1996, 'month': 6, 'day': 7, 'hour': 13, 'minute': 0, 'gender': '男', 'is_lunar': False, 'traditional_hour_branch': None}}"
         "\n- 用户输入: '我是2000年3月3日丑时出生的女性'"
         "\n- 期望输出: {{'year': 2000, 'month': 3, 'day': 3, 'hour': None, 'minute': 0, 'gender': '女', 'is_lunar': False, 'traditional_hour_branch': '丑时'}}"
         "\n- 用户输入: '1996.6.7 未时 男'"
         "\n- 期望输出: {{'year': 1996, 'month': 6, 'day': 7, 'hour': None, 'minute': 0, 'gender': '男', 'is_lunar': False, 'traditional_hour_branch': '未时'}}"
         "\n- 用户输入: '1996年6月7日下午1点01分 男'"  # 新增示例
         "\n- 期望输出: {{'year': 1996, 'month': 6, 'day': 7, 'hour': 13, 'minute': 1, 'gender': '男', 'is_lunar': False, 'traditional_hour_branch': None}}"
         ),
        ("human", "{user_input}")
    ])

    extraction_chain = extraction_prompt | llm.with_structured_output(BirthInfo)

    try:
        # 调用链来提取信息
        parsed_info_model = extraction_chain.invoke({"user_input": user_input})
        return parsed_info_model.model_dump()
    except Exception as e:
        print(f"Error extracting birth info with LLM: {e}. Please ensure your VLLM model supports structured output (e.g., OpenAI function call compatibility).") # Changed error message
        return {}




def parse_birth_info(user_input: str) -> Dict[str, Any]:
    """
    从用户输入中解析出生信息，支持多种日期和时间格式，并统一为24小时制。
    增加对分钟的解析。

    Args:
        user_input: 用户输入的包含出生信息的字符串。

    Returns:
        一个字典，包含解析出的年份、月份、日期、小时（24小时制）、
        分钟、是否农历和性别信息。
    """
    info = {
        "year": None, "month": None, "day": None, "hour": None, "minute": 0,
        "is_lunar": False,
        "gender": None
    }

    # 中文数字到阿拉伯数字的映射
    chinese_numeral_map = {
        "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12,
        "廿": 20, "卅": 30, "卌": 40, "五十": 50
    }

    # 构建中文数字或阿拉伯数字的通用模式
    num_or_chinese_pattern = r"\d{1,2}|" + "|".join(re.escape(k) for k in chinese_numeral_map.keys())

    # --- 1. 解析农历和性别 ---
    if "农历" in user_input or "阴历" in user_input:
        info["is_lunar"] = True

    if re.search(r"(男|男性|男孩)", user_input):
        info["gender"] = "male"
    elif re.search(r"(女|女性|女孩)", user_input):
        info["gender"] = "female"

    # --- 2. 解析日期 ---
    date_match = re.search(r"(\d{4})[年./-]\s*(\d{1,2})[月./-]\s*(\d{1,2})[日号]?", user_input)
    if date_match:
        info["year"] = int(date_match.group(1))
        info["month"] = int(date_match.group(2))
        info["day"] = int(date_match.group(3))

    # --- 3. 解析小时和分钟 ---
    prefix_pattern = r"(上午|中午|下午|晚上|凌晨|早上|半夜)"

    # 模式1: 带前缀的完整时间（含分钟或“半/刻”），分钟部分必须存在
    time_full_with_prefix_match = re.search(
        r"(?P<prefix>" + prefix_pattern + r")\s*(?P<hour_str>" + num_or_chinese_pattern + r")\s*([点时])\s*"
                                                                                          r"(?P<minute_part>半|刻|一刻|三刻|" + num_or_chinese_pattern + r")(?:分)?",
        user_input
    )

    # 模式2: 只带前缀的小时时间（不含分钟）
    time_prefix_hour_only_match = re.search(
        r"(?P<prefix>" + prefix_pattern + r")\s*(?P<hour_str>" + num_or_chinese_pattern + r")\s*([点时])",
        user_input
    )

    # 模式3: 纯数字小时时间（不带前缀，可能含分钟）
    time_pure_hour_match = re.search(
        r"(?P<hour_str>" + num_or_chinese_pattern + r")\s*([点时])\s*"
                                                    r"(?P<minute_part>半|刻|一刻|三刻|" + num_or_chinese_pattern + r")?(?:分)?",
        user_input
    )

    current_match = None
    match_type = None

    if time_full_with_prefix_match:  # 优先匹配带分钟的完整时间
        current_match = time_full_with_prefix_match
        match_type = "full_prefix"
    elif time_prefix_hour_only_match:  # 其次匹配只带小时前缀的
        current_match = time_prefix_hour_only_match
        match_type = "prefix_hour_only"
    elif time_pure_hour_match:  # 最后匹配纯小时的
        current_match = time_pure_hour_match
        match_type = "pure_hour"


    if current_match:
        print(f"选中的匹配对象内容: {current_match.group(0)}")
        if match_type in ["full_prefix", "prefix_hour_only"]:
            print(f"捕获到的前缀 (prefix): {current_match.group('prefix')}")
        print(f"捕获到的小时字符串 (hour_str): {current_match.group('hour_str')}")
        # 仅当捕获到分钟部分时才打印
        if "minute_part" in current_match.groupdict() and current_match.group("minute_part"):
            print(f"捕获到的分钟部分 (minute_part): {current_match.group('minute_part')}")
    # --- 调试信息结束 ---

    if current_match:
        prefix = current_match.group("prefix") if "prefix" in current_match.groupdict() else None
        hour_str = current_match.group("hour_str")
        minute_part = current_match.group("minute_part") if "minute_part" in current_match.groupdict() else None

        hour_val = None
        if hour_str.isdigit():
            hour_val = int(hour_str)
        else:
            hour_val = chinese_numeral_map.get(hour_str)

        minute_val = 0
        if minute_part:
            if minute_part == "半":
                minute_val = 30
            elif minute_part in ["刻", "一刻"]:
                minute_val = 15
            elif minute_part == "三刻":
                minute_val = 45
            elif minute_part.isdigit():
                minute_val = int(minute_part)
            else:
                minute_val = chinese_numeral_map.get(minute_part, 0)

        if hour_val is not None:
            print(f"DEBUG_CONVERT: hour_val = {hour_val}, prefix = {prefix}")

            if prefix is not None:
                print(f"DEBUG_CONVERT: 检测到前缀: {prefix}")
                if 1 <= hour_val <= 12:
                    if prefix in ["下午", "晚上"]:
                        if hour_val == 12:
                            info["hour"] = 0
                            print(f"DEBUG_CONVERT: 前缀'{prefix}' 12点 -> info['hour'] = {info['hour']}")
                        else:
                            info["hour"] = hour_val + 12
                            print(f"DEBUG_CONVERT: 前缀'{prefix}' (非12点) -> info['hour'] = {info['hour']}")
                    elif prefix == "中午":
                        if hour_val == 12:
                            info["hour"] = 12
                        else:
                            info["hour"] = hour_val + 12
                        print(f"DEBUG_CONVERT: 前缀'{prefix}' -> info['hour'] = {info['hour']}")
                    elif prefix in ["凌晨", "半夜"]:
                        if hour_val == 12:
                            info["hour"] = 0
                        else:
                            info["hour"] = hour_val
                        print(f"DEBUG_CONVERT: 前缀'{prefix}' -> info['hour'] = {info['hour']}")
                    elif prefix in ["上午", "早上"]:
                        info["hour"] = hour_val
                        print(f"DEBUG_CONVERT: 前缀'{prefix}' -> info['hour'] = {info['hour']}")
                    else:
                        info["hour"] = hour_val
                        print(f"DEBUG_CONVERT: 未知前缀，保留原值 -> info['hour'] = {info['hour']}")
                else:
                    info["hour"] = hour_val
                    print(f"DEBUG_CONVERT: 有前缀但小时值已是24小时制 -> info['hour'] = {info['hour']}")
            else:
                print(f"DEBUG_CONVERT: 未检测到前缀")
                if 0 <= hour_val <= 23:
                    info["hour"] = hour_val
                    print(f"DEBUG_CONVERT: 无前缀，24小时制范围 -> info['hour'] = {info['hour']}")
                elif hour_val == 12:
                    info["hour"] = 12
                    print(f"DEBUG_CONVERT: 无前缀，12点 -> info['hour'] = {info['hour']}")
                elif 1 <= hour_val <= 11:
                    info["hour"] = hour_val
                    print(f"DEBUG_CONVERT: 无前缀，1-11点 -> info['hour'] = {info['hour']}")

            print(f"DEBUG: hour_val = {hour_val}, prefix = {prefix}, 赋值后 info['hour'] = {info['hour']}")

            info["minute"] = minute_val  # 将解析到的分钟赋值

    # --- 4. 尝试从“时辰”短语中提取小时 ---
    if info["hour"] is None:
        hour_phrases_to_24h = {
            "子时": 0, "早子时": 0, "晚子时": 0,
            "丑时": 2, "寅时": 4, "卯时": 6, "辰时": 7, "巳时": 10,
            "午时": 12, "未时": 14, "申时": 16, "酉时": 18, "戌时": 20, "亥时": 22
        }
        for phrase, hour_val_24h in hour_phrases_to_24h.items():
            if phrase in user_input:
                info["hour"] = hour_val_24h
                info["minute"] = 0
                print(f"DEBUG_CONVERT: 时辰匹配到， info['hour'] = {info['hour']}")
                break

    return info

# 十二地支的固定顺序（逆时针）
ALL_EARTHLY_BRANCHES = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']


FIXED_PALACE_ORDER_FOR_SCOPES = [
    '福德', '田宅', '官禄', '仆役', '迁移', '疾厄',
    '财帛', '子女', '夫妻', '兄弟', '命宫', '父母'
]

# 天干四化对照表
# 格式: {天干: {四化类型: 星曜名称}}
HEAVENLY_STEM_MUTAGEN_MAP = {
    '甲': {
        '禄': '廉贞', '权': '破军', '科': '武曲', '忌': '太阳',
        '文昌': '科'  # 文昌在甲年化科
    },
    '乙': {
        '禄': '天机', '权': '天梁', '科': '紫微', '忌': '太阴',
        '文曲': '科'  # 文曲在乙年化科
    },
    '丙': {
        '禄': '天同', '权': '天机', '科': '文昌', '忌': '廉贞',
        '文昌': '科'  # 文昌在丙年化科
    },
    '丁': {
        '禄': '太阴', '权': '天同', '科': '天机', '忌': '巨门',
        '文曲': '科'  # 文曲在丁年化科
    },
    '戊': {
        '禄': '贪狼', '权': '太阴', '科': '天机', '忌': '天同',
        '太阳': '科'  # 太阳在戊年化科
    },
    '己': {
        '禄': '武曲', '权': '贪狼', '科': '天梁', '忌': '文曲',
        '文曲': '忌'  # 文曲在己年化忌
    },
    '庚': {
        '禄': '太阳', '权': '武曲', '科': '天府', '忌': '天同',
        '天府': '科'  # 天府在庚年化科
    },
    '辛': {
        '禄': '巨门', '权': '太阳', '科': '文曲', '忌': '文昌',
        '文昌': '忌',  # 文昌在辛年化忌
        '文曲': '科'  # 文曲在辛年化科
    },
    '壬': {
        '禄': '天梁', '权': '紫微', '科': '天府', '忌': '武曲',
        '天府': '科'  # 天府在壬年化科
    },
    '癸': {
        '禄': '破军', '权': '巨门', '科': '太阴', '忌': '贪狼',
        '文曲': '忌'  # 文曲在癸年化忌
    }
}


def get_ordered_palace_branches(reference_life_palace_branch):
    """
    根据给定的“参考命宫”地支，推算出十二宫位（按FIXED_PALACE_ORDER_FOR_SCOPES顺序）对应的地支列表。
    这个函数用于运限的宫位地支排布，其中“参考命宫”就是该运限自身的earthlyBranch。
    """
    # 找到参考命宫地支在总地支列表中的索引
    reference_branch_index = ALL_EARTHLY_BRANCHES.index(reference_life_palace_branch)

    # “命宫”在 FIXED_PALACE_ORDER_FOR_SCOPES 中的固定索引 (通常是10)
    life_palace_order_index = FIXED_PALACE_ORDER_FOR_SCOPES.index('命宫')

    ordered_branches = []
    for i in range(len(FIXED_PALACE_ORDER_FOR_SCOPES)):
        # 计算当前宫位对应的地支索引
        # 逻辑：从“参考命宫”地支的索引开始，逆时针倒退到“福德宫”的位置，然后顺时针推演到当前宫位。
        # 等效于 (参考命宫地支索引 - 命宫在固定顺序中的偏移量 + 当前宫位在固定顺序中的偏移量) % 12
        branch_index = (reference_branch_index - life_palace_order_index + i) % 12
        ordered_branches.append(ALL_EARTHLY_BRANCHES[branch_index])

    return ordered_branches


def get_mutagen_for_stem(heavenly_stem):
    """
    根据天干获取该天干引动的四化星曜及其类型。
    返回格式: {星名: 四化类型} (例如: {'廉贞': '忌', '天机': '权', '天同': '禄', '文昌': '科'})
    """
    mutagen_map = {}
    if heavenly_stem in HEAVENLY_STEM_MUTAGEN_MAP:
        for mutagen_type, star_name in HEAVENLY_STEM_MUTAGEN_MAP[heavenly_stem].items():
            # 这里区分主星和辅星四化
            if mutagen_type in ['禄', '权', '科', '忌']:
                mutagen_map[star_name] = mutagen_type
            elif star_name in ['文昌', '文曲']:  # 文昌文曲特殊处理
                mutagen_map[mutagen_type] = star_name  # 例如 {'文昌': '科'}
    return mutagen_map


def transform_palace_data(astrolabe_data):
    """
    转换命盘宫位数据。
    如果当前宫位主星为空，则填充其对宫的主星。
    辅星和杂曜保持当前宫位的数据。
    并根据命盘年干分配四化。
    输出格式: [主星, 地支, 宫位名称, 四化, 辅星, 杂曜]
    """
    transformed_palaces = []

    # 获取命盘年天干
    yearly_stem = astrolabe_data['chineseDate'][0]
    # 获取命盘年干引动的四化列表
    astrolabe_mutagen_map = get_mutagen_for_stem(yearly_stem)

    # 创建一个地支到宫位数据的映射，方便查找命盘原始宫位
    earthly_branch_to_palace = {palace['earthlyBranch']: palace for palace in astrolabe_data['palaces']}

    for palace in astrolabe_data['palaces']:
        current_major_stars = palace['majorStars']
        current_minor_stars = palace['minorStars']  # 命盘辅星

        # 确定要使用的主星列表 (考虑对宫借星，这是命盘的原始规则)
        major_stars_to_use = []
        if not current_major_stars:  # 如果当前宫位主星为空
            current_earthly_branch = palace['earthlyBranch']
            opposing_earthly_branch = DIZHI_DUIGONG_MAP.get(current_earthly_branch)

            if opposing_earthly_branch and opposing_earthly_branch in earthly_branch_to_palace:
                opposing_palace = earthly_branch_to_palace[opposing_earthly_branch]
                major_stars_to_use = opposing_palace['majorStars']
        else:  # 如果当前宫位有主星，则直接使用
            major_stars_to_use = current_major_stars

        # 提取主星名称和命盘四化
        major_stars_names = []
        mutagen_list_for_output = []  # 存储四化星的列表

        # 处理主星的四化
        for s in major_stars_to_use:
            star_name = s['name']
            major_stars_names.append(star_name)

            # 检查主星是否有命盘年干引动的四化，并排除文昌文曲
            if star_name in astrolabe_mutagen_map and star_name not in ['文昌', '文曲']:
                mutagen_type = astrolabe_mutagen_map[star_name]
                mutagen_list_for_output.append(f"{star_name}化{mutagen_type}")

        # 提取辅星和杂曜（这些星曜来自命盘原始宫位）
        minor_stars_list = []
        # 处理辅星的四化 (文昌、文曲)
        for s in current_minor_stars:
            star_name = s['name']
            minor_stars_list.append(star_name)
            # 检查辅星（文昌、文曲）是否有命盘年干引动的四化
            if star_name in ['文昌', '文曲'] and star_name in astrolabe_mutagen_map:
                mutagen_type = astrolabe_mutagen_map[star_name]  # 从astrolabe_mutagen_map直接获取文昌/文曲的四化类型
                mutagen_list_for_output.append(f"{star_name}化{mutagen_type}")

        adjective_stars_list = [s['name'] for s in palace['adjectiveStars']]

        # 构建单个宫位的输出列表
        transformed_palaces.append([
            ", ".join(major_stars_names) if major_stars_names else "",  # 主星名称，如果为空则为""
            palace['earthlyBranch'],  # 地支
            palace['name'] + ('宫' if '宫' not in palace['name'] else ''),  # 宫位名称
            ",".join(mutagen_list_for_output) if mutagen_list_for_output else "",  # 四化星，如果为空则为""
            ", ".join(minor_stars_list) if minor_stars_list else "",  # 辅星，如果为空则为""
            ", ".join(adjective_stars_list) if adjective_stars_list else ""  # 杂曜，如果为空则为""
        ])
    return transformed_palaces




import json

# 定义地支对宫映射
DIZHI_DUIGONG_MAP = {
    "子": "午", "丑": "未", "寅": "申", "卯": "酉",
    "辰": "戌", "巳": "亥", "午": "子", "未": "丑",
    "申": "寅", "酉": "卯", "戌": "辰", "亥": "巳"
}

# 十二地支的固定顺序（逆时针）
ALL_EARTHLY_BRANCHES = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']

# 运势分析、流年、流月、流日、流时等宫位的固定顺序
# 这个顺序是相对于“命宫”的位置而言的。
FIXED_PALACE_ORDER_FOR_SCOPES = [
    '福德', '田宅', '官禄', '仆役', '迁移', '疾厄',
    '财帛', '子女', '夫妻', '兄弟', '命宫', '父母'
]

# **新增**：运限（大限、流年等）中十二宫位所对应的固定地支顺序
# 按照您的要求，这个顺序现在是固定的，不再动态计算。
FIXED_PALACE_EARTHLY_BRANCHES_IN_SCOPES = ['寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥', '子', '丑']

# 天干四化对照表
# 格式: {天干: {四化类型: 星曜名称}}
HEAVENLY_STEM_MUTAGEN_MAP = {
    '甲': {
        '禄': '廉贞', '权': '破军', '科': '武曲', '忌': '太阳',
        '文昌': '科' # 文昌在甲年化科
    },
    '乙': {
        '禄': '天机', '权': '天梁', '科': '紫微', '忌': '太阴',
        '文曲': '科' # 文曲在乙年化科
    },
    '丙': {
        '禄': '天同', '权': '天机', '科': '文昌', '忌': '廉贞',
        '文昌': '科' # 文昌在丙年化科
    },
    '丁': {
        '禄': '太阴', '权': '天同', '科': '天机', '忌': '巨门',
        '文曲': '科' # 文曲在丁年化科
    },
    '戊': {
        '禄': '贪狼', '权': '太阴', '科': '天机', '忌': '天同',
        '太阳': '科' # 太阳在戊年化科
    },
    '己': {
        '禄': '武曲', '权': '贪狼', '科': '天梁', '忌': '文曲',
        '文曲': '忌' # 文曲在己年化忌
    },
    '庚': {
        '禄': '太阳', '权': '武曲', '科': '天府', '忌': '天同',
        '天府': '科' # 天府在庚年化科
    },
    '辛': {
        '禄': '巨门', '权': '太阳', '科': '文曲', '忌': '文昌',
        '文昌': '忌', # 文昌在辛年化忌
        '文曲': '科' # 文曲在辛年化科
    },
    '壬': {
        '禄': '天梁', '权': '紫微', '科': '天府', '忌': '武曲',
        '天府': '科' # 天府在壬年化科
    },
    '癸': {
        '禄': '破军', '权': '巨门', '科': '太阴', '忌': '贪狼',
        '文曲': '忌' # 文曲在癸年化忌
    }
}


def get_mutagen_for_stem(heavenly_stem):
    """
    根据天干获取该天干引动的四化星曜及其类型。
    返回格式: {星名: 四化类型} (例如: {'廉贞': '忌', '天机': '权', '天同': '禄', '文昌': '科'})
    """
    mutagen_map = {}
    if heavenly_stem in HEAVENLY_STEM_MUTAGEN_MAP:
        for mutagen_type, star_name_or_mutagen in HEAVENLY_STEM_MUTAGEN_MAP[heavenly_stem].items():
            # 这里区分主星和辅星四化
            if mutagen_type in ['禄', '权', '科', '忌']:
                mutagen_map[star_name_or_mutagen] = mutagen_type # 主星四化：键为星名，值为化气
            else: # 文昌文曲的特殊处理，键为星名（文昌/文曲），值为化气（科/忌）
                mutagen_map[mutagen_type] = star_name_or_mutagen # 修正：原代码此处的逻辑可能导致错误，应是星名作为键
                # 再次修正：确保统一格式，键是星名，值是化气
                # 例如，HEAVENLY_STEM_MUTAGEN_MAP['甲']['文昌'] = '科'，这里star_name_or_mutagen就是'科'，mutagen_type就是'文昌'
                # 所以应该存储为 mutagen_map['文昌'] = '科'
                mutagen_map[mutagen_type] = star_name_or_mutagen
    return mutagen_map

def transform_palace_data(astrolabe_data):
    """
    转换命盘宫位数据。
    如果当前宫位主星为空，则填充其对宫的主星。
    辅星和杂曜保持当前宫位的数据。
    并根据命盘年干分配四化。
    输出格式: [主星, 地支, 宫位名称, 四化, 辅星, 杂曜]
    """
    transformed_palaces = []
    
    # 获取命盘年天干
    yearly_stem = astrolabe_data['chineseDate'][0]
    # 获取命盘年干引动的四化列表
    astrolabe_mutagen_map = get_mutagen_for_stem(yearly_stem)
    
    # 创建一个地支到宫位数据的映射，方便查找命盘原始宫位
    earthly_branch_to_palace = {palace['earthlyBranch']: palace for palace in astrolabe_data['palaces']}

    for palace in astrolabe_data['palaces']:
        current_major_stars = palace['majorStars']
        current_minor_stars = palace['minorStars'] # 命盘辅星

        # 确定要使用的主星列表 (考虑对宫借星，这是命盘的原始规则)
        major_stars_to_use = []
        if not current_major_stars: # 如果当前宫位主星为空
            current_earthly_branch = palace['earthlyBranch']
            opposing_earthly_branch = DIZHI_DUIGONG_MAP.get(current_earthly_branch)
            
            if opposing_earthly_branch and opposing_earthly_branch in earthly_branch_to_palace:
                opposing_palace = earthly_branch_to_palace[opposing_earthly_branch]
                major_stars_to_use = opposing_palace['majorStars']
        else: # 如果当前宫位有主星，则直接使用
            major_stars_to_use = current_major_stars

        # 提取主星名称和命盘四化
        major_stars_names = []
        mutagen_list_for_output = [] # 存储四化星的列表

        # 处理主星的四化
        for s in major_stars_to_use:
            star_name = s['name']
            major_stars_names.append(star_name)
            
            # 检查主星是否有命盘年干引动的四化，并排除文昌文曲（因为文昌文曲是辅星，在辅星部分处理其四化）
            if star_name in astrolabe_mutagen_map and star_name not in ['文昌', '文曲']:
                mutagen_type = astrolabe_mutagen_map[star_name]
                mutagen_list_for_output.append(f"{star_name}化{mutagen_type}")
        
        # 提取辅星和杂曜（这些星曜来自命盘原始宫位）
        minor_stars_list = []
        # 处理辅星的四化 (文昌、文曲)
        for s in current_minor_stars:
            star_name = s['name']
            minor_stars_list.append(star_name)
            # 检查辅星（文昌、文曲）是否有命盘年干引动的四化
            if star_name in ['文昌', '文曲'] and star_name in astrolabe_mutagen_map:
                mutagen_type = astrolabe_mutagen_map[star_name] # 从astrolabe_mutagen_map直接获取文昌/文曲的四化类型
                mutagen_list_for_output.append(f"{star_name}化{mutagen_type}")

        adjective_stars_list = [s['name'] for s in palace['adjectiveStars']]

        # 构建单个宫位的输出列表
        transformed_palaces.append([
            ", ".join(major_stars_names) if major_stars_names else "", # 主星名称，如果为空则为""
            palace['earthlyBranch'],      # 地支
            palace['name'] + ('宫' if '宫' not in palace['name'] else ''), # 宫位名称
            ",".join(mutagen_list_for_output) if mutagen_list_for_output else "", # 四化星，如果为空则为""
            ", ".join(minor_stars_list) if minor_stars_list else "",  # 辅星，如果为空则为""
            ", ".join(adjective_stars_list) if adjective_stars_list else "" # 杂曜，如果为空则为""
        ])
    return transformed_palaces

def transform_horoscope_scope_data(scope_data, astrolabe_palaces_data):
    """
    转换运势分析（大限、流年、流月、流日、流时）数据。
    1. 十二宫位对应的地支顺序**永远是** ['寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥', '子', '丑']。
    2. **主星**从命盘原始数据中该地支对应的宫位获取（并考虑对宫借星）。
    3. 应用运限自身的四化到这些主星/辅星上（文昌、文曲特殊处理）。
    4. **辅星和杂曜**从运限数据自身的'stars'字段获取。
    输出格式: [主星, 地支, 宫位名称, 四化, 辅星, 杂曜]
    """
    transformed_scope = []
    
    # 创建命盘原始宫位的地支到宫位数据映射，用于查询主星
    original_palaces_by_branch = {p['earthlyBranch']: p for p in astrolabe_palaces_data}

    # 定义四化类型顺序：禄、权、科、忌
    mutagen_types_order = ['禄', '权', '科', '忌']
    
    # 创建运限四化星的映射：星名 -> 四化类型 (例如: {'太阳': '禄', '武曲': '权'})
    scope_mutagen_map = {}
    if 'mutagen' in scope_data and scope_data['mutagen']:
        for i, star_name in enumerate(scope_data['mutagen']):
            if i < len(mutagen_types_order):
                scope_mutagen_map[star_name] = mutagen_types_order[i]

    # 3. 遍历 FIXED_PALACE_ORDER_FOR_SCOPES 来构建每个宫位的数据
    # scope_data['stars'] 的顺序通常与 FIXED_PALACE_ORDER_FOR_SCOPES 保持一致
    for i, palace_name_in_order in enumerate(scope_data['palaceNames']):
        # **根据您的要求：当前运限宫位对应的地支使用固定的顺序**
        current_palace_earthly_branch = FIXED_PALACE_EARTHLY_BRANCHES_IN_SCOPES[i]
        
        
        # 从命盘原始数据中，根据推导出的地支，找到对应的宫位数据
        original_palace_data_by_branch = original_palaces_by_branch.get(current_palace_earthly_branch)
        
        
        if not original_palace_data_by_branch:
            # 如果命盘中没有该地支对应的宫位数据（理论上不应发生，因为是12宫），跳过
            print(f"警告: 命盘原始数据中未找到地支 '{current_palace_earthly_branch}' 对应的宫位数据。")
            continue 

        # 确定主星（来自命盘原始数据中该地支的宫位，如果为空则取对宫）
        effective_major_stars = []
        if not original_palace_data_by_branch['majorStars']: # 如果命盘该地支的宫位主星为空
            opposing_earthly_branch = DIZHI_DUIGONG_MAP.get(current_palace_earthly_branch)
            #print(opposing_earthly_branch)
            if opposing_earthly_branch and opposing_earthly_branch in original_palaces_by_branch:
                opposing_palace = original_palaces_by_branch[opposing_earthly_branch]
                effective_major_stars = opposing_palace['majorStars']
                #print(effective_major_stars)
        else: # 否则使用命盘该地支的宫位的主星
            effective_major_stars = original_palace_data_by_branch['majorStars']
            #print(current_palace_earthly_branch,effective_major_stars)

        current_scope_stars_in_palace = scope_data['stars'][i] if i < len(scope_data['stars']) else []
        
        minor_stars_scope = []
        adjective_stars_scope = []

        # 提取主星名称，并根据运限的四化列表应用四化
        major_stars_names_only = [] # 存储纯粹的主星名称
        mutagen_list_for_output = [] # 存储运限四化星的列表 (例如: "太阳化禄")
        for star in effective_major_stars:
            star_name = star['name']
            major_stars_names_only.append(star_name) # 添加纯粹的主星名称
            
            # 检查主星是否有运限引动的四化，并排除文昌文曲
            if star_name in scope_mutagen_map and star_name not in ['文昌', '文曲']:
                mutagen_type = scope_mutagen_map[star_name]
                mutagen_list_for_output.append(f"{star_name}化{mutagen_type}")
        
        # 处理辅星和杂曜，并分配文昌文曲的四化
        for star in current_scope_stars_in_palace:
            star_name = star['name']
            # 根据星曜类型进行分类（'soft', 'lucun', 'tough', 'tianma' 归类为辅星，其他为杂曜）
            if star['type'] in ['soft', 'lucun', 'tough', 'tianma']: 
                minor_stars_scope.append(star_name)
                # 检查文昌文曲是否有运限引动的四化
                if star_name in ['文昌', '文曲'] and star_name in scope_mutagen_map:
                    mutagen_type = scope_mutagen_map[star_name]
                    mutagen_list_for_output.append(f"{star_name}化{mutagen_type}")
            else: 
                adjective_stars_scope.append(star_name)

        # 构建单个运限宫位的输出列表
        transformed_scope.append([
            ", ".join(major_stars_names_only) if major_stars_names_only else "", # 主星名称，如果为空则为""
            current_palace_earthly_branch,        # 固定地支
            palace_name_in_order + ('宫' if '宫' not in palace_name_in_order else ''), # 宫位名称 (固定顺序中的名称)
            ",".join(mutagen_list_for_output) if mutagen_list_for_output else "", # 四化星，如果为空则为""
            ", ".join(minor_stars_scope) if minor_stars_scope else "",            # 辅星（来自运限数据），如果为空则为""
            ", ".join(adjective_stars_scope) if adjective_stars_scope else ""      # 杂曜（来自运限数据），如果为空则为""
        ])
    
    return transformed_scope


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

def _extract_first_json_object(s: str) -> str | None:
    """
    通过括号平衡，从字符串中精确提取第一个完整的JSON对象。

    Args:
        s: 包含JSON的字符串。

    Returns:
        第一个完整JSON对象的字符串，如果找不到则返回None。
    """
    try:
        # 寻找JSON的起始位置
        first_brace = s.find('{')
        first_bracket = s.find('[')

        if first_brace == -1 and first_bracket == -1:
            return None

        if first_brace == -1:
            start_pos = first_bracket
            start_char = '['
            end_char = ']'
        elif first_bracket == -1:
            start_pos = first_brace
            start_char = '{'
            end_char = '}'
        else:
            if first_brace < first_bracket:
                start_pos = first_brace
                start_char = '{'
                end_char = '}'
            else:
                start_pos = first_bracket
                start_char = '['
                end_char = ']'

        # 从起始位置开始扫描
        json_candidate = s[start_pos:]

        depth = 0
        in_string = False

        for i, char in enumerate(json_candidate):
            if char == '"' and (i == 0 or json_candidate[i - 1] != '\\'):
                in_string = not in_string

            if not in_string:
                if char == start_char or char == '[' or char == '{':
                    depth += 1
                elif char == end_char or char == ']' or char == '}':
                    depth -= 1

            if depth == 0:
                # 找到了一个完整的、括号平衡的对象
                return json_candidate[:i + 1]

    except Exception:
        # 如果在提取过程中发生任何意外，则返回None
        return None

    # 如果循环结束，depth不为0，说明对象本身被截断
    return None


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


import re

from datetime import datetime
def simple_clean_query_intent(raw_output: dict) -> dict:
    """
    对LLM返回的查询意图字典进行简单、高成功率的清洗。
    """
    if not isinstance(raw_output, dict):
        return {}

    cleaned = raw_output.copy()

    # 1. 修复常见的意图和分析级别别名/缩写
    intent_map = {
        'birth_chart': 'birth_chart_analysis',
        'horoscope': 'horoscope_analysis',
        'general': 'general_question',
        'missing': 'missing_birth_info'
    }
    analysis_level_map = {
        'birth': 'birth_chart',
        'decade': 'decadal'  # 'decadal' 的常见拼写错误
    }

    if isinstance(cleaned.get('intent_type'), str) and cleaned['intent_type'] in intent_map:
        cleaned['intent_type'] = intent_map[cleaned['intent_type']]

    if isinstance(cleaned.get('analysis_level'), str) and cleaned['analysis_level'] in analysis_level_map:
        cleaned['analysis_level'] = analysis_level_map[cleaned['analysis_level']]

    # 2. 修复错位的复合日期字符串
    # 检查 target_year, target_month, target_day 是否被误用
    for field in ['target_year', 'target_month', 'target_day']:
        value = cleaned.get(field)
        if isinstance(value, str):
            try:
                # 尝试将这个字符串解析为日期
                dt_obj = datetime.strptime(value, '%Y-%m-%d')
                # 如果成功，用正确的值覆盖
                cleaned['target_year'] = dt_obj.year
                cleaned['target_month'] = dt_obj.month
                cleaned['target_day'] = dt_obj.day
                # 清理被误用的字段
                if field not in ['target_year', 'target_month', 'target_day']:
                    cleaned[field] = None
                # 通常，如果LLM这样返回，它可能没有正确设置 resolved_horoscope_date
                if cleaned.get('resolved_horoscope_date') is None:
                    cleaned['resolved_horoscope_date'] = dt_obj.strftime('%Y-%m-%d %H:%M')
                break  # 修复一次即可
            except (ValueError, TypeError):
                pass  # 解析失败则忽略

    # 3. 确保数值字段是正确的类型或 None
    for field in ['target_year', 'target_month', 'target_day', 'target_hour', 'target_minute']:
        value = cleaned.get(field)
        if isinstance(value, str) and not value.isdigit():
            cleaned[field] = None
        elif isinstance(value, str) and value.isdigit():
            cleaned[field] = int(value)

    return cleaned



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



def is_valid_datetime_string(dt_str: Optional[str]) -> bool:
    """检查字符串是否是 'YYYY-MM-DD HH:MM:SS' 格式的有效日期时间。"""
    if not dt_str or not isinstance(dt_str, str):
        return False
    # 使用正则表达式进行初步、快速的格式检查
    if not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}(:\d{2})?$', dt_str.strip()):
        return False
    # 尝试将其转换为 datetime 对象以进行最终验证
    try:
        # 兼容 YYYY-MM-DD HH:MM 和 YYYY-MM-DD HH:MM:SS
        if len(dt_str.strip()) == 16: # YYYY-MM-DD HH:MM
             datetime.strptime(dt_str.strip(), '%Y-%m-%d %H:%M')
        else: # YYYY-MM-DD HH:MM:SS
             datetime.strptime(dt_str.strip(), '%Y-%m-%d %H:%M:%S')
        return True
    except ValueError:
        return False


def validate_ziwei_payload(payload: dict):
    """
    在发送前验证紫微API的payload。
    如果验证失败，会抛出 ValueError。
    """
    # 检查 dateStr
    date_str = payload.get('dateStr')
    if not date_str or not isinstance(date_str, str):
        raise ValueError("payload中缺少或无效的'dateStr'字段")

    # 使用正则表达式严格匹配 YYYY-MM-DD 格式
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        raise ValueError(f"dateStr格式不正确，应为YYYY-MM-DD，实际为: '{date_str}'")

    # 进一步验证日期是否真实存在
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"dateStr虽然格式正确，但日期无效: '{date_str}'")

    # 检查 timeIndex
    time_index = payload.get('timeIndex')
    if not isinstance(time_index, int) or not (0 <= time_index <= 12):
        raise ValueError(f"timeIndex必须是0-12之间的整数，实际为: {time_index}")

    # 检查 gender
    gender = payload.get('gender')
    if gender not in ['男', '女']:
        raise ValueError(f"gender必须是'男'或'女'，实际为: '{gender}'")

    # 检查 horoscopeDate (如果存在)
    horoscope_date = payload.get('horoscopeDate')
    if horoscope_date is not None:
        if not isinstance(horoscope_date, str) or not re.match(r'^\d{4}-\d{2}-\d{2}$', horoscope_date):
            raise ValueError(f"horoscopeDate格式不正确，应为YYYY-MM-DD，实际为: '{horoscope_date}'")
        try:
            datetime.strptime(horoscope_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"horoscopeDate虽然格式正确，但日期无效: '{horoscope_date}'")


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


from lunardate import LunarDate
# 【核心修正】: 从库的内部模块导入正确的工具
from datetime import datetime
import traceback
from datetime import datetime, timedelta


def to_chinese_month_name(month_number: int) -> str:
    """
    根据农历月份数字（可能为负，代表闰月）返回中文名称。
    """
    month_map = {
        1: '正月', 2: '二月', 3: '三月', 4: '四月', 5: '五月', 6: '六月',
        7: '七月', 8: '八月', 9: '九月', 10: '十月', 11: '冬月', 12: '腊月'
    }
    if month_number < 0:
        return f"闰{month_map[abs(month_number)]}"
    else:
        return month_map.get(month_number, "")


def get_lunar_month_range_string(dt_obj: datetime) -> str:
    """
    【V9版 - 完全自主逻辑，不再依赖任何不确定的库API】
    """
    try:
        # 1. 将公历日期转换为一个基准农历日期对象
        base_lunar_date = LunarDate.fromSolarDate(dt_obj.year, dt_obj.month, dt_obj.day)

        target_lunar_year = base_lunar_date.year
        target_lunar_month = base_lunar_date.month

        # 2. 创建本农历月的第一天的对象
        first_day_lunar = LunarDate(target_lunar_year, target_lunar_month, 1)

        # 3. 计算下个月的第一天
        next_month_year = target_lunar_year
        next_month_num = abs(target_lunar_month) + 1

        if next_month_num > 12:
            next_month_num = 1
            next_month_year += 1

        next_month_first_day_lunar = LunarDate(next_month_year, next_month_num, 1)

        # 4. 将下个月第一天转换回公历，然后减去一天，得到本月的最后一天
        next_month_first_day_solar = next_month_first_day_lunar.toSolarDate()
        last_day_solar = next_month_first_day_solar - timedelta(days=1)

        # 5. 获取本月第一天的公历日期
        first_day_solar = first_day_lunar.toSolarDate()

        # 6. 【核心修正】: 使用我们自己的函数来获取中文月份名称
        lunar_month_chinese_name = to_chinese_month_name(target_lunar_month)

        # 7. 组装最终的描述字符串
        return (
            #f"针对“{dt_obj.strftime('%-m月')}”（{lunar_month_chinese_name}，"
            f"针对“{dt_obj.strftime('%-m月')}”，"
            f"即{first_day_solar.strftime('%-m月%d日')}至{last_day_solar.strftime('%-m月%d日')}"
        )
    except Exception as e:
        print(f"Error calculating lunar month range: {e}\n{traceback.format_exc()}")
        return f"针对公历{dt_obj.year}年{dt_obj.month}月"


from typing import List, Dict, Tuple, Optional

YANG_GAN = ['甲', '丙', '戊', '庚', '壬']
YIN_GAN = ['乙', '丁', '己', '辛', '癸']

def calculate_decadal_start_age_exact(wuxingju: str) -> int:
    """
    根据五行局计算大运开始的精确年龄。
    """
    if "水二局" in wuxingju:
        return 2
    if "木三局" in wuxingju:
        return 3
    if "金四局" in wuxingju:
        return 4
    if "土五局" in wuxingju:
        return 5
    if "火六局" in wuxingju:
        return 6
    return 6 # 默认值

def calculate_all_decadal_periods(
        birth_year: int,
        gender: str,
        year_gan: str,  # 例如 "阳男", "阴女"
        wuxingju: str,
        palaces: List[Dict]  # API返回的完整宫位列表
) -> Optional[Dict[str, Tuple[int, int]]]:
    """
    计算所有大限的干支及其对应的年龄范围。
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

        start_age_first_decadal = calculate_decadal_start_age_exact(wuxingju)


        ming_gong_index = -1
        for i, palace in enumerate(palaces):
            if palace.get("name") == "命宫":
                ming_gong_index = i
                break

        if ming_gong_index == -1:
            return None

        decadal_periods = {}
        current_age = start_age_first_decadal

        # 循环12个宫位来确定12个大限
        for i in range(12):
            index = 0
            if is_forward:
                index = (ming_gong_index + i) % 12
            else:
                index = (ming_gong_index - i + 12) % 12

            palace_info = palaces[index]
            decadal_stem = palace_info.get("decadal", {}).get("heavenlyStem")
            decadal_branch = palace_info.get("decadal", {}).get("earthlyBranch")

            if decadal_stem and decadal_branch:
                decadal_gan_zhi = f"{decadal_stem}{decadal_branch}"
                start_age = current_age
                end_age = current_age + 9
                decadal_periods[decadal_gan_zhi] = (start_age, end_age)
                current_age += 10

        return decadal_periods
    except Exception as e:
        print(f"Error calculating all decadal periods: {e}")
        return None

from typing import Dict, Any, List, Literal, Optional, AsyncGenerator
from typing import List, Dict, Tuple, Optional,Union


def calculate_evidence_score(
        data: Union[Dict, List],
        positive_map: Dict[str, float],
        negative_map: Dict[str, float],
        _depth: int = 0  # 新增一个内部参数，用于追踪递归深度，方便看日志
) -> float:
    """
    【V3版 - 详细日志调试】
    递归地遍历JSON数据结构，并打印每一步的计分过程。
    """
    net_score = 0.0
    # 日志缩进，方便观察递归层次
    indent = "  " * _depth

    # 如果当前数据是字典
    if isinstance(data, dict):
        # logger.debug(f"{indent}递归字典，键: {list(data.keys())}")
        for key, value in data.items():
            # 我们只对 '宫位信息' 和 '关键信息汇总' 这两个特定字段的值进行计分
            if key in ["宫位信息", "关键信息汇总"]:
                # logger.debug(f"{indent}找到目标键: '{key}'，准备深入分析其值...")
                # 递归调用来处理值
                net_score += calculate_evidence_score(value, positive_map, negative_map, _depth + 1)
            else:
                # 如果不是目标字段，但其值是字典或列表，我们仍然需要递归下去
                if isinstance(value, (dict, list)):
                    net_score += calculate_evidence_score(value, positive_map, negative_map, _depth + 1)

    # 如果当前数据是列表
    elif isinstance(data, list):
        # logger.debug(f"{indent}递归列表，长度: {len(data)}")
        for i, item in enumerate(data):
            # logger.debug(f"{indent}  处理列表项 #{i}...")
            net_score += calculate_evidence_score(item, positive_map, negative_map, _depth + 1)

    # 如果当前数据是字符串，执行精确的关键词计分
    elif isinstance(data, str):
        # logger.debug(f"{indent}分析字符串: '{data[:80]}...'") # 只打印前80个字符，避免日志过长

        # 计算正面分数
        for keyword, score in positive_map.items():
            if keyword in data:
                net_score += score
                # 【核心调试日志】: 打印每一次加分
                logger.info(
                    f"【加分项】在 '{data[:30]}...' 中找到关键词 '{keyword}'，加分: {score} -> 当前净分: {net_score:.2f}")

        # 计算负面分数
        for keyword, score in negative_map.items():
            if keyword in data:
                net_score += score  # score 本身是负数
                # 【核心调试日志】: 打印每一次减分
                logger.info(
                    f"【减分项】在 '{data[:30]}...' 中找到关键词 '{keyword}'，加分: {score} -> 当前净分: {net_score:.2f}")

    return net_score


def save_string_to_file(filename, content):
    """
    将给定的字符串内容保存到指定的文本文件中。

    Args:
        filename (str): 要创建或写入的文件名（包括路径，如果需要）。
        content (str): 要写入文件的字符串内容。
    """
    try:
        # 使用 'w' 模式打开文件，如果文件不存在则创建，如果存在则清空内容
        # 'utf-8' 编码确保可以正确处理中文字符
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"字符串已成功保存到文件: {filename}")
    except IOError as e:
        # 捕获文件操作相关的错误，并打印错误信息
        print(f"写入文件时发生错误: {e}")
    except Exception as e:
        # 捕获其他任何可能的错误
        print(f"发生未知错误: {e}")





VALID_XINGXI_DIZHI_COMBOS = {
"巨门,午","七杀,午","七杀,卯","七杀,子","七杀,寅","七杀,戌","七杀,申","七杀,辰",
"天同,卯","天同,巳","天同,戌","天同,辰","天同,酉","天同，天梁,卯","天同，天梁,寅",
"天同，天梁,申","天同，太阴,午","天同，太阴,子","天同，巨门,丑","天同，巨门,未",
"天同，巨门,申","天同，巨门,酉","天府,丑","天府,卯","天府,巳","天府,未","天府,酉",
"天机,丑","天机,亥","天机,午","天机,子","天机,巳","天机,未","天机，天梁,戌","天机，天梁,辰",
"天机，太阴,寅","天机，太阴,申","天机，巨门,卯","天机，巨门,酉","天梁,丑","天梁,亥","天梁,午",
"天梁,子","天梁,巳","天梁,未","天相,亥","天相,卯","天相,巳","天相,未","天相,酉","太阳,亥",
"太阳,午","太阳,子","太阳,巳","太阳,戌","太阳,未","太阳,辰","太阳，天梁,卯","太阳，天梁,酉",
"太阳，太阴,丑","太阳，太阴,未","太阳，巨门,寅","太阳，巨门,申","太阴,亥","太阴,卯","太阴,戌",
"太阴,辰","太阴,酉","巨门,亥","巨门,午","巨门,子","巨门,戌","巨门,辰","廉贞,寅","廉贞,申",
"廉贞，七杀,丑","廉贞，七杀,未","廉贞，天府,戌","廉贞，天府,辰","廉贞，天相,子","廉贞，破军,卯",
"廉贞，破军,酉","廉贞，贪狼,亥","廉贞，贪狼,巳","武曲,戌","武曲,辰","武曲，七杀,卯","武曲，七杀,酉",
"武曲，天府,午","武曲，天府,子","武曲，天相,寅","武曲，天相,申","武曲，破军,亥","武曲，破军,巳",
"武曲，贪狼,丑","武曲，贪狼,未","破军,午","破军,子","破军,寅","破军,戌","破军,申","破军,辰",
"紫微,午","紫微,子","紫微,未","紫微，七杀,亥","紫微，七杀,巳","紫微，天府,寅","紫微，天府,申",
"紫微，天相,戌","紫微，天相,辰","紫微，破军,丑","紫微，破军,未","紫微，贪狼,卯","紫微，贪狼,酉",
"贪狼,午","贪狼,子","贪狼,寅","贪狼,戌","贪狼,申"}
