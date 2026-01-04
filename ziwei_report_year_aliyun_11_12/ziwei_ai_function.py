import re
from typing import Dict, Any, List, Literal, Optional
import logging
import calendar

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

    # 将Prompt和LLM组合成一个链，并使用with_structured_output确保返回结构化数据
    # 这要求VLLM模型支持OpenAI兼容的函数调用或工具使用
    extraction_chain = extraction_prompt | llm.with_structured_output(BirthInfo)

    try:
        # 调用链来提取信息
        parsed_info_model = extraction_chain.invoke({"user_input": user_input})
        # 将Pydantic模型转换为字典，使用model_dump()解决Pydantic V2的兼容性警告
        print(f"LLM extracted raw birth info model: {parsed_info_model}")  # Debug print
        return parsed_info_model.model_dump()
    except Exception as e:
        print(
            f"Error extracting birth info with LLM: {e}. Please ensure your VLLM model supports structured output (e.g., OpenAI function call compatibility).")  # Changed error message
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

    # *** 关键修正：调整匹配优先级 ***
    # 1. 最精确的：带前缀且带分钟的模式
    # 2. 其次：只带前缀小时的模式
    # 3. 最后：纯小时的模式
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

    # --- 调试信息开始 ---
    print(f"用户输入: {user_input}")
    print(f"模式1 (带前缀和分钟) 匹配结果: {time_full_with_prefix_match}")
    print(f"模式2 (只带前缀小时) 匹配结果: {time_prefix_hour_only_match}")
    print(f"模式3 (纯小时) 匹配结果: {time_pure_hour_match}")
    print(f"最终选中的匹配类型: {match_type}")

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


# ALL_EARTHLY_BRANCHES,FIXED_PALACE_ORDER_FOR_SCOPES,HEAVENLY_STEM_MUTAGEN_MAP,get_ordered_palace_branches,get_mutagen_for_stem,transform_palace_data,transform_horoscope_scope_data

# 十二地支的固定顺序（逆时针）
ALL_EARTHLY_BRANCHES = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']

# 运势分析、流年、流月、流日、流时等宫位的固定顺序
# 这个顺序是相对于“命宫”的位置而言的。
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


def transform_horoscope_scope_data(scope_data, astrolabe_palaces_data):
    """
    转换运势分析（大限、流年、流月、流日、流时）数据。
    1. 根据运限自身的'earthlyBranch'（视为该运限的“命宫”所在），重新推导十二宫位对应的地支。
    2. 主星从命盘原始数据中该地支对应的宫位获取（并考虑对宫借星）。
    3. 应用运限自身的四化到这些主星/辅星上（文昌、文曲特殊处理）。
    4. 辅星和杂曜从运限数据自身的'stars'字段获取。
    输出格式: [主星, 地支, 宫位名称, 四化, 辅星, 杂曜]
    """
    transformed_scope = []

    # 创建命盘原始宫位的地支到宫位数据映射，用于查询主星
    original_palaces_by_branch = {p['earthlyBranch']: p for p in astrolabe_palaces_data}

    # 1. 确定当前运限的“命宫”地支 (直接使用scope_data中的earthlyBranch)
    scope_life_palace_branch = scope_data.get('earthlyBranch')

    if not scope_life_palace_branch:
        # 如果运限数据中没有earthlyBranch，则无法进行推导
        print(f"警告: 运限 '{scope_data.get('name', '未知')}' 数据中未找到earthlyBranch。")
        return []

    # 2. 根据运限的“命宫”地支，推导出该运限中十二宫位（按固定顺序）对应的地支
    ordered_earthly_branches = get_ordered_palace_branches(scope_life_palace_branch)

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
    for i, palace_name_in_order in enumerate(FIXED_PALACE_ORDER_FOR_SCOPES):
        current_palace_earthly_branch = ordered_earthly_branches[i]  # 获取该运限宫位对应的地支

        # 从命盘原始数据中，根据推导出的地支，找到对应的宫位数据
        original_palace_data_by_branch = original_palaces_by_branch.get(current_palace_earthly_branch)

        if not original_palace_data_by_branch:
            # 如果命盘中没有该地支对应的宫位数据（理论上不应发生，因为是12宫），跳过
            continue

            # 确定主星（来自命盘原始数据中该地支的宫位，如果为空则取对宫）
        effective_major_stars = []
        if not original_palace_data_by_branch['majorStars']:  # 如果命盘该地支的宫位主星为空
            opposing_earthly_branch = DIZHI_DUIGONG_MAP.get(current_palace_earthly_branch)
            if opposing_earthly_branch and opposing_earthly_branch in original_palaces_by_branch:
                opposing_palace = original_palaces_by_branch[opposing_earthly_branch]
                effective_major_stars = opposing_palace['majorStars']
        else:  # 否则使用命盘该地支的宫位的主星
            effective_major_stars = original_palace_data_by_branch['majorStars']

        # 获取当前运限宫位中的辅星和杂曜
        # 这些星曜直接来自 scope_data['stars'] 字段，并根据其 'type' 进行分类
        # 确保索引不越界，因为某些运限的stars数组可能长度不足12
        current_scope_stars_in_palace = scope_data['stars'][i] if i < len(scope_data['stars']) else []

        minor_stars_scope = []
        adjective_stars_scope = []

        # 提取主星名称，并根据运限的四化列表应用四化
        major_stars_names_only = []  # 存储纯粹的主星名称
        mutagen_list_for_output = []  # 存储运限四化星的列表 (例如: "太阳化禄")
        for star in effective_major_stars:
            star_name = star['name']
            major_stars_names_only.append(star_name)  # 添加纯粹的主星名称

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
            ", ".join(major_stars_names_only) if major_stars_names_only else "",  # 主星名称，如果为空则为""
            current_palace_earthly_branch,  # 推导出的地支
            palace_name_in_order + ('宫' if '宫' not in palace_name_in_order else ''),  # 宫位名称 (固定顺序中的名称)
            ",".join(mutagen_list_for_output) if mutagen_list_for_output else "",  # 四化星，如果为空则为""
            ", ".join(minor_stars_scope) if minor_stars_scope else "",  # 辅星（来自运限数据），如果为空则为""
            ", ".join(adjective_stars_scope) if adjective_stars_scope else ""  # 杂曜（来自运限数据），如果为空则为""
        ])
    return transformed_scope


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
                mutagen_map[star_name_or_mutagen] = mutagen_type  # 主星四化：键为星名，值为化气
            else:  # 文昌文曲的特殊处理，键为星名（文昌/文曲），值为化气（科/忌）
                mutagen_map[mutagen_type] = star_name_or_mutagen  # 修正：原代码此处的逻辑可能导致错误，应是星名作为键
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
        if not original_palace_data_by_branch['majorStars']:  # 如果命盘该地支的宫位主星为空
            opposing_earthly_branch = DIZHI_DUIGONG_MAP.get(current_palace_earthly_branch)
            # print(opposing_earthly_branch)
            if opposing_earthly_branch and opposing_earthly_branch in original_palaces_by_branch:
                opposing_palace = original_palaces_by_branch[opposing_earthly_branch]
                effective_major_stars = opposing_palace['majorStars']
                # print(effective_major_stars)
        else:  # 否则使用命盘该地支的宫位的主星
            effective_major_stars = original_palace_data_by_branch['majorStars']
            # print(current_palace_earthly_branch,effective_major_stars)

        # 获取当前运限宫位中的辅星和杂曜
        # 这些星曜直接来自 scope_data['stars'] 字段，并根据其 'type' 进行分类
        # 确保索引不越界，因为某些运限的stars数组可能长度不足12
        current_scope_stars_in_palace = scope_data['stars'][i] if i < len(scope_data['stars']) else []

        minor_stars_scope = []
        adjective_stars_scope = []

        # 提取主星名称，并根据运限的四化列表应用四化
        major_stars_names_only = []  # 存储纯粹的主星名称
        mutagen_list_for_output = []  # 存储运限四化星的列表 (例如: "太阳化禄")
        for star in effective_major_stars:
            star_name = star['name']
            major_stars_names_only.append(star_name)  # 添加纯粹的主星名称

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
            ", ".join(major_stars_names_only) if major_stars_names_only else "",  # 主星名称，如果为空则为""
            current_palace_earthly_branch,  # 固定地支
            palace_name_in_order + ('宫' if '宫' not in palace_name_in_order else ''),  # 宫位名称 (固定顺序中的名称)
            ",".join(mutagen_list_for_output) if mutagen_list_for_output else "",  # 四化星，如果为空则为""
            ", ".join(minor_stars_scope) if minor_stars_scope else "",  # 辅星（来自运限数据），如果为空则为""
            ", ".join(adjective_stars_scope) if adjective_stars_scope else ""  # 杂曜（来自运限数据），如果为空则为""
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
        if len(dt_str.strip()) == 16:  # YYYY-MM-DD HH:MM
            datetime.strptime(dt_str.strip(), '%Y-%m-%d %H:%M')
        else:  # YYYY-MM-DD HH:MM:SS
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
            # f"针对“{dt_obj.strftime('%-m月')}”（{lunar_month_chinese_name}，"
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
    return 6  # 默认值


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

        # 【核心修正】: 删除了下面这行错误的、残留的代码
        # is_forward = "阳男" in yin_yang_year or "阴女" in yin_yang_year

        # 找到命宫的索引
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
from typing import List, Dict, Tuple, Optional, Union


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
    "巨门,午", "七杀,午", "七杀,卯", "七杀,子", "七杀,寅", "七杀,戌", "七杀,申", "七杀,辰",
    "天同,卯", "天同,巳", "天同,戌", "天同,辰", "天同,酉", "天同，天梁,卯", "天同，天梁,寅",
    "天同，天梁,申", "天同，太阴,午", "天同，太阴,子", "天同，巨门,丑", "天同，巨门,未",
    "天同，巨门,申", "天同，巨门,酉", "天府,丑", "天府,卯", "天府,巳", "天府,未", "天府,酉",
    "天机,丑", "天机,亥", "天机,午", "天机,子", "天机,巳", "天机,未", "天机，天梁,戌", "天机，天梁,辰",
    "天机，太阴,寅", "天机，太阴,申", "天机，巨门,卯", "天机，巨门,酉", "天梁,丑", "天梁,亥", "天梁,午",
    "天梁,子", "天梁,巳", "天梁,未", "天相,亥", "天相,卯", "天相,巳", "天相,未", "天相,酉", "太阳,亥",
    "太阳,午", "太阳,子", "太阳,巳", "太阳,戌", "太阳,未", "太阳,辰", "太阳，天梁,卯", "太阳，天梁,酉",
    "太阳，太阴,丑", "太阳，太阴,未", "太阳，巨门,寅", "太阳，巨门,申", "太阴,亥", "太阴,卯", "太阴,戌",
    "太阴,辰", "太阴,酉", "巨门,亥", "巨门,午", "巨门,子", "巨门,戌", "巨门,辰", "廉贞,寅", "廉贞,申",
    "廉贞，七杀,丑", "廉贞，七杀,未", "廉贞，天府,戌", "廉贞，天府,辰", "廉贞，天相,子", "廉贞，破军,卯",
    "廉贞，破军,酉", "廉贞，贪狼,亥", "廉贞，贪狼,巳", "武曲,戌", "武曲,辰", "武曲，七杀,卯", "武曲，七杀,酉",
    "武曲，天府,午", "武曲，天府,子", "武曲，天相,寅", "武曲，天相,申", "武曲，破军,亥", "武曲，破军,巳",
    "武曲，贪狼,丑", "武曲，贪狼,未", "破军,午", "破军,子", "破军,寅", "破军,戌", "破军,申", "破军,辰",
    "紫微,午", "紫微,子", "紫微,未", "紫微，七杀,亥", "紫微，七杀,巳", "紫微，天府,寅", "紫微，天府,申",
    "紫微，天相,戌", "紫微，天相,辰", "紫微，破军,丑", "紫微，破军,未", "紫微，贪狼,卯", "紫微，贪狼,酉",
    "贪狼,午", "贪狼,子", "贪狼,寅", "贪狼,戌", "贪狼,申"}

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


def calculate_score_by_rules1(current_liu_nian_palace, liu_nian_parsed, da_xian_parsed):
    """
    (V26 - 增加负分总和统计)
    - 新增功能: 在函数内初始化一个 `negative_score_sum` 变量，用于累计所有减分项。
    - 在每次 `score` 减少的地方，同步将减分值累加到 `negative_score_sum`。
    - 在最终的 `return` 语句中，增加 `negative_score_sum` 作为新的输出项。
    - 维持V25的所有计分逻辑。
    """
    # 【改动 1】: 初始化总分、事件列表和新增的负分总和变量
    score, events = 0.0, []
    negative_score_sum = 0.0  # <--- 新增变量，用于累计所有负分

    current_di_zhi, current_palace_name = current_liu_nian_palace['di_zhi'], current_liu_nian_palace['palace_name']

    # --- 1. 数据准备 (此部分无改动) ---
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

    def record_event_with_level(reason, score_val, event_type, context, interaction_level, **kwargs):
        kwargs['intensity_desc'] = INTERACTION_INTENSITY_DESC.get(interaction_level, "强度未知")
        record_event(reason, score_val, event_type, context, events, **kwargs)

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
                for required_sihua, combo_name, combo_score_val in SIHUA_GOOD_COMBOS:
                    if required_sihua.issubset(combined_sihua_types) and \
                            not sihua_types1.isdisjoint(required_sihua) and \
                            not sihua_types2.isdisjoint(required_sihua):
                        interaction_sihua_score += combo_score_val
                        record_event_with_level(f"由 [{get_name(k1)} 与 {get_name(k2)}] 互动形成: {combo_name}",
                                                combo_score_val, "sihua_good_combo", f"{get_source_key(k1, k2)}", level,
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
                neg_score = -1.5
                interaction_sihua_score += neg_score
                # 【改动 2】: 累加负分
                negative_score_sum += neg_score  # <--- 累加
                record_event_with_level(f"'化忌'({get_name(k1)})与'化忌'({get_name(k2)})重叠", neg_score,
                                        "sihua_overlap",
                                        f"{get_source_key(k1, k2)}", level, sihua_pair=('化忌', '化忌'),
                                        location1=get_name(k1), location2=get_name(k2))

        # --- 2.2 星曜层面计分 ---
        coefficient = get_interaction_coefficient(k1, k2)
        stars1, stars2 = set(P[k1]['all_stars']), set(P[k2]['all_stars'])
        for star in stars1.intersection(stars2):
            if star in GOOD_STARS or star in BAD_STARS:
                final_s = (BASE_SCORE_STAR if star in GOOD_STARS else -BASE_SCORE_STAR) * coefficient
                interaction_star_score += final_s
                if final_s < 0:
                    # 【改动 3】: 累加负分
                    negative_score_sum += final_s  # <--- 累加
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
                if final_s < 0:
                    # 【改动 4】: 累加负分
                    negative_score_sum += final_s  # <--- 累加
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
            s = BASE_SCORE_STAR if is_good else -BASE_SCORE_STAR
            score += s
            if s < 0:
                # 【改动 5】: 累加负分
                negative_score_sum += s  # <--- 累加
            record_event_with_level(f"本宫内自带星曜对: {sorted(list(pair))}", s,
                                    "good_pair" if is_good else "bad_pair", "本宫自身性质", "ln_internal", pair=pair,
                                    location1=get_name("ln_本宫"))
            processed_stars_in_self.update(pair)

    ln_sihua_types = {get_si_hua_type(si) for si in P['ln_本宫']['si_hua']}
    if '化忌' not in ln_sihua_types:
        SIHUA_GOOD_COMBOS_SELF = [({"化禄", "化权", "化科"}, "三奇加会(禄权科)", 2.0),
                                  ({"化禄", "化权"}, "禄权交会", 2.0), ({"化禄", "化科"}, "禄科交会", 2.0),
                                  ({"化权", "化科"}, "权科交会", 2.0)]
        for required_sihua, combo_name, combo_score_val in SIHUA_GOOD_COMBOS_SELF:
            if required_sihua.issubset(ln_sihua_types):
                score += combo_score_val
                record_event_with_level(f"本宫内自带吉格: {combo_name}", combo_score_val, "sihua_good_combo",
                                        "本宫自身性质", "ln_internal", combo_name=combo_name,
                                        sihua_members=list(required_sihua))
                break
    if any(s in GOOD_SI_HUA_TYPES for s in ln_sihua_types): score += 1.0; record_event_with_level("流年本宫有吉化", 1.0,
                                                                                                  "single_good",
                                                                                                  "本宫自身性质",
                                                                                                  "ln_internal",
                                                                                                  source="吉化")
    if any(s in BAD_SI_HUA_TYPES for s in ln_sihua_types):
        neg_score = -1.0
        score += neg_score
        # 【改动 6】: 累加负分
        negative_score_sum += neg_score  # <--- 累加
        record_event_with_level("流年本宫有化忌", neg_score,
                                "single_bad",
                                "本宫自身性质",
                                "ln_internal",
                                source="化忌")

    for star in P['ln_本宫']['aux_stars']:
        if star in processed_stars_in_self: continue
        if star in GOOD_STARS or star in BAD_STARS:
            s = 1.0 if star in GOOD_STARS else -1.0
            score += s
            if s < 0:
                # 【改动 7】: 累加负分
                negative_score_sum += s  # <--- 累加
            record_event_with_level(f"流年本宫有{'吉' if s > 0 else '煞'}辅星'{star}'", s,
                                    "single_good" if s > 0 else "single_bad", "本宫自身性质", "ln_internal",
                                    source=star)

    summary_dict = generate_qualitative_summary(current_palace_name, score, events)
    filtered_source_scores = {k: round(v, 2) for k, v in source_scores.items() if v != 0.0}

    # 【改动 8】: 在函数的返回元组中增加 negative_score_sum
    return score, filtered_source_scores, summary_dict, negative_score_sum


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


def transform_palace_data_new(astrolabe_data):
    """
    转换命盘宫位数据。
    如果当前宫位主星为空，则填充其对宫的主星和地支。
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

        major_stars_to_use = []
        earthly_branch_to_use = palace['earthlyBranch']

        if not current_major_stars:  # 如果当前宫位主星为空
            current_earthly_branch = palace['earthlyBranch']
            opposing_earthly_branch = DIZHI_DUIGONG_MAP.get(current_earthly_branch)

            if opposing_earthly_branch and opposing_earthly_branch in earthly_branch_to_palace:
                opposing_palace = earthly_branch_to_palace[opposing_earthly_branch]
                major_stars_to_use = opposing_palace['majorStars']

                earthly_branch_to_use = opposing_earthly_branch
        else:  # 如果当前宫位有主星，则直接使用
            major_stars_to_use = current_major_stars

        # 提取主星名称和命盘四化
        major_stars_names = []
        mutagen_list_for_output = []  # 存储四化星的列表

        # 处理主星的四化 (基于最终要使用的主星)
        for s in major_stars_to_use:
            star_name = s['name']
            major_stars_names.append(star_name)

            # 检查主星是否有命盘年干引动的四化，并排除文昌文曲
            if star_name in astrolabe_mutagen_map and star_name not in ['文昌', '文曲']:
                mutagen_type = astrolabe_mutagen_map[star_name]
                mutagen_list_for_output.append(f"{star_name}化{mutagen_type}")

        # 提取辅星和杂曜（这些星曜始终来自命盘的原始当前宫位）
        minor_stars_list = []
        # 处理辅星的四化 (文昌、文曲)
        for s in current_minor_stars:
            star_name = s['name']
            minor_stars_list.append(star_name)
            # 检查辅星（文昌、文曲）是否有命盘年干引动的四化
            if star_name in ['文昌', '文曲'] and star_name in astrolabe_mutagen_map:
                mutagen_type = astrolabe_mutagen_map[star_name]
                mutagen_list_for_output.append(f"{star_name}化{mutagen_type}")

        adjective_stars_list = [s['name'] for s in palace['adjectiveStars']]

        # 构建单个宫位的输出列表
        transformed_palaces.append([
            ", ".join(major_stars_names) if major_stars_names else "",  # 主星名称
            earthly_branch_to_use,  # ★ 使用可能已更新的地支
            palace['name'] + ('宫' if '宫' not in palace['name'] else ''),  # 宫位名称
            ",".join(mutagen_list_for_output) if mutagen_list_for_output else "",  # 四化星
            ", ".join(minor_stars_list) if minor_stars_list else "",  # 辅星
            ", ".join(adjective_stars_list) if adjective_stars_list else ""  # 杂曜
        ])
    return transformed_palaces
