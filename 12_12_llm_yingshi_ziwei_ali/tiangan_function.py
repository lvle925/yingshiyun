import pandas as pd
wildcard_match_cols = ['年干', '年支', '月支', '时辰']
def get_matching_phenomena(row_combinations, grouped_ziwei_df):
    """
    根据匹配规则，从grouped_ziwei中查找并合并现象。

    参数:
    row_combinations (pd.Series): combinations_df中的一行数据。
    grouped_ziwei_df (pd.DataFrame): grouped_ziwei表。

    返回:
    str: 合并后的现象字符串，如果无匹配则返回原始现象。
    """
    matched_phenomena = []

    # 首先根据精确匹配的列进行筛选，提高效率
    filtered_grouped_ziwei = grouped_ziwei_df[
        (grouped_ziwei_df['星系'] == row_combinations['星系']) &
        (grouped_ziwei_df['地支'] == row_combinations['地支']) &
        (grouped_ziwei_df['宫位'] == row_combinations['宫位'])
        ].copy()  # 使用.copy()避免SettingWithCopyWarning

    # 遍历筛选后的grouped_ziwei行，进行通配符匹配
    for _, row_grouped in filtered_grouped_ziwei.iterrows():
        is_match = True
        for col in wildcard_match_cols:
            # 如果grouped_ziwei中的列值为NaN，则视为通配符，匹配成功。
            # 否则，必须与combinations_df中的对应列值精确匹配。
            if pd.notna(row_grouped[col]) and row_grouped[col] != row_combinations[col]:
                is_match = False
                break

        if is_match:
            # 将匹配到的现象添加到列表中
            matched_phenomena.append(row_grouped['现象'])

    # 如果有匹配到的现象，则用“。”连接并返回，否则返回combinations_df中原始的现象值
    return '。'.join(matched_phenomena) if matched_phenomena else row_combinations['现象']


def parse_bazi_components(bazi_dict):
    """
    Parses a dictionary containing Bazi components (stems and branches).

    Args:
        bazi_dict (dict): A dictionary with keys 'yearly', 'monthly', 'daily', 'hourly',
                          each containing a list of two elements [stem, branch].

    Returns:
        dict: A dictionary with clearly labeled stems and branches,
              or prints the information if no return is needed.
    """
    parsed_info = {
        "year_stem": bazi_dict['yearly'][0],
        "year_branch": bazi_dict['yearly'][1],
        "month_stem": bazi_dict['monthly'][0],
        "month_branch": bazi_dict['monthly'][1],
        "day_stem": bazi_dict['daily'][0],
        "day_branch": bazi_dict['daily'][1],
        "hour_stem": bazi_dict['hourly'][0],
        "hour_branch": bazi_dict['hourly'][1]
    }
    return parsed_info


import json


def replace_key_names(data, mapping):
    """递归替换字典中的键名"""
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            # 先处理值中的嵌套结构
            new_value = replace_key_names(v, mapping)
            # 替换当前键名
            new_key = mapping.get(k, k)
            new_dict[new_key] = new_value
        return new_dict
    elif isinstance(data, list):
        return [replace_key_names(item, mapping) for item in data]
    else:
        return data


# 创建字段映射表
key_mapping = {
    # 顶层字段
    "soul": "命主",
    "body": "身主",
    "fiveElementsClass": "五行局",
    "palaces": "宫位数组数据",

    # 宫位字段
    "index": "宫位索引",
    "name": "宫位名称",
    "isBodyPalace": "是否身宫",
    "isOriginalPalace": "是否莱因宫",
    "heavenlyStem": "宫位天干",
    "earthlyBranch": "宫位地支",

    # 四化字段
    "monthlyMutagen": "生月四化",
    "dailyMutagen": "生日四化",
    "hourlyMutagen": "生时四化",
    "yearlyMutagen": "生年四化",
    "Mutagen": "原局四化",

    # 主星结构
    "majorStars": "主星数组",
    "minorStars": "辅星数组",
    "adjectiveStars": "杂曜数组",

    # 通用星曜字段
    "name": "名称",
    "type": "类型",
    "tscope": "作用范围",
    "brightness": "亮度",

    # 主星专用字段
    "majorStars:name": "主星名称",
    "majorStars:type": "主星类型",

    # 辅星专用字段
    "minorStars:name": "辅星名称",
    "minorStars:type": "辅星类型",

    # 杂曜专用字段
    "adjectiveStars:name": "杂曜名称",
    "adjectiveStars:type": "杂曜类型",

    # 流年神煞
    "changsheng12": "长生12神",
    "boshi12": "博士12神",
    "jiangqian12": "将前12神",
    "suiqian12": "岁前12神"
}

# 使用示例


# with open('转换后数据.json', 'w', encoding='utf-8') as f:
#    json.dump(converted_data, f, ensure_ascii=False, indent=2)
def zhuxing1(data):
    a = ""
    for i in data:
        try:
            a = a+(i["名称"])+","
        except Exception as e:
            Exception
    return a

def ziweids2(converted_data):
    results_array = []  # 初始化一个空列表来存储结果

    for i in range(0, 12):
        # 主星 (Main Stars)
        # 假设 zhuxing1 是一个提取主星的函数，可能返回一个空列表
        ans1 = zhuxing1(converted_data['data']['astrolabe']['宫位数组数据'][i]['主星数组'])

        # 如果主星数组为空，则尝试从对宫（i+6）获取主星
        if len(ans1) == 0:
            # 处理索引循环，确保在 0-11 范围内
            if i + 6 > 11:
                ans1 = zhuxing1(converted_data['data']['astrolabe']['宫位数组数据'][i + 6 - 12]['主星数组'])
            else:
                ans1 = zhuxing1(converted_data['data']['astrolabe']['宫位数组数据'][i + 6]['主星数组'])

        # 地支 (Earthly Branch)
        ans2 = converted_data['data']['astrolabe']['宫位数组数据'][i]['宫位地支']

        # 宫位 (Palace Name)
        ans3 = converted_data['data']['astrolabe']['宫位数组数据'][i]['名称']

        # 将 ans1, ans2, 和 ans3 作为列表（或元组）添加到 results_array 中
        # 每个内部列表代表一个宫位的数据：[主星, 地支, 宫位名称]
        results_array.append([ans1[:-1].replace(",","，"),ans2, ans3+"宫"])

    return results_array


import pandas as pd
from typing import Dict, Any, List

from typing import Dict, Any, List


def describe_ziwei_chart(chart_data: List[List[str]]) -> str:
    """
    根据紫微斗数星盘数据生成一段描述，包含每个宫位的三方四正和夹宫信息。

    Args:
        chart_data: 一个列表，其中每个元素是一个宫位的详细信息列表，
                    格式为: [主星, 地支, 宫位名称, 化忌/化权等信息, 辅星/杂耀]
                    示例: ['廉贞，天府', '辰', '命宫', '廉贞化忌', '陀罗,红鸾']

    Returns:
        一段描述紫微斗数星盘的文本。
    """

    description_parts = []

    # --- 1. 数据预处理与宫位索引构建 ---
    # 标准化宫位名称，并创建宫位到其数据项的映射
    # 同时构建地支到标准化宫位名称的映射，方便通过地支查找宫位
    # 以及宫位名称到地支的映射，方便通过宫位名称查找地支
    gong_wei_data_map: Dict[str, List[str]] = {}  # 宫位名称 -> 完整数据项
    di_zhi_to_gong_wei_map: Dict[str, str] = {}  # 地支 -> 宫位名称
    gong_wei_to_di_zhi_map: Dict[str, str] = {}  # 宫位名称 -> 地支

    # 为了计算夹宫和三方四正，我们需要知道所有宫位的地支以及它们之间的顺序关系。
    # 斗数盘的十二宫位是固定的，地支也是固定的顺序。
    # 我们先建立一个按地支顺序排列的宫位列表，方便查找相邻宫位。
    # 宫位地支顺序（顺时针）
    DI_ZHI_ORDER = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]

    # 建立地支到其在 DI_ZHI_ORDER 中索引的映射
    di_zhi_index_map = {di_zhi: idx for idx, di_zhi in enumerate(DI_ZHI_ORDER)}

    # 填充上述映射
    for item in chart_data:
        # 修正宫位名称，统一为常见名称
        cleaned_gong_wei = item[2].replace('宫宫', '宫').replace('官禄宫', '事业宫').replace('仆役宫', '交友宫')
        gong_wei_data_map[cleaned_gong_wei] = item
        di_zhi_to_gong_wei_map[item[1]] = cleaned_gong_wei
        gong_wei_to_di_zhi_map[cleaned_gong_wei] = item[1]  # 存储宫位名称到地支的映射

    # --- 2. 辅助函数：计算三方四正 ---
    def get_related_palaces(current_di_zhi: str) -> Dict[str, str]:
        """根据当前地支获取其三方四正的地支及对应宫位名称。"""
        if current_di_zhi not in DI_ZHI_ORDER:
            return {'对宫': '未知宫位', '三方一': '未知宫位', '三方二': '未知宫位'}

        current_idx = di_zhi_index_map[current_di_zhi]

        # 对宫（六合位，相隔6个地支）
        opposite_idx = (current_idx + 6) % 12
        opposite_di_zhi = DI_ZHI_ORDER[opposite_idx]

        # 三方（三合位，相隔4个和8个地支）
        san_fang_1_idx = (current_idx + 4) % 12  # 逆时针数第四个
        san_fang_1_di_zhi = DI_ZHI_ORDER[san_fang_1_idx]

        san_fang_2_idx = (current_idx + 8) % 12  # 顺时针数第四个（或逆时针数第八个）
        san_fang_2_di_zhi = DI_ZHI_ORDER[san_fang_2_idx]

        related_palaces = {}
        # 查找对应的宫位名称，如果找不到则返回 '未知宫位'
        related_palaces['对宫'] = di_zhi_to_gong_wei_map.get(opposite_di_zhi, '未知宫位')
        related_palaces['三方一'] = di_zhi_to_gong_wei_map.get(san_fang_1_di_zhi, '未知宫位')
        related_palaces['三方二'] = di_zhi_to_gong_wei_map.get(san_fang_2_di_zhi, '未知宫位')

        return related_palaces

    # --- 3. 辅助函数：检查夹宫 ---
    def check_clamping(current_gong_wei: str) -> List[str]:
        """
        检查给定宫位是否存在羊陀夹、火铃夹或天梁化忌夹。
        返回一个包含夹宫类型描述的列表。
        """
        clamping_types = []

        current_di_zhi = gong_wei_to_di_zhi_map.get(current_gong_wei)
        if not current_di_zhi:
            return clamping_types

        current_idx = di_zhi_index_map[current_di_zhi]

        # 计算前一宫和后一宫的地支
        prev_idx = (current_idx - 1 + 12) % 12  # 逆时针上一宫
        next_idx = (current_idx + 1) % 12  # 顺时针下一宫

        prev_di_zhi = DI_ZHI_ORDER[prev_idx]
        next_di_zhi = DI_ZHI_ORDER[next_idx]

        # 获取前后宫位的名称和星曜信息
        prev_gong_wei_name = di_zhi_to_gong_wei_map.get(prev_di_zhi)
        next_gong_wei_name = di_zhi_to_gong_wei_map.get(next_di_zhi)

        prev_stars = ""
        next_stars = ""
        prev_hua_info = ""
        next_hua_info = ""

        if prev_gong_wei_name and prev_gong_wei_name in gong_wei_data_map:
            prev_item = gong_wei_data_map[prev_gong_wei_name]
            prev_stars = (prev_item[0] + "," + prev_item[4]).replace('，', ',').strip(',').replace(',,', ',')
            prev_hua_info = prev_item[3]

        if next_gong_wei_name and next_gong_wei_name in gong_wei_data_map:
            next_item = gong_wei_data_map[next_gong_wei_name]
            next_stars = (next_item[0] + "," + next_item[4]).replace('，', ',').strip(',').replace(',,', ',')
            next_hua_info = next_item[3]

        # 将星曜字符串转换为集合，方便检查是否存在
        prev_stars_set = set(s.strip() for s in prev_stars.split(',') if s.strip())
        next_stars_set = set(s.strip() for s in next_stars.split(',') if s.strip())

        # 1. 羊陀夹 (擎羊 & 陀罗)
        is_qian_yang_prev = "擎羊" in prev_stars_set
        is_tuo_luo_next = "陀罗" in next_stars_set
        is_tuo_luo_prev = "陀罗" in prev_stars_set
        is_qian_yang_next = "擎羊" in next_stars_set

        if (is_qian_yang_prev and is_tuo_luo_next) or \
                (is_tuo_luo_prev and is_qian_yang_next):
            clamping_types.append(f"被【擎羊】和【陀罗】夹（羊陀夹），可能预示着阻碍、纠缠或不顺。")

        # 2. 火铃夹 (火星 & 铃星)
        is_huo_xing_prev = "火星" in prev_stars_set
        is_ling_xing_next = "铃星" in next_stars_set
        is_ling_xing_prev = "铃星" in prev_stars_set
        is_huo_xing_next = "火星" in next_stars_set

        if (is_huo_xing_prev and is_ling_xing_next) or \
                (is_ling_xing_prev and is_huo_xing_next):
            clamping_types.append(f"被【火星】和【铃星】夹（火铃夹），可能带来急躁、冲突或突发事件。")

        # 3. 天梁化忌夹 (天梁 & 化忌星)
        is_tian_liang_prev = "天梁" in prev_stars_set
        is_hua_ji_next = next_hua_info and "化忌" in next_hua_info  # 检查化忌星是否在下一宫
        is_tian_liang_next = "天梁" in next_stars_set
        is_hua_ji_prev = prev_hua_info and "化忌" in prev_hua_info  # 检查化忌星是否在上一宫

        if (is_tian_liang_prev and is_hua_ji_next) or \
                (is_hua_ji_prev and is_tian_liang_next):
            clamping_types.append(f"被【天梁】和【化忌星】夹，可能意味着需要经历波折或考验才能化解困境。")

        return clamping_types

    # --- 4. 生成描述文本 ---

    # 首先处理命宫，这是最重要的宫位
    ming_gong_info = None
    # 查找命宫信息，注意可能存在“命宫宫”等非标准命名
    for item in chart_data:
        cleaned_gong_wei = item[2].replace('宫宫', '宫')  # 仅处理“宫宫”
        if cleaned_gong_wei.startswith("命宫"):
            ming_gong_info = item
            break

    if ming_gong_info:
        main_stars = ming_gong_info[0].replace('，', '、')  # 替换逗号为顿号
        di_zhi = ming_gong_info[1]
        hua_info = ming_gong_info[3]  # 化禄、化权、化科、化忌信息
        aux_misc_stars = ming_gong_info[4]  # 辅星和杂耀
        cleaned_ming_gong_name = ming_gong_info[2].replace('宫宫', '宫')  # 用于夹宫检查

        description = f"您的命宫位于{di_zhi}，主星为{main_stars}。"
        if hua_info:
            description += f" 更有{hua_info}在命宫，这可能意味着在性格或个人发展方面会有一些挑战或需要特别注意的方面。"
        if aux_misc_stars:
            description += f" 同时，命宫还见{aux_misc_stars.replace(',', '、')}等辅星/杂耀，这些星曜将共同影响您的天赋和命运基调。"

        # 获取命宫的三方四正
        ming_gong_related = get_related_palaces(di_zhi)
        description += f" 命宫的三方四正为：{ming_gong_related['对宫']}（对宫）、{ming_gong_related['三方一']}（三方一）、{ming_gong_related['三方二']}（三方二），这些宫位共同影响您的本命特质和发展走向。"

        # 检查命宫的夹宫情况
        ming_gong_clamping_info = check_clamping(cleaned_ming_gong_name)
        if ming_gong_clamping_info:
            description += f" 此外，您的命宫还存在夹宫情况：{' '.join(ming_gong_clamping_info)}"

        description_parts.append(description)
        description_parts.append("\n")  # 添加空行以分隔

    description_parts.append("现在，让我们逐一看看您其他主要宫位的配置，及其三方四正和夹宫情况：\n")

    # 遍历所有宫位，生成描述
    for item in chart_data:
        main_stars = item[0].replace('，', '、')
        di_zhi = item[1]
        gong_wei_original = item[2]
        # 统一宫位名称
        gong_wei_cleaned = gong_wei_original.replace('宫宫', '宫').replace('官禄宫', '事业宫').replace('仆役宫',
                                                                                                       '交友宫')
        hua_info = item[3]
        aux_misc_stars = item[4]

        # 如果已经处理过命宫，并且当前是命宫，则跳过，避免重复
        if gong_wei_cleaned.startswith("命宫") and ming_gong_info and item == ming_gong_info:
            continue

        part = f"- {gong_wei_cleaned}坐落于{di_zhi}位，主要星曜为{main_stars}。"
        if hua_info:
            part += f" 逢{hua_info}，其影响需要特别关注。"
        if aux_misc_stars:
            part += f" 同时，有{aux_misc_stars.replace(',', '、')}等辅星/杂耀照耀。"

        # 获取当前宫位的三方四正
        current_gong_related = get_related_palaces(di_zhi)
        part += f" 该宫位的三方四正是：{current_gong_related['对宫']}（对宫）、{current_gong_related['三方一']}（三方）、{current_gong_related['三方二']}（三方）。"

        # 检查当前宫位的夹宫情况
        current_gong_clamping_info = check_clamping(gong_wei_cleaned)
        if current_gong_clamping_info:
            part += f" 此外，该宫位还存在夹宫情况：{' '.join(current_gong_clamping_info)}"

        description_parts.append(part)

    final_description = "\n".join(description_parts)
    return final_description