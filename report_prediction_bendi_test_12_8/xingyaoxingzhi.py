import pandas as pd
from collections import defaultdict
import os

# --- 用户配置区域 ---
PAIRED_STARS = [
    {'左辅', '右弼'},
    {'文昌', '文曲'},
    {'天魁', '天钺'},
    {'擎羊', '陀罗'},
    {'火星', '铃星'},
    {'地空', '地劫'},
    {'三台', '八座'},
    {'龙池', '凤阁'},
    {'天哭', '天虚'},
    {'孤辰', '寡宿'},
    {'红鸾', '天喜'},
    {'蜚廉', '破碎'},
    {'咸池', '天姚'},
    {'恩光', '天贵'},
    {'天官', '天福'},
    {'天空', '截空'},
    {'旬空', '截空'},
]

WEIGHTS = {
    '辅佐煞空': 2.0,
    '对照': 1.8,
    '夹宫': 1.5,
    '会照': 1.5,
    '会和': 1.2,
    '杂曜': 1.0,
}

SHA_YAO = {'擎羊', '陀罗', '火星', '铃星', '地空', '地劫'}

# 命盘宫位数据
initial_data = [
    ['太阳, 巨门', '寅', '官禄宫', '', '天马', '天月, 孤辰'],
    ['廉贞, 破军', '卯', '仆役宫', '廉贞化忌,文昌化科', '文昌', '红鸾, 天伤'],
    ['天机, 天梁', '辰', '迁移宫', '天机化权', '地空, 陀罗', '天姚, 三台, 龙池, 华盖, 截空'],
    ['天府', '巳', '疾厄宫', '', '禄存, 铃星', '天官, 月德, 劫杀, 破碎, 天使'],
    ['天同, 太阴', '午', '财帛宫', '天同化禄', '地劫, 擎羊', '天哭, 天虚'],
    ['武曲, 贪狼', '未', '子女宫', '', '左辅, 右弼', '天贵, 龙德, 大耗'],
    ['太阳, 巨门', '申', '夫妻宫', '', '', '旬空, 蜚廉, 阴煞'],
    ['天相', '酉', '兄弟宫', '', '天钺, 火星', '天喜, 咸池, 封诰, 天德'],
    ['天机, 天梁', '戌', '命宫', '天机化权', '', '解神, 八座, 凤阁, 天才, 寡宿, 年解'],
    ['紫微, 七杀', '亥', '父母宫', '', '文曲, 天魁', '恩光, 天巫'],
    ['天同, 太阴', '子', '福德宫', '天同化禄', '', '天寿, 天福, 天厨, 天刑'],
    ['武曲, 贪狼', '丑', '田宅宫', '', '', '台辅, 天空']
]

initial_data = [
    ['太阳, 巨门', '寅', '官禄宫', '', '天马', '天月, 孤辰'],
    ['廉贞, 破军', '卯', '仆役宫', '廉贞化忌,文昌化科', '文昌', '红鸾, 天伤'],
    ['天机, 天梁', '辰', '迁移宫', '天机化权', '地空, 陀罗', '天姚, 三台, 龙池, 华盖, 截空'],
    ['天府', '巳', '疾厄宫', '', '禄存, 铃星', '天官, 月德, 劫杀, 破碎, 天使'],
    ['天同, 太阴', '午', '财帛宫', '天同化禄', '地劫, 擎羊', '天哭, 天虚'],
    ['武曲, 贪狼', '未', '子女宫', '', '左辅, 右弼', '天贵, 龙德, 大耗'],
    ['太阳, 巨门', '申', '夫妻宫', '', '', '旬空, 蜚廉, 阴煞'],
    ['天相', '酉', '兄弟宫', '', '天钺, 火星', '天喜, 咸池, 封诰, 天德'],
    ['天机, 天梁', '戌', '命宫', '天机化权', '', '解神, 八座, 凤阁, 天才, 寡宿, 年解'],
    ['紫微, 七杀', '亥', '父母宫', '', '文曲, 天魁', '恩光, 天巫'],
    ['天同, 太阴', '子', '福德宫', '天同化禄', '', '天寿, 天福, 天厨, 天刑'],
    ['武曲, 贪狼', '丑', '田宅宫', '', '', '台辅, 天空']
]

PALACE_ORDER = ['命宫', '兄弟宫', '夫妻宫', '子女宫', '财帛宫', '疾厄宫', '迁移宫', '仆役宫', '官禄宫', '田宅宫',
                '福德宫', '父母宫']

# 天干
tiangan = '丙'
# Excel文件路径
excel_path = r'D:\星曜互涉\合并结果.xlsx'
sheet_name = '组合性质判断'

# --- DEBUG CONFIG ---
# 设置为 True 来开启对“命宫”的详细日志
ENABLE_DEBUG_LOG = True
DEBUG_PALACE_NAME = '命宫'
# 设置为您怀疑的那条规则的 Excel 行号，程序会高亮显示它
DEBUG_RULE_ROW_NUMBER = 4281  # 请根据您的Excel修改此行号


def safe_split(s):
    if isinstance(s, str) and s:
        return [item.strip() for item in s.replace('，', ',').split(',') if item.strip()]
    return []


def preprocess_rules(rules_df):
    """预处理规则，将字符串转为列表，并识别出需要成对匹配的星曜"""
    rules_df_processed = rules_df.copy()
    cols_to_process = ['星系', '辅佐煞空', '杂曜', '夹宫', '对照', '会和', '会照']

    for col in cols_to_process:
        if col in rules_df_processed.columns:
            # 1. 转换为列表
            rules_df_processed[col] = rules_df_processed[col].apply(safe_split)

            # 2. 识别需要成对出现的星曜
            # 创建一个新列来存储需要严格成对匹配的星曜集合
            strict_pairs_col_name = f"{col}_strict_pairs"
            rules_df_processed[strict_pairs_col_name] = rules_df_processed[col].apply(
                lambda star_list: [pair for pair in PAIRED_STARS if pair.issubset(set(star_list))]
            )

    return rules_df_processed


def get_palace_formations(main_stars_str, prev_stars, next_stars):
    formations = []
    adjacent_stars = prev_stars | next_stars
    if ((
                '化禄' in adjacent_stars or '天同化禄' in adjacent_stars or '禄存' in adjacent_stars) and '天梁' in adjacent_stars) and (
            '天相' in main_stars_str or '天府' in main_stars_str):
        formations.append('财荫夹印')
    if ((
                '化忌' in adjacent_stars or '廉贞化忌' in adjacent_stars or '擎羊' in adjacent_stars or '巨门' in adjacent_stars) and '天梁' in adjacent_stars) and (
            '天相' in main_stars_str):
        formations.append('刑忌夹印')

    if ('左辅' in prev_stars and '右弼' in next_stars) or ('左辅' in next_stars and '右弼' in prev_stars):
        formations.append('左右夹')
    if ('文昌' in prev_stars and '文曲' in next_stars) or ('文昌' in next_stars and '文曲' in prev_stars):
        formations.append('昌曲夹')
    if ('天魁' in prev_stars and '天钺' in next_stars) or ('天魁' in next_stars and '天钺' in prev_stars):
        formations.append('魁钺夹')

    if ('擎羊' in prev_stars and '陀罗' in next_stars) or ('擎羊' in next_stars and '陀罗' in prev_stars):
        formations.append('羊陀夹')
    if ('火星' in prev_stars and '铃星' in next_stars) or ('火星' in next_stars and '铃星' in prev_stars):
        formations.append('火铃夹')
    if ('地空' in prev_stars and '地劫' in next_stars) or ('地空' in next_stars and '地劫' in prev_stars):
        formations.append('空劫夹')

    return formations


def preprocess_chart(chart_data):
    """【最终版】预处理，在第一轮就完成所有精确的格局判断"""
    DIZHI_ORDER = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']
    chart_map = {palace[1]: palace for palace in chart_data}
    processed_chart = {}

    def get_all_stars_from_palace(p_data):
        if not p_data: return set()
        return set(safe_split(p_data[0]) + safe_split(p_data[3]) + safe_split(p_data[4]) + safe_split(p_data[5]))

    # 第一轮：计算每个宫位自身的基础信息和精确格局
    for i, dizhi in enumerate(DIZHI_ORDER):
        if dizhi not in chart_map: continue
        palace_data = chart_map[dizhi]
        main_stars_str = palace_data[0]

        prev_palace_data = chart_map.get(DIZHI_ORDER[(i - 1 + 12) % 12], [])
        next_palace_data = chart_map.get(DIZHI_ORDER[(i + 1) % 12], [])

        prev_stars = get_all_stars_from_palace(prev_palace_data)
        next_stars = get_all_stars_from_palace(next_palace_data)

        formations = get_palace_formations(main_stars_str, prev_stars, next_stars)

        processed_chart[dizhi] = {
            '宫位名': palace_data[2], '地支': dizhi, '星系_str': main_stars_str,
            '本宫辅佐煞空': safe_split(palace_data[4]),
            '本宫杂曜': safe_split(palace_data[5]),
            '夹宫组件': sorted(list(prev_stars | next_stars)) + formations,
            '本宫格局': formations  # 只存储本宫确实成立的格局
        }

    # 第二轮：计算三方四正，只注入星曜和远方宫位已成立的格局
    for dizhi in DIZHI_ORDER:
        if dizhi not in processed_chart: continue
        i = DIZHI_ORDER.index(dizhi)
        opposition_dizhi = DIZHI_ORDER[(i + 6) % 12]
        trine1_dizhi = DIZHI_ORDER[(i + 4) % 12]
        trine2_dizhi = DIZHI_ORDER[(i + 8) % 12]

        opposition_stars = get_all_stars_from_palace(chart_map.get(opposition_dizhi, []))
        if opposition_dizhi in processed_chart:
            opposition_stars.update(processed_chart[opposition_dizhi]['本宫格局'])  # 注入对宫已成立的格局

        trine1_stars = get_all_stars_from_palace(chart_map.get(trine1_dizhi, []))
        if trine1_dizhi in processed_chart:
            trine1_stars.update(processed_chart[trine1_dizhi]['本宫格局'])

        trine2_stars = get_all_stars_from_palace(chart_map.get(trine2_dizhi, []))
        if trine2_dizhi in processed_chart:
            trine2_stars.update(processed_chart[trine2_dizhi]['本宫格局'])

        trine_stars = trine1_stars | trine2_stars

        processed_chart[dizhi]['对照宫组件'] = sorted(list(opposition_stars))
        processed_chart[dizhi]['会和宫组件'] = sorted(list(trine_stars))
        processed_chart[dizhi]['会照宫组件'] = sorted(list(opposition_stars | trine_stars))
    return processed_chart


def check_column_match_layered(required_stars_list, palace_stars):
    if not required_stars_list: return True
    palace_star_set = set(palace_stars)

    normal_reqs, special_reqs = [], []
    for req in required_stars_list:
        if "且不见煞" in req:
            special_reqs.append(req.replace("且不见煞", "").strip())
        else:
            normal_reqs.append(req)

    if special_reqs:
        if not SHA_YAO.isdisjoint(palace_star_set): return False
        if not set(special_reqs).issubset(palace_star_set): return False

    if not normal_reqs: return True

    required_star_set = set(normal_reqs)
    remaining_required_set = required_star_set.copy()

    any_pair_matched = False
    for pair in PAIRED_STARS:
        if pair.issubset(required_star_set):
            if pair.issubset(palace_star_set):
                any_pair_matched = True
                remaining_required_set.difference_update(pair)

    any_leftover_matched = False
    if not remaining_required_set:
        return any_pair_matched

    if len(remaining_required_set) >= 3:
        if len(palace_star_set.intersection(remaining_required_set)) >= 2:
            any_leftover_matched = True
    else:
        if remaining_required_set.issubset(palace_star_set):
            any_leftover_matched = True

    return any_pair_matched or any_leftover_matched


def analyze_chart_final(processed_chart, rules_df, chart_tiangan):
    """【最终版 V3】分析器，支持衍生的会照规则合并本宫星曜进行判断"""
    rules_df_processed = preprocess_rules(rules_df)
    final_analysis = {}
    sorted_weight_cols = sorted(WEIGHTS, key=WEIGHTS.get, reverse=True)

    for dizhi, palace_info in processed_chart.items():
        properties = defaultdict(lambda: {'total_score': 0.0, 'reasons': []})

        palace_data_map = {
            '星系': safe_split(palace_info['星系_str']),
            '本宫辅佐煞空': palace_info['本宫辅佐煞空'],
            '本宫杂曜': palace_info['本宫杂曜'],
            '夹宫': palace_info['夹宫组件'],
            '对照': palace_info['对照宫组件'],
            '会和': palace_info['会和宫组件'],
            '会照': palace_info['会照宫组件']
        }

        for index, rule in rules_df_processed.iterrows():
            if (rule['天干'] and rule['天干'] != chart_tiangan): continue
            if (rule['地支'] and rule['地支'] != dizhi): continue

            all_cols_match = True
            # 将列名统一，方便后续处理
            column_mapping = {
                '辅佐煞空': '本宫辅佐煞空',
                '杂曜': '本宫杂曜',
                '夹宫': '夹宫',
                '对照': '对照',
                '会和': '会和',
                '会照': '会照',
                '星系': '星系'
            }

            for rule_col, data_col in column_mapping.items():
                if rule_col not in rule.index: continue  # 跳过不存在的列

                required_stars = rule[rule_col]
                palace_stars_to_check = palace_data_map[data_col]

                # --- 核心修改：判断是否为衍生规则，并合并星曜 ---
                if rule_col == '会照' and rule.get('来源') == '杂曜衍生':
                    # 如果是衍生规则，则检查范围扩大到 本宫(辅佐煞空+杂曜) + 会照宫
                    combined_stars = (
                            set(palace_data_map['本宫辅佐煞空']) |
                            set(palace_data_map['本宫杂曜']) |
                            set(palace_data_map['会照'])
                    )
                    palace_stars_to_check = list(combined_stars)
                # --- 修改结束 ---

                if not check_column_match_layered(required_stars, palace_stars_to_check):
                    all_cols_match = False
                    break

            if all_cols_match:
                score = 1.0;
                source_column = '基础条件'
                for col in sorted_weight_cols:
                    if col in rule.index and rule[col]:
                        score = WEIGHTS[col]
                        source_column = col
                        break

                prop = rule['性质']
                # 为衍生规则添加特殊标记，便于调试和理解
                reason_tag = " (衍生)" if rule.get('来源') == '杂曜衍生' else ""
                reason = f"匹配规则 #{index + 2}{reason_tag} (得分: {score:.1f}, 来自: {source_column})"
                properties[prop]['total_score'] += score
                properties[prop]['reasons'].append(reason)

        final_analysis[palace_info['宫位名']] = dict(properties)
    return final_analysis


def analyze_chart_final_optimized(processed_chart, rules_df, chart_tiangan):
    """【V4 - 性能优化版】"""
    # 准备工作：在循环外只做一次
    # 使用 .to_dict('records') 迭代，比 iterrows() 快得多
    rules_list = rules_df.to_dict('records')
    final_analysis = {}
    sorted_weight_cols = sorted(WEIGHTS, key=WEIGHTS.get, reverse=True)
    column_mapping = {
        '辅佐煞空': '本宫辅佐煞空', '杂曜': '本宫杂曜', '夹宫': '夹宫',
        '对照': '对照', '会和': '会和', '会照': '会照', '星系': '星系'
    }

    # 外层循环：遍历12宫
    for dizhi, palace_info in processed_chart.items():
        properties = defaultdict(lambda: {'total_score': 0.0, 'reasons': []})

        palace_data_map = {
            '星系': safe_split(palace_info['星系_str']),
            '本宫辅佐煞空': palace_info['本宫辅佐煞空'],
            '本宫杂曜': palace_info['本宫杂曜'],
            '夹宫': palace_info['夹宫组件'],
            '对照': palace_info['对照宫组件'],
            '会和': palace_info['会和宫组件'],
            '会照': palace_info['会照宫组件']
        }

        # <<< 优化点 #1：为衍生规则预先计算合并后的星曜 >>>
        # 这个计算对于一个宫位来说是固定的，不需要在内层循环里重复10000次
        combined_stars_for_derived_rules = list(
            set(palace_data_map['本宫辅佐煞空']) |
            set(palace_data_map['本宫杂曜']) |
            set(palace_data_map['会照'])
        )

        # 内层循环：遍历规则
        # enumerate(rules_list) 比 iterrows() 更高效
        for index, rule in enumerate(rules_list):
            if (rule.get('天干') and rule['天干'] != chart_tiangan): continue
            if (rule.get('地支') and rule['地支'] != dizhi): continue

            all_cols_match = True

            for rule_col, data_col in column_mapping.items():
                required_stars = rule.get(rule_col)
                # 如果规则中该列为空，则跳过检查，进一步提速
                if not required_stars: continue

                # <<< 优化点 #2：使用预先计算好的星曜列表 >>>
                if rule_col == '会照' and rule.get('来源') == '杂曜衍生':
                    palace_stars_to_check = combined_stars_for_derived_rules
                else:
                    palace_stars_to_check = palace_data_map[data_col]

                if not check_column_match_layered(safe_split(required_stars), palace_stars_to_check):
                    all_cols_match = False
                    break

            if all_cols_match:
                score = 1.0;
                source_column = '基础条件'
                for col in sorted_weight_cols:
                    if rule.get(col):
                        score = WEIGHTS[col]
                        source_column = col
                        break

                prop = rule['性质']
                reason_tag = " (衍生)" if rule.get('来源') == '杂曜衍生' else ""
                # 注意：因为我们用了 to_dict，原始的 index 不可用了，如果需要可以用原始行号
                reason = f"匹配规则 #{index + 2}{reason_tag} (得分: {score:.1f}, 来自: {source_column})"
                properties[prop]['total_score'] += score
                properties[prop]['reasons'].append(reason)

        final_analysis[palace_info['宫位名']] = dict(properties)
    return final_analysis


def perform_astrological_analysis(raw_data, scores_data, excel_path, sheet_name='性质区分判断', analysis_level='all'):
    """
    对输入的紫微斗数数据进行结构化的多时间层级性质分析，并返回结果字典。

    Args:
        raw_data (dict): 包含'命盘宫位'及各时间层级宫位数据的字典。
        scores_data (list): 包含原局各宫位星系组合的性质得分的列表。
        excel_path (str): 规则Excel文件的完整路径。
        sheet_name (str, optional): Excel文件中规则所在的工作表名称。默认为 '性质区分判断'。
        analysis_level (str, optional): 指定分析的层级。
            可选值: '大运', '流年', '流月', '流日', '流时', 'all'。
            分析会包含指定层级及其所有上级层级。例如'流日'会包含大运、流年、流月和流日。
            默认为 'all'。

    Returns:
        dict: 一个包含完整分析结果的嵌套字典。
              结构为: {宫位名: {时间层级: [规则详情字典, ...], ...}, ...}
              如果发生错误，则返回一个空字典。
    """
    # 1. 读取并验证规则文件 (这部分逻辑不变)
    try:
        rules_df = pd.read_excel(excel_path, sheet_name=sheet_name)
        print("规则文件读取成功！")
    except FileNotFoundError:
        print(f"错误：无法找到规则文件，请检查路径：'{excel_path}'")
        return {}
    except Exception as e:
        print(f"读取 Excel 文件时发生未知错误：{e}")
        return {}

    # 2. 清洗和补全规则 (这部分逻辑不变)
    columns_to_clean = ['星系', '地支', '行经地支', '性质']
    for col in columns_to_clean:
        if col in rules_df.columns:
            rules_df[col] = rules_df[col].astype(str).str.strip()
            if col == '星系':
                rules_df[col] = rules_df[col].str.replace('，', ',', regex=False).str.replace(' ', '', regex=False)

    all_dizhi = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']
    opposite_palace_map = {'子': '午', '丑': '未', '寅': '申', '卯': '酉', '辰': '戌', '巳': '亥', '午': '子',
                           '未': '丑', '申': '寅', '酉': '卯', '戌': '辰', '亥': '巳'}
    unique_base_rules = rules_df[['星系', '地支']].drop_duplicates()
    new_rules_list = []
    for _, base_rule in unique_base_rules.iterrows():
        subset = rules_df[(rules_df['星系'] == base_rule['星系']) & (rules_df['地支'] == base_rule['地支'])]
        existing_transits = set(subset['行经地支'].unique())
        missing_transits = set(all_dizhi) - existing_transits
        for missing in missing_transits:
            opposite = opposite_palace_map[missing]
            if opposite in existing_transits:
                rows_to_copy = subset[subset['行经地支'] == opposite].copy()
                rows_to_copy['行经地支'] = missing
                new_rules_list.append(rows_to_copy)
    if new_rules_list:
        new_rules_df = pd.concat(new_rules_list, ignore_index=True)
        rules_df = pd.concat([rules_df, new_rules_df], ignore_index=True)
        print(f"规则补全完成：新增 {len(new_rules_df)} 条规则，总数 {len(rules_df)}。")
    else:
        print("规则库完整，无需补全。")

    # 3. 准备数据查询字典 (这部分逻辑不变)
    def _parse_scores(score_str):
        if not isinstance(score_str, str): return {}
        score_dict = {}
        for part in score_str.replace('，', ',').split(','):
            if ':' in part:
                key, value = part.split(':', 1)
                score_dict[key.strip()] = float(value.strip())
        return score_dict

    scores_lookup = {(item[0].replace('，', ',').replace(' ', ''), item[1].strip()): _parse_scores(item[2]) for item in
                     scores_data}
    original_palace_map = {
        item[2].strip(): {'星系': item[0].replace('，', ',').replace(' ', ''), '地支': item[1].strip()} for item in
        raw_data['原局盘_new']}

    # --- 核心修改：定义分析层级和选择要分析的时间段 ---
    # raw_data的键到用户友好名称的映射
    time_periods_map = {'大限盘': '大限', '流年盘': '流年', '流月盘': '流月', '流日盘': '流日', '流时盘': '流时'}

    # 用户输入到raw_data键列表的映射 (实现层级包含)
    level_to_keys_map = {
        '大运': ['大限盘'],
        '流年': ['大限盘', '流年盘'],
        '流月': ['大限盘', '流年盘', '流月盘'],
        '流日': ['大限盘', '流年盘', '流月盘', '流日盘'],
        '流时': ['大限盘', '流年盘', '流月盘', '流日盘', '流时盘'],
        'all': ['大限盘', '流年盘', '流月盘', '流日盘', '流时盘']
    }

    if analysis_level not in level_to_keys_map:
        print(f"错误：无效的 analysis_level '{analysis_level}'。将使用 'all'。")
        analysis_level = 'all'

    selected_period_keys = level_to_keys_map[analysis_level]
    # --- 修改结束 ---

    # 4. 主分析流程 (重构为构建字典)
    analysis_results = {}  # 初始化最终结果字典

    for original_gongwei_name, original_info in original_palace_map.items():
        original_xingxi = original_info['星系']
        original_dizhi = original_info['地支']

        # 为当前宫位创建条目
        analysis_results[original_gongwei_name] = {
            "原局信息": {"星系": original_xingxi, "地支": original_dizhi}
        }

        scores_dict = scores_lookup.get((original_xingxi, original_dizhi), {})
        total_score = sum(scores_dict.values()) if scores_dict else 0

        # 遍历选定的时间层级
        for period_key in selected_period_keys:
            period_name = time_periods_map[period_key]
            transit_dizhi = next(
                (item[1].strip() for item in raw_data.get(period_key, []) if item[2].strip() == original_gongwei_name),
                None)

            if transit_dizhi:
                matched_rules = rules_df[
                    (rules_df['星系'] == original_xingxi) &
                    (rules_df['地支'] == original_dizhi) &
                    (rules_df['行经地支'] == transit_dizhi)
                    ]

                # 为当前时间层级创建列表
                if period_name not in analysis_results[original_gongwei_name]:
                    analysis_results[original_gongwei_name][period_name] = []

                if not matched_rules.empty:
                    for _, row in matched_rules.iterrows():
                        current_nature = row['性质']
                        description = row['性质与现象描述']

                        score_text = f"无法计算 (分数列表中未提供 '{current_nature}' 的分数)"
                        if total_score > 0 and current_nature in scores_dict:
                            nature_score = scores_dict[current_nature]
                            percentage = (nature_score / total_score) * 100
                            score_text = f"{percentage:.2f}% ({nature_score} / {total_score})"

                        # 构建规则详情字典并添加到列表中
                        rule_details = {
                            "性质": current_nature,
                            "现象描述": description,
                            "得分": score_text,
                            "行经地支": transit_dizhi
                        }
                        analysis_results[original_gongwei_name][period_name].append(rule_details)

    print("\n分析流程执行完毕。")
    return analysis_results


# --- 主程序执行 ---
if __name__ == "__main__":
    print("开始进行命盘性质分析 (最终版 - 全逻辑整合)...")

    try:
        rules_df = pd.read_excel(excel_path, sheet_name=sheet_name, keep_default_na=False)
        print(f"成功从 '{excel_path}' 的 '{sheet_name}' 工作表中加载 {len(rules_df)} 条规则。")
    except Exception as e:
        print(f"加载Excel文件时出错: {e}")
        exit()

    rules_df['来源'] = '原始'  # 首先给所有原始规则添加“来源”列
    new_rules = []

    for index, rule in rules_df.iterrows():
        zayao_content = str(rule.get('杂曜', ''))
        # 确保其他影响宫位为空
        if (zayao_content.strip() and
                not str(rule.get('会照', '')).strip() and
                not str(rule.get('会和', '')).strip() and
                not str(rule.get('对照', '')).strip()):
            new_rule = rule.copy()
            new_rule['会照'] = new_rule['杂曜']
            new_rule['杂曜'] = ''
            new_rule['来源'] = '杂曜衍生'  # ★ 关键：给衍生规则打上标签
            new_rules.append(new_rule)

    if new_rules:
        new_rules_df = pd.DataFrame(new_rules)
        rules_df = pd.concat([rules_df, new_rules_df], ignore_index=True)
        print(f"根据“杂曜”规则衍生出 {len(new_rules)} 条新的“会照”规则。总规则数变为: {len(rules_df)}")
    # --- 衍生逻辑结束 ---

    processed_chart_data = preprocess_chart(initial_data)
    chart_by_name = {v['宫位名']: v for k, v in processed_chart_data.items()}
    print("\n命盘数据预处理完成。")

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # print(rules_df.loc[(rules_df["星系"] == '天机，天梁') & (rules_df["地支"] == '戌')])

    analysis_result = analyze_chart_final(processed_chart_data, rules_df, tiangan)

    # print(f"\n{'='*25} 最终分析结果 {'='*25}")

    palace_info_map = {p[2]: (p[0], p[1]) for p in initial_data}
    summary_output = []

    for palace_name in PALACE_ORDER:
        palace_details = chart_by_name.get(palace_name)
        if not palace_details: continue

        properties = analysis_result.get(palace_name, {})

        # print(f"\n---------- {palace_name} ({palace_details.get('地支', '?')}) ----------")
        # print("palace_details",palace_details)
        # print("  [宫位详情]")
        fuzuo_str = ', '.join(sorted(palace_details.get('本宫辅佐煞空', []))) or "无"
        zayao_str = ', '.join(sorted(palace_details.get('本宫杂曜', []))) or "无"
        jiagong_str = ', '.join(sorted(palace_details.get('夹宫组件', []))) or "无"
        prev_str = ', '.join(sorted(palace_details.get('prev_palace_stars', []))) or "无"
        next_str = ', '.join(sorted(palace_details.get('next_palace_stars', []))) or "无"
        duizhao_str = ', '.join(sorted(palace_details.get('对照宫组件', []))) or "无"
        huihe_str = ', '.join(sorted(palace_details.get('会和宫组件', []))) or "无"
        # print(f"    主星: {palace_details.get('星系_str', '未知')}")
        # print(f"    同宫辅佐煞空: {fuzuo_str}")
        # print(f"    同宫杂曜: {zayao_str}")
        # print(f"    夹宫 (邻宫合计): {jiagong_str}")
        # # print(f"      - 前一宫: {prev_str}")
        # # print(f"      - 后一宫: {next_str}")
        # print(f"    对照宫 (含格局): {duizhao_str}")
        # print(f"    会和宫 (含格局): {huihe_str}")

        print("  [性质分析]")
        if not properties:
            print("    >> 未匹配到任何性质。")
        else:
            sorted_properties = sorted(properties.items(), key=lambda item: item[1]['total_score'], reverse=True)
            for prop, data in sorted_properties:
                # print(f"    >> {prop}: {data['total_score']:.1f}")
                sorted_reasons = sorted(data['reasons'])
                # for reason in sorted_reasons:
                #     print(f"       - {reason}")
                unique_reasons = sorted(list(set(data['reasons'])))
                # print(f"       (由 {len(data['reasons'])} 条规则构成)")

        main_stars, dizhi = palace_info_map.get(palace_name, ('未知', '?'))
        if not properties:
            summary_str = "未匹配到任何性质"
        else:
            sorted_properties = sorted(properties.items(), key=lambda item: item[1]['total_score'], reverse=True)
            summary_str = ','.join([f"{prop}:{data['total_score']:.1f}" for prop, data in sorted_properties])
        summary_output.append([main_stars, dizhi, summary_str])

    print("\n" + "=" * 65)
    print(f"\n{'=' * 25} 最终摘要输出 {'=' * 25}")
    print(summary_output)
    print("\n" + "=" * 65)





