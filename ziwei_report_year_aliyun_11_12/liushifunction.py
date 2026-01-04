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


key_translation_map = {
    "code": "代码",
    "data": "数据",
    "astrolabe": "命盘",
    "gender": "性别",
    "solarDate": "公历日期",
    "lunarDate": "农历日期",
    "chineseDate": "干支纪年",
    "rawDates": "原始日期数据",
    "lunarYear": "农历年份",
    "lunarMonth": "农历月份",
    "lunarDay": "农历日期",
    "isLeap": "是否闰月",
    "yearly": "年柱",
    "monthly": "月柱",
    "daily": "日柱",
    "hourly": "时柱",
    "time": "时辰",
    "timeRange": "时辰范围",
    "sign": "星座",
    "zodiac": "生肖",
    "earthlyBranchOfBodyPalace": "身宫地支",
    "earthlyBranchOfSoulPalace": "命宫地支",
    "soul": "命主星",
    "body": "身主星",
    "fiveElementsClass": "五行局",
    "palaces": "宫位",
    "index": "索引",
    "name": "名称",
    "isBodyPalace": "是否身宫",
    "isOriginalPalace": "是否本命宫位",
    "heavenlyStem": "天干",
    "earthlyBranch": "地支",
    "majorStars": "主星",
    "minorStars": "辅星煞星",
    "adjectiveStars": "杂曜",
    "type": "类型",
    "scope": "作用范围",
    "brightness": "亮度",
    "mutagen": "化曜",
    "changsheng12": "长生十二神",
    "boshi12": "博士十二神",
    "jiangqian12": "将前十二神",
    "suiqian12": "岁前十二神",
    "decadal": "大限数据",
    "range": "年龄范围",
    "ages": "流年年龄",
    "horoscope": "运势数据",
    "age": "小限数据",
    "yearly": "流年数据",
    "monthly": "流月数据",
    "daily": "流日数据",
    "hourly": "流时数据",
    "palaceNames": "宫位列表",
    "stars": "星曜列表",
    "nominalAge": "虚岁",
    "yearlyDecStar": "流年神煞" # 根据内容推测
}

# 定义值的翻译映射表 (保留之前的值翻译)
value_translation_map = {
    "Male": "男",
    "Servants Palace": "仆役",
    "Migration Palace": "迁移",
    "Illness Palace": "疾厄",
    "Wealth Palace": "财帛",
    "Children Palace": "子女",
    "Spouse Palace": "夫妻",
    "Siblings Palace": "兄弟",
    "Life Palace": "命宫",
    "Parents Palace": "父母",
    "Fortune and Virtue Palace": "福德",
    "Property Palace": "田宅",
    "Career Palace": "官禄",
    "major": "主星",
    "soft": "辅星", # 或吉星
    "tough": "煞星", # 或凶星
    "adjective": "杂曜",
    "flower": "桃花星",
    "tianma": "天马",
    "helper": "辅助星",
    "origin": "本命",
    "decadal": "大限",
    "yearly": "流年",
    "Temple": "庙",
    "Prosperous": "旺",
    "Average": "平",
    "Trapped/Weak": "陷", # 根据语境可能需要调整
    "Leisure": "闲",
    "Lu": "禄",
    "Quan": "权",
    "Ke": "科",
    "Ji": "忌",
    "Extinction": "绝",
    "Ambush": "伏兵",
    "Death God": "亡神",
    "Illness Charm": "病符",
    "Decadal Limit": "大限",
    "Yun Yue": "运钺",
    "Yun Qu": "运曲"}


def translate_json(data, key_map, value_map):
    """
    递归遍历 JSON 数据，将英文键和值翻译成中文。

    Args:
        data: JSON 数据（字典或列表）。
        key_map: 键的翻译映射字典。
        value_map: 值的翻译映射字典。

    Returns:
        翻译后的 JSON 数据。
    """
    if isinstance(data, dict):
        # 如果是字典，遍历键值对，翻译键和值
        translated_dict = {}
        for key, value in data.items():
            # 翻译键，如果找不到则保留原键
            translated_key = key_map.get(key, key)
            # 递归翻译值
            translated_value = translate_json(value, key_map, value_map)
            translated_dict[translated_key] = translated_value
        return translated_dict
    elif isinstance(data, list):
        # 如果是列表，遍历列表元素，递归翻译
        return [translate_json(item, key_map, value_map) for item in data]
    elif isinstance(data, str):
        # 如果是字符串，查找值的翻译映射
        return value_map.get(data, data) # 如果找不到值的翻译，返回原字符串
    else:
        # 其他类型（数字、布尔值等）直接返回
        return data



def sihua(data, name, rihuayao):
    """
    Extracts Si Hua information for stars in a list based on name and rihuayao keys.
    Only includes the "化" part if the rihuayao value is not empty.
    (Based on user's provided sihua code)

    Args:
        data (list): A list of dictionaries, where each dictionary represents a star.
        name (str): The key in each star dictionary that holds the star's name.
        rihuayao (str): The key in each star dictionary that holds the Si Hua information.

    Returns:
        str: A comma-separated string of Si Hua information (e.g., "紫微化权,天府化科").
    """
    a = ""
    # print(f"sihua received data: {data}") # Debug print
    # print(f"sihua received name key: {name}") # Debug print
    # print(f"sihua received rihuayao key: {rihuayao}") # Debug print
    if not isinstance(data, list):
        # logging.warning(f"sihua expected list, but received {type(data)}")
        return ""

    for i in data:
        try:
            # Ensure 'i' is a dictionary and contains both 'name' and 'rihuayao' keys
            # Added check: and i[rihuayao] to ensure the rihuayao value is not empty/falsy
            if isinstance(i, dict) and name in i and rihuayao in i and i[rihuayao]:
                # Concatenate star name, "化", and rihuayao value
                a = a + str(i[name]) + "化" + str(i[rihuayao]) + ","
            # else:
                 # logging.warning(f"Skipping item in sihua: {i} (not dict or missing keys '{name}' or '{rihuayao}' or rihuayao value is empty)")
        except Exception as e:
            # logging.error(f"Error processing star data in sihua: {e}", exc_info=True)
            pass # Continue processing other stars even if one fails

    return a.strip(',') # Remove trailing comma


def zhuxing(data, name):
    a = ""
    if not isinstance(data, list):
        return ""
    for i in data:
        try:
            if isinstance(i, dict) and name in i:
                 a = a + str(i[name]) + ","
        except Exception as e:
            pass # Or raise the exception if it's critical
    return a.strip(',') # Remove trailing comma


# Corrected ziweids function definition and logic with sorting
def ziweids(
    converted_data,
    data_key,
    mingzhu_key,
    shenzhu_key,
    wuxingju_key,
    gongwei_list_key,
    zhuxing_list_key,     # Key for the list of main stars within a palace dict
    dizhi_key,
    rihuayao_key,         # Key for Ri Hua Yao (used by sihua function)
    fuxing_list_key,      # Key for the list of auxiliary/sha stars within a palace dict
    zayao_list_key,       # Key for the list of minor stars within a palace dict
    palace_name_key,      # Key for the palace name and also star name within star dicts
    get_star_names_func   # The actual function to get star names (i.e., the 'zhuxing' function)
):
    """
    Processes Zi Wei Dou Shu data to generate concatenated strings for each palace.
    Includes sorting for main stars, auxiliary/sha stars, and Si Hua information.

    Args:
        converted_data (dict): The main data structure.
        data_key (str): Key to access the main data section (e.g., "命盘").
        mingzhu_key (str): Key for Ming Zhu star.
        shenzhu_key (str): Key for Shen Zhu star.
        wuxingju_key (str): Key for Wu Xing Ju.
        gongwei_list_key (str): Key for the list of palace data.
        zhuxing_list_key (str): Key within palace data for the list of main stars.
        dizhi_key (str): Key within palace data for the Earthly Branch.
        rihuayao_key (str): Key for Ri Hua Yao (used by sihua function).
        fuxing_list_key (str): Key within palace data for the list of auxiliary/sha stars.
        zayao_list_key (str): Key within palace data for the list of minor stars.
        palace_name_key (str): Key for the palace name and also star name within star dicts.
        get_star_names_func (function): The function to extract star names from a list of star data.

    Returns:
        list: A list of concatenated strings, one for each palace.
    """
    # Define sorting orders
    zhuxing_order = ["紫微","天机","太阳","武曲","天同","巨门","廉贞","天府","太阴","贪狼","天相","天梁","七杀","破军"]
    fuxing_order = ["天魁","天钺","左辅","右弼","文昌","文曲","禄存","天马","火星","铃星","擎羊","陀罗","地空","地劫"]
    sihua_order = ['紫微化禄','紫微化权','紫微化科','紫微化忌','天机化禄','天机化权','天机化科','天机化忌','太阳化禄','太阳化权','太阳化科','太阳化忌',
                   '武曲化禄','武曲化权','武曲化科','武曲化忌','天同化禄','天同化权','天同化科','天同化忌','巨门化禄','巨门化权','巨门化科',
                   '巨门化忌','廉贞化禄','廉贞化权','廉贞化科','廉贞化忌','天府化禄','天府化权','天府化科','天府化忌','太阴化禄','太阴化权',
                   '太阴化科','太阴化忌','贪狼化禄','贪狼化权','贪狼化科','贪狼化忌','天相化禄','天相化权','天相化科','天相化忌','天梁化禄',
                   '天梁化权','天梁化科','天梁化忌','七杀化禄','七杀化权','七杀化科','七杀化忌','破军化禄','破军化权','破军化科','破军化忌',
                   '文昌化禄','文昌化权','文曲化科','文曲化忌','文昌化禄','文昌化权','文昌化科','文昌化忌','文曲化禄','文曲化权','文曲化科','文曲化忌'] # Corrected duplicate entries and added missing ones based on common Si Hua

    results_array = []
    # Ensure gongwei_list_key exists and is a list
    palaces_data = converted_data.get(data_key, {}).get(gongwei_list_key, [])
    if not isinstance(palaces_data, list) or len(palaces_data) != 12:
         print(f"Error: Expected a list of 12 palaces for key '{gongwei_list_key}', but got {type(palaces_data)} with length {len(palaces_data) if isinstance(palaces_data, list) else 'N/A'}")
         return [] # Return empty list if palace data is not as expected


    for i in range(0, 12):
        palace_data = palaces_data[i]
        # Ensure palace_data is a dictionary
        if not isinstance(palace_data, dict):
             print(f"Warning: Expected palace data at index {i} to be a dictionary, but got {type(palace_data)}. Skipping.")
             lists.append("") # Add empty string or handle as needed
             continue

        # Safely get base data using .get()
        ans0 = converted_data.get(data_key, {}).get(mingzhu_key, "")
        # print(f"Ming Zhu: {ans0}")
        ans00 = converted_data.get(data_key, {}).get(shenzhu_key, "")
        # print(f"Shen Zhu: {ans00}")
        ans000 = converted_data.get(data_key, {}).get(wuxingju_key, "")
        # print(f"Wu Xing Ju: {ans000}")

        # 星系 (Main Stars)
        # Get the list of main stars using the correct key
        main_stars_list = palace_data.get(zhuxing_list_key, [])
        # Call the passed function to get star names string
        ans1_str = get_star_names_func(main_stars_list, palace_name_key)
        # Split into list, sort, and join back
        ans1_list = [star.strip() for star in ans1_str.split(',') if star.strip()] # Handle potential empty strings after split
        ans1_list.sort(key=lambda x: zhuxing_order.index(x) if x in zhuxing_order else len(zhuxing_order)) # Sort based on predefined order, put unknown at the end
        ans1 = ",".join(ans1_list)


        if len(ans1) == 0:
            # print(f"No main stars found in palace {i}, checking opposite palace.")
            # Assuming the logic here is to get the opposite palace's stars if the current one is empty
            # Need to ensure the index calculation is correct and doesn't go out of bounds
            opposite_palace_index = (i + 6) % 12 # Use modulo 12 for cyclic palaces
            opposite_palace_data = palaces_data[opposite_palace_index]
            if isinstance(opposite_palace_data, dict):
                 opposite_main_stars_list = opposite_palace_data.get(zhuxing_list_key, [])
                 ans1_str_opposite = get_star_names_func(opposite_main_stars_list, palace_name_key)
                 ans1_list_opposite = [star.strip() for star in ans1_str_opposite.split(',') if star.strip()]
                 ans1_list_opposite.sort(key=lambda x: zhuxing_order.index(x) if x in zhuxing_order else len(zhuxing_order))
                 ans1 = ",".join(ans1_list_opposite)
                 # print(f"Main stars from opposite palace {opposite_palace_index}: {ans1}")
            # else:
                 # print(f"Warning: Opposite palace data at index {opposite_palace_index} is not a dictionary.")


        # 地支 (Earthly Branch)
        ans2 = palace_data.get(dizhi_key, "")

        # 宫位 (Palace Name)
        ans3 = palace_data.get(palace_name_key, "")

        # 四化 (Si Hua)
        # Call sihua function to get Si Hua string
        ans4_str = sihua(main_stars_list, palace_name_key, rihuayao_key)
        # Split into list, sort, and join back
        ans4_list = [sihua_item.strip() for sihua_item in ans4_str.split(',') if sihua_item.strip()]
        ans4_list.sort(key=lambda x: sihua_order.index(x) if x in sihua_order else len(sihua_order)) # Sort based on predefined order
        ans4 = ",".join(ans4_list)


        # 辅佐煞空 (Auxiliary Stars, Sha Stars, etc.)
        # Get the list of auxiliary/sha stars using the correct key
        fuxing_list = palace_data.get(fuxing_list_key, [])
        # Call the passed function to get star names string
        ans5_str = get_star_names_func(fuxing_list, palace_name_key)
        # Split into list, sort, and join back
        ans5_list = [star.strip() for star in ans5_str.split(',') if star.strip()]
        ans5_list.sort(key=lambda x: fuxing_order.index(x) if x in fuxing_order else len(fuxing_order)) # Sort based on predefined order
        ans5 = ",".join(ans5_list)


        # 杂曜 (Minor Stars)
        # Get the list of minor stars using the correct key
        zayao_list = palace_data.get(zayao_list_key, [])
        # Call the passed function to get star names string
        ans6_str = get_star_names_func(zayao_list, palace_name_key)
        # Minor stars are not requested to be sorted, so just use the string directly
        #ans6 = ans6_str


        # Concatenate the results
        # Ensure all parts are strings before concatenation
        #ans7 = f"{ans0}{ans00}{ans000}{ans1}{ans2}{ans3}{ans4}{ans5}{ans6}"
        ans7 = f"{ans1}{ans2}{ans3}{ans4}{ans5}"
        #lists.append(ans7) # Remove commas from the final concatenated string
        results_array.append([ans1.replace(",","，"),ans2, ans3+"宫",ans4,ans5])

    return results_array



def ziweids_conzayao(
    converted_data,
    data_key,
    mingzhu_key,
    shenzhu_key,
    wuxingju_key,
    gongwei_list_key,
    zhuxing_list_key,     # Key for the list of main stars within a palace dict
    dizhi_key,
    rihuayao_key,         # Key for Ri Hua Yao (used by sihua function)
    fuxing_list_key,      # Key for the list of auxiliary/sha stars within a palace dict
    zayao_list_key,       # Key for the list of minor stars within a palace dict
    palace_name_key,      # Key for the palace name and also star name within star dicts
    get_star_names_func   # The actual function to get star names (i.e., the 'zhuxing' function)
):
    """
    Processes Zi Wei Dou Shu data to generate concatenated strings for each palace.
    Includes sorting for main stars, auxiliary/sha stars, and Si Hua information.

    Args:
        converted_data (dict): The main data structure.
        data_key (str): Key to access the main data section (e.g., "命盘").
        mingzhu_key (str): Key for Ming Zhu star.
        shenzhu_key (str): Key for Shen Zhu star.
        wuxingju_key (str): Key for Wu Xing Ju.
        gongwei_list_key (str): Key for the list of palace data.
        zhuxing_list_key (str): Key within palace data for the list of main stars.
        dizhi_key (str): Key within palace data for the Earthly Branch.
        rihuayao_key (str): Key for Ri Hua Yao (used by sihua function).
        fuxing_list_key (str): Key within palace data for the list of auxiliary/sha stars.
        zayao_list_key (str): Key within palace data for the list of minor stars.
        palace_name_key (str): Key for the palace name and also star name within star dicts.
        get_star_names_func (function): The function to extract star names from a list of star data.

    Returns:
        list: A list of concatenated strings, one for each palace.
    """
    # Define sorting orders
    zhuxing_order = ["紫微","天机","太阳","武曲","天同","巨门","廉贞","天府","太阴","贪狼","天相","天梁","七杀","破军"]
    fuxing_order = ["天魁","天钺","左辅","右弼","文昌","文曲","禄存","天马","火星","铃星","擎羊","陀罗","地空","地劫"]
    sihua_order = ['紫微化禄','紫微化权','紫微化科','紫微化忌','天机化禄','天机化权','天机化科','天机化忌','太阳化禄','太阳化权','太阳化科','太阳化忌',
                   '武曲化禄','武曲化权','武曲化科','武曲化忌','天同化禄','天同化权','天同化科','天同化忌','巨门化禄','巨门化权','巨门化科',
                   '巨门化忌','廉贞化禄','廉贞化权','廉贞化科','廉贞化忌','天府化禄','天府化权','天府化科','天府化忌','太阴化禄','太阴化权',
                   '太阴化科','太阴化忌','贪狼化禄','贪狼化权','贪狼化科','贪狼化忌','天相化禄','天相化权','天相化科','天相化忌','天梁化禄',
                   '天梁化权','天梁化科','天梁化忌','七杀化禄','七杀化权','七杀化科','七杀化忌','破军化禄','破军化权','破军化科','破军化忌',
                   '文昌化禄','文昌化权','文曲化科','文曲化忌','文昌化禄','文昌化权','文昌化科','文昌化忌','文曲化禄','文曲化权','文曲化科','文曲化忌'] # Corrected duplicate entries and added missing ones based on common Si Hua

    results_array = []
    # Ensure gongwei_list_key exists and is a list
    palaces_data = converted_data.get(data_key, {}).get(gongwei_list_key, [])
    if not isinstance(palaces_data, list) or len(palaces_data) != 12:
         print(f"Error: Expected a list of 12 palaces for key '{gongwei_list_key}', but got {type(palaces_data)} with length {len(palaces_data) if isinstance(palaces_data, list) else 'N/A'}")
         return [] # Return empty list if palace data is not as expected


    for i in range(0, 12):
        palace_data = palaces_data[i]
        # Ensure palace_data is a dictionary
        if not isinstance(palace_data, dict):
             print(f"Warning: Expected palace data at index {i} to be a dictionary, but got {type(palace_data)}. Skipping.")
             lists.append("") # Add empty string or handle as needed
             continue

        # Safely get base data using .get()
        ans0 = converted_data.get(data_key, {}).get(mingzhu_key, "")
        # print(f"Ming Zhu: {ans0}")
        ans00 = converted_data.get(data_key, {}).get(shenzhu_key, "")
        # print(f"Shen Zhu: {ans00}")
        ans000 = converted_data.get(data_key, {}).get(wuxingju_key, "")
        # print(f"Wu Xing Ju: {ans000}")

        # 星系 (Main Stars)
        # Get the list of main stars using the correct key
        main_stars_list = palace_data.get(zhuxing_list_key, [])
        # Call the passed function to get star names string
        ans1_str = get_star_names_func(main_stars_list, palace_name_key)
        # Split into list, sort, and join back
        ans1_list = [star.strip() for star in ans1_str.split(',') if star.strip()] # Handle potential empty strings after split
        ans1_list.sort(key=lambda x: zhuxing_order.index(x) if x in zhuxing_order else len(zhuxing_order)) # Sort based on predefined order, put unknown at the end
        ans1 = ",".join(ans1_list)


        if len(ans1) == 0:
            # print(f"No main stars found in palace {i}, checking opposite palace.")
            # Assuming the logic here is to get the opposite palace's stars if the current one is empty
            # Need to ensure the index calculation is correct and doesn't go out of bounds
            opposite_palace_index = (i + 6) % 12 # Use modulo 12 for cyclic palaces
            opposite_palace_data = palaces_data[opposite_palace_index]
            if isinstance(opposite_palace_data, dict):
                 opposite_main_stars_list = opposite_palace_data.get(zhuxing_list_key, [])
                 ans1_str_opposite = get_star_names_func(opposite_main_stars_list, palace_name_key)
                 ans1_list_opposite = [star.strip() for star in ans1_str_opposite.split(',') if star.strip()]
                 ans1_list_opposite.sort(key=lambda x: zhuxing_order.index(x) if x in zhuxing_order else len(zhuxing_order))
                 ans1 = ",".join(ans1_list_opposite)
                 # print(f"Main stars from opposite palace {opposite_palace_index}: {ans1}")
            # else:
                 # print(f"Warning: Opposite palace data at index {opposite_palace_index} is not a dictionary.")


        # 地支 (Earthly Branch)
        ans2 = palace_data.get(dizhi_key, "")

        # 宫位 (Palace Name)
        ans3 = palace_data.get(palace_name_key, "")

        # 四化 (Si Hua)
        # Call sihua function to get Si Hua string
        ans4_str = sihua(main_stars_list, palace_name_key, rihuayao_key)
        # Split into list, sort, and join back
        ans4_list = [sihua_item.strip() for sihua_item in ans4_str.split(',') if sihua_item.strip()]
        ans4_list.sort(key=lambda x: sihua_order.index(x) if x in sihua_order else len(sihua_order)) # Sort based on predefined order
        ans4 = ",".join(ans4_list)


        # 辅佐煞空 (Auxiliary Stars, Sha Stars, etc.)
        # Get the list of auxiliary/sha stars using the correct key
        fuxing_list = palace_data.get(fuxing_list_key, [])
        # Call the passed function to get star names string
        ans5_str = get_star_names_func(fuxing_list, palace_name_key)
        # Split into list, sort, and join back
        ans5_list = [star.strip() for star in ans5_str.split(',') if star.strip()]
        ans5_list.sort(key=lambda x: fuxing_order.index(x) if x in fuxing_order else len(fuxing_order)) # Sort based on predefined order
        ans5 = ",".join(ans5_list)


        # 杂曜 (Minor Stars)
        # Get the list of minor stars using the correct key
        zayao_list = palace_data.get(zayao_list_key, [])
        # Call the passed function to get star names string
        ans6_str = get_star_names_func(zayao_list, palace_name_key)
        # Minor stars are not requested to be sorted, so just use the string directly
        #ans6 = ans6_str


        # Concatenate the results
        # Ensure all parts are strings before concatenation
        #ans7 = f"{ans0}{ans00}{ans000}{ans1}{ans2}{ans3}{ans4}{ans5}{ans6}"
        ans7 = f"{ans1}{ans2}{ans3}{ans4}{ans5}"
        #lists.append(ans7) # Remove commas from the final concatenated string
        results_array.append([ans1.replace(",","，"),ans2, ans3+"宫",ans4,ans5,ans6_str])

    return results_array




def ziweids_daily(
    converted_data,
    mingzhu_key,
    shenzhu_key,
    wuxingju_key,
    gongwei_list_key,
    zhuxing_list_key,     # Key for the list of main stars within a palace dict
    dizhi_key,
    rihuayao_key,         # Key for Ri Hua Yao (used by sihua function)
    fuxing_list_key,      # Key for the list of auxiliary/sha stars within a palace dict
    zayao_list_key,       # Key for the list of minor stars within a palace dict
    palace_name_key,      # Key for the palace name and also star name within star dicts
    get_star_names_func   # The actual function to get star names (i.e., the 'zhuxing' function)
):
    """
    Processes Zi Wei Dou Shu data to generate concatenated strings for each palace.
    Includes sorting for main stars, auxiliary/sha stars, and Si Hua information.

    Args:
        converted_data (dict): The main data structure.
        mingzhu_key (str): Key for Ming Zhu star.
        shenzhu_key (str): Key for Shen Zhu star.
        wuxingju_key (str): Key for Wu Xing Ju.
        gongwei_list_key (str): Key for the list of palace data.
        zhuxing_list_key (str): Key within palace data for the list of main stars.
        dizhi_key (str): Key within palace data for the Earthly Branch.
        rihuayao_key (str): Key for Ri Hua Yao (used by sihua function).
        fuxing_list_key (str): Key within palace data for the list of auxiliary/sha stars.
        zayao_list_key (str): Key within palace data for the list of minor stars.
        palace_name_key (str): Key for the palace name and also star name within star dicts.
        get_star_names_func (function): The function to extract star names from a list of star data.

    Returns:
        list: A list of concatenated strings, one for each palace.
    """
    # Define sorting orders
    zhuxing_order = ["紫微","天机","太阳","武曲","天同","巨门","廉贞","天府","太阴","贪狼","天相","天梁","七杀","破军"]
    fuxing_order = ["天魁","天钺","左辅","右弼","文昌","文曲","禄存","天马","火星","铃星","擎羊","陀罗","地空","地劫"]
    sihua_order = ['紫微化禄','紫微化权','紫微化科','紫微化忌','天机化禄','天机化权','天机化科','天机化忌','太阳化禄','太阳化权','太阳化科','太阳化忌',
                   '武曲化禄','武曲化权','武曲化科','武曲化忌','天同化禄','天同化权','天同化科','天同化忌','巨门化禄','巨门化权','巨门化科',
                   '巨门化忌','廉贞化禄','廉贞化权','廉贞化科','廉贞化忌','天府化禄','天府化权','天府化科','天府化忌','太阴化禄','太阴化权',
                   '太阴化科','太阴化忌','贪狼化禄','贪狼化权','贪狼化科','贪狼化忌','天相化禄','天相化权','天相化科','天相化忌','天梁化禄',
                   '天梁化权','天梁化科','天梁化忌','七杀化禄','七杀化权','七杀化科','七杀化忌','破军化禄','破军化权','破军化科','破军化忌',
                   '文昌化禄','文昌化权','文曲化科','文曲化忌','文昌化禄','文昌化权','文昌化科','文昌化忌','文曲化禄','文曲化权','文曲化科','文曲化忌'] # Corrected duplicate entries and added missing ones based on common Si Hua

    results_array = []
    # Ensure gongwei_list_key exists and is a list
    palaces_data = converted_data.get(gongwei_list_key, [])
    if not isinstance(palaces_data, list) or len(palaces_data) != 12:
         print(f"Error: Expected a list of 12 palaces for key '{gongwei_list_key}', but got {type(palaces_data)} with length {len(palaces_data) if isinstance(palaces_data, list) else 'N/A'}")
         return [] # Return empty list if palace data is not as expected


    for i in range(0, 12):
        palace_data = palaces_data[i]
        # Ensure palace_data is a dictionary
        if not isinstance(palace_data, dict):
             print(f"Warning: Expected palace data at index {i} to be a dictionary, but got {type(palace_data)}. Skipping.")
             lists.append("") # Add empty string or handle as needed
             continue

        # Safely get base data using .get()
        ans0 = converted_data.get(mingzhu_key, "")
        # print(f"Ming Zhu: {ans0}")
        ans00 = converted_data.get(shenzhu_key, "")
        # print(f"Shen Zhu: {ans00}")
        ans000 = converted_data.get(wuxingju_key, "")
        # print(f"Wu Xing Ju: {ans000}")

        # 星系 (Main Stars)
        # Get the list of main stars using the correct key
        main_stars_list = palace_data.get(zhuxing_list_key, [])
        # Call the passed function to get star names string
        ans1_str = get_star_names_func(main_stars_list, palace_name_key)
        # Split into list, sort, and join back
        ans1_list = [star.strip() for star in ans1_str.split(',') if star.strip()] # Handle potential empty strings after split
        ans1_list.sort(key=lambda x: zhuxing_order.index(x) if x in zhuxing_order else len(zhuxing_order)) # Sort based on predefined order, put unknown at the end
        ans1 = ",".join(ans1_list)


        if len(ans1) == 0:
            # print(f"No main stars found in palace {i}, checking opposite palace.")
            # Assuming the logic here is to get the opposite palace's stars if the current one is empty
            # Need to ensure the index calculation is correct and doesn't go out of bounds
            opposite_palace_index = (i + 6) % 12 # Use modulo 12 for cyclic palaces
            opposite_palace_data = palaces_data[opposite_palace_index]
            if isinstance(opposite_palace_data, dict):
                 opposite_main_stars_list = opposite_palace_data.get(zhuxing_list_key, [])
                 ans1_str_opposite = get_star_names_func(opposite_main_stars_list, palace_name_key)
                 ans1_list_opposite = [star.strip() for star in ans1_str_opposite.split(',') if star.strip()]
                 ans1_list_opposite.sort(key=lambda x: zhuxing_order.index(x) if x in zhuxing_order else len(zhuxing_order))
                 ans1 = ",".join(ans1_list_opposite)
                 # print(f"Main stars from opposite palace {opposite_palace_index}: {ans1}")
            # else:
                 # print(f"Warning: Opposite palace data at index {opposite_palace_index} is not a dictionary.")


        # 地支 (Earthly Branch)
        ans2 = palace_data.get(dizhi_key, "")

        # 宫位 (Palace Name)
        ans3 = palace_data.get(palace_name_key, "")

        # 四化 (Si Hua)
        # Call sihua function to get Si Hua string
        ans4_str = sihua(main_stars_list, palace_name_key, rihuayao_key)
        # Split into list, sort, and join back
        ans4_list = [sihua_item.strip() for sihua_item in ans4_str.split(',') if sihua_item.strip()]
        ans4_list.sort(key=lambda x: sihua_order.index(x) if x in sihua_order else len(sihua_order)) # Sort based on predefined order
        ans4 = ",".join(ans4_list)


        # 辅佐煞空 (Auxiliary Stars, Sha Stars, etc.)
        # Get the list of auxiliary/sha stars using the correct key
        fuxing_list = palace_data.get(fuxing_list_key, [])
        # Call the passed function to get star names string
        ans5_str = get_star_names_func(fuxing_list, palace_name_key)
        # Split into list, sort, and join back
        ans5_list = [star.strip() for star in ans5_str.split(',') if star.strip()]
        ans5_list.sort(key=lambda x: fuxing_order.index(x) if x in fuxing_order else len(fuxing_order)) # Sort based on predefined order
        ans5 = ",".join(ans5_list)


        # 杂曜 (Minor Stars)
        # Get the list of minor stars using the correct key
        zayao_list = palace_data.get(zayao_list_key, [])
        # Call the passed function to get star names string
        ans6_str = get_star_names_func(zayao_list, palace_name_key)
        # Minor stars are not requested to be sorted, so just use the string directly
        ans6 = ans6_str


        # Concatenate the results
        # Ensure all parts are strings before concatenation
        #ans7 = f"{ans0}{ans00}{ans000}{ans1}{ans2}{ans3}{ans4}{ans5}{ans6}"
        ans7 = f"{ans1}{ans2}{ans3}{ans4}{ans5}"
        #lists.append(ans7) # Remove commas from the final concatenated string
        results_array.append([ans1.replace(",","，"),ans2, ans3+"宫",ans4,ans5,ans6])

    return results_array


def filter_data_by_palace_names(palace_names_to_filter, data_list):
    filtered_list = []
    for data_string in data_list:
        # Check if any of the palace names are present in the data string
        if any(name in data_string for name in palace_names_to_filter):
            filtered_list.append(data_string)
    return filtered_list

def get_bad_items_in_palace(palace_data, zhuxing_key, fuxing_key, zayao_key, palace_name_key, sihua_str,
                            bad_stars_list_to_check):
    bad_items_found = []
    bad_sihua_suffix = "化忌"

    # Check Si Hua string for "化忌"
    sihua_items = [item.strip() for item in sihua_str.split(',') if item.strip()]
    for sihua_item in sihua_items:
        if sihua_item.endswith(bad_sihua_suffix):
            bad_items_found.append(sihua_item)

    # Check main stars, auxiliary/sha stars, and minor stars against the provided bad_stars_list_to_check
    for star_list_key in [zhuxing_key, fuxing_key, zayao_key]:
        stars = palace_data.get(star_list_key, [])
        if isinstance(stars, list):
            for star in stars:
                if isinstance(star, dict) and palace_name_key in star:
                    star_name = star.get(palace_name_key)
                    if star_name in bad_stars_list_to_check and star_name not in bad_items_found:  # Avoid duplicates
                        bad_items_found.append(star_name)

    return bad_items_found


def classify_palaces_run(
        palaces_data,
        keys,  # Dictionary of keys
        funcs,  # Dictionary of functions (get_star_names_func, sihua_func)
        bad_stars_list_to_check,  # List of bad stars for this run (for standard checks)
        standard_related_offsets_to_check  # List of standard palace offsets to check (0, 6, 4, 8)
):
    bad_palaces_list = []
    good_palaces_list = []

    for i in range(0, 12):
        current_palace_data = palaces_data[i]
        current_palace_name = current_palace_data.get(keys['palace_name_key'], f"未知宫位{i + 1}")

        if not isinstance(current_palace_data, dict):
            # print(f"Warning: Expected palace data at index {i} to be a dictionary, but got {type(current_palace_data)}. Skipping classification for this palace.")
            continue

        is_bad_palace = False
        reasons_for_bad_palace = []

        # --- Check standard related palaces (0, 6, 4, 8) ---
        for offset in standard_related_offsets_to_check:
            related_index = (i + offset) % 12
            related_palace_data = palaces_data[related_index]
            related_palace_name = related_palace_data.get(keys['palace_name_key'], f"未知宫位{related_index + 1}")

            if not isinstance(related_palace_data, dict):
                # print(f"Warning: Expected related palace data at index {related_index} to be a dictionary, but got {type(related_palace_data)}. Skipping check for this related palace.")
                continue

            # Get star lists and Si Hua string for the related palace
            related_main_stars_list = related_palace_data.get(keys['zhuxing_list_key'], [])
            related_fuxing_list = related_palace_data.get(keys['fuxing_list_key'], [])
            related_zayao_list = related_palace_data.get(keys['zayao_list_key'], [])

            # Calculate Si Hua string for the related palace
            related_sihua_str = funcs['sihua_func'](related_main_stars_list, keys['palace_name_key'],
                                                    keys['rihuayao_key'])

            # Get specific bad items from the related palace based on current bad_stars_list_to_check
            bad_items = get_bad_items_in_palace(
                related_palace_data,
                keys['zhuxing_list_key'],
                keys['fuxing_list_key'],
                keys['zayao_list_key'],
                keys['palace_name_key'],
                related_sihua_str,
                bad_stars_list_to_check  # Use the bad stars list for this run
            )

            if bad_items:
                is_bad_palace = True
                palace_relation = ""
                if offset == 0:
                    palace_relation = "当前宫位"
                elif offset == 6:
                    palace_relation = "对宫"
                elif offset == 4:
                    palace_relation = f"{related_index + 1}宫 (i+4)"  # Use 1-based indexing
                elif offset == 8:
                    palace_relation = f"{related_index + 1}宫 (i+8)"  # Use 1-based indexing

                for item in bad_items:
                    full_reason = f"由{palace_relation} ({related_palace_name}) 包含{item}"
                    if full_reason not in reasons_for_bad_palace:
                        reasons_for_bad_palace.append(full_reason)

        # --- New Special Check: (i+1) and (i+11) combination of 火星 and 铃星 ---
        palace_plus_1_index = (i + 1) % 12
        palace_plus_11_index = (i + 11) % 12  # Note: (i + 11) % 12 is the palace before the current one

        palace_plus_1_data = palaces_data[palace_plus_1_index]
        palace_plus_11_data = palaces_data[palace_plus_11_index]

        palace_plus_1_name = palace_plus_1_data.get(keys['palace_name_key'], f"未知宫位{palace_plus_1_index + 1}")
        palace_plus_11_name = palace_plus_11_data.get(keys['palace_name_key'], f"未知宫位{palace_plus_11_index + 1}")

        if isinstance(palace_plus_1_data, dict) and isinstance(palace_plus_11_data, dict):
            # Check if (i+1) has 火星 and (i+11) has 铃星
            has_huoxing_in_plus_1 = any(star.get(keys['palace_name_key']) == "火星" for star in
                                        palace_plus_1_data.get(keys['zhuxing_list_key'], []) + palace_plus_1_data.get(
                                            keys['fuxing_list_key'], []) + palace_plus_1_data.get(
                                            keys['zayao_list_key'], []))
            has_lingxing_in_plus_11 = any(star.get(keys['palace_name_key']) == "铃星" for star in
                                          palace_plus_11_data.get(keys['zhuxing_list_key'],
                                                                  []) + palace_plus_11_data.get(keys['fuxing_list_key'],
                                                                                                []) + palace_plus_11_data.get(
                                              keys['zayao_list_key'], []))

            if has_huoxing_in_plus_1 and has_lingxing_in_plus_11:
                is_bad_palace = True
                reason = f"由相邻宫位 ({palace_plus_1_name}) 包含火星 且 相邻宫位 ({palace_plus_11_name}) 包含铃星"
                if reason not in reasons_for_bad_palace:
                    reasons_for_bad_palace.append(reason)

            # Check if (i+1) has 铃星 and (i+11) has 火星
            has_lingxing_in_plus_1 = any(star.get(keys['palace_name_key']) == "铃星" for star in
                                         palace_plus_1_data.get(keys['zhuxing_list_key'], []) + palace_plus_1_data.get(
                                             keys['fuxing_list_key'], []) + palace_plus_1_data.get(
                                             keys['zayao_list_key'], []))
            has_huoxing_in_plus_11 = any(star.get(keys['palace_name_key']) == "火星" for star in
                                         palace_plus_11_data.get(keys['zhuxing_list_key'],
                                                                 []) + palace_plus_11_data.get(keys['fuxing_list_key'],
                                                                                               []) + palace_plus_11_data.get(
                                             keys['zayao_list_key'], []))

            if has_lingxing_in_plus_1 and has_huoxing_in_plus_11:
                is_bad_palace = True
                reason = f"由相邻宫位 ({palace_plus_1_name}) 包含铃星 且 相邻宫位 ({palace_plus_11_name}) 包含火星"
                if reason not in reasons_for_bad_palace:
                    reasons_for_bad_palace.append(reason)

        if is_bad_palace:
            bad_palaces_list.append({'name': current_palace_name, 'reasons': reasons_for_bad_palace})
        else:
            good_palaces_list.append(current_palace_name)

    return bad_palaces_list, good_palaces_list


def ziweids_day(
    converted_data,
    mingzhu_key,
    shenzhu_key,
    wuxingju_key,
    gongwei_list_key,
    zhuxing_list_key,     # Key for the list of main stars within a palace dict
    dizhi_key,
    rihuayao_key,         # Key for Ri Hua Yao (used by sihua function)
    fuxing_list_key,      # Key for the list of auxiliary/sha stars within a palace dict
    zayao_list_key,       # Key for the list of minor stars within a palace dict
    palace_name_key,      # Key for the palace name and also star name within star dicts
    get_star_names_func   # The actual function to get star names (i.e., the 'zhuxing' function)
):
    # Define sorting orders
    zhuxing_order = ["紫微","天机","太阳","武曲","天同","巨门","廉贞","天府","太阴","贪狼","天相","天梁","七杀","破军"]
    fuxing_order = ["天魁","天钺","左辅","右弼","文昌","文曲","禄存","天马","火星","铃星","擎羊","陀罗","地空","地劫"]
    sihua_order = ['紫微化禄','紫微化权','紫微化科','紫微化忌','天机化禄','天机化权','天机化科','天机化忌','太阳化禄','太阳化权','太阳化科','太阳化忌',
                   '武曲化禄','武曲化权','武曲化科','武曲化忌','天同化禄','天同化权','天同化科','天同化忌','巨门化禄','巨门化权','巨门化科',
                   '巨门化忌','廉贞化禄','廉贞化权','廉贞化科','廉贞化忌','天府化禄','天府化权','天府化科','天府化忌','太阴化禄','太阴化权',
                   '太阴化科','太阴化忌','贪狼化禄','贪狼化权','贪狼化科','贪狼化忌','天相化禄','天相化权','天相化科','天相化忌','天梁化禄',
                   '天梁化权','天梁化科','天梁化忌','七杀化禄','七杀化权','七杀化科','七杀化忌','破军化禄','破军化权','破军化科','破军化忌',
                   '文昌化禄','文昌化权','文曲化科','文曲化忌','文昌化禄','文昌化权','文昌化科','文昌化忌','文曲化禄','文曲化权','文曲化科','文曲化忌'] # Corrected duplicate entries and added missing ones based on common Si Hua

    lists = []
    # Ensure gongwei_list_key exists and is a list
    palaces_data = converted_data.get(gongwei_list_key, [])
    if not isinstance(palaces_data, list) or len(palaces_data) != 12:
         print(f"Error: Expected a list of 12 palaces for key '{gongwei_list_key}', but got {type(palaces_data)} with length {len(palaces_data) if isinstance(palaces_data, list) else 'N/A'}")
         return [] # Return empty list if palace data is not as expected


    for i in range(0, 12):
        palace_data = palaces_data[i]
        # Ensure palace_data is a dictionary
        if not isinstance(palace_data, dict):
             print(f"Warning: Expected palace data at index {i} to be a dictionary, but got {type(palace_data)}. Skipping.")
             lists.append("") # Add empty string or handle as needed
             continue

        # Safely get base data using .get()
        ans0 = converted_data.get(mingzhu_key, "")
        # print(f"Ming Zhu: {ans0}")
        ans00 = converted_data.get(shenzhu_key, "")
        # print(f"Shen Zhu: {ans00}")
        ans000 = converted_data.get(wuxingju_key, "")
        # print(f"Wu Xing Ju: {ans000}")

        # 星系 (Main Stars)
        # Get the list of main stars using the correct key
        main_stars_list = palace_data.get(zhuxing_list_key, [])
        # Call the passed function to get star names string
        ans1_str = get_star_names_func(main_stars_list, palace_name_key)
        # Split into list, sort, and join back
        ans1_list = [star.strip() for star in ans1_str.split(',') if star.strip()] # Handle potential empty strings after split
        ans1_list.sort(key=lambda x: zhuxing_order.index(x) if x in zhuxing_order else len(zhuxing_order)) # Sort based on predefined order, put unknown at the end
        ans1 = ",".join(ans1_list)


        if len(ans1) == 0:
            # print(f"No main stars found in palace {i}, checking opposite palace.")
            # Assuming the logic here is to get the opposite palace's stars if the current one is empty
            # Need to ensure the index calculation is correct and doesn't go out of bounds
            opposite_palace_index = (i + 6) % 12 # Use modulo 12 for cyclic palaces
            opposite_palace_data = palaces_data[opposite_palace_index]
            if isinstance(opposite_palace_data, dict):
                 opposite_main_stars_list = opposite_palace_data.get(zhuxing_list_key, [])
                 ans1_str_opposite = get_star_names_func(opposite_main_stars_list, palace_name_key)
                 ans1_list_opposite = [star.strip() for star in ans1_str_opposite.split(',') if star.strip()]
                 ans1_list_opposite.sort(key=lambda x: zhuxing_order.index(x) if x in zhuxing_order else len(zhuxing_order))
                 ans1 = ",".join(ans1_list_opposite)
                 # print(f"Main stars from opposite palace {opposite_palace_index}: {ans1}")
            # else:
                 # print(f"Warning: Opposite palace data at index {opposite_palace_index} is not a dictionary.")


        # 地支 (Earthly Branch)
        ans2 = palace_data.get(dizhi_key, "")

        # 宫位 (Palace Name)
        ans3 = palace_data.get(palace_name_key, "")

        # 四化 (Si Hua)
        # Call sihua function to get Si Hua string
        ans4_str = sihua(main_stars_list, palace_name_key, rihuayao_key)
        # Split into list, sort, and join back
        ans4_list = [sihua_item.strip() for sihua_item in ans4_str.split(',') if sihua_item.strip()]
        ans4_list.sort(key=lambda x: sihua_order.index(x) if x in sihua_order else len(sihua_order)) # Sort based on predefined order
        ans4 = ",".join(ans4_list)


        # 辅佐煞空 (Auxiliary Stars, Sha Stars, etc.)
        # Get the list of auxiliary/sha stars using the correct key
        fuxing_list = palace_data.get(fuxing_list_key, [])
        # Call the passed function to get star names string
        ans5_str = get_star_names_func(fuxing_list, palace_name_key)
        # Split into list, sort, and join back
        ans5_list = [star.strip() for star in ans5_str.split(',') if star.strip()]
        ans5_list.sort(key=lambda x: fuxing_order.index(x) if x in fuxing_order else len(fuxing_order)) # Sort based on predefined order
        ans5 = ",".join(ans5_list)


        # 杂曜 (Minor Stars)
        # Get the list of minor stars using the correct key
        zayao_list = palace_data.get(zayao_list_key, [])
        # Call the passed function to get star names string
        ans6_str = get_star_names_func(zayao_list, palace_name_key)
        # Minor stars are not requested to be sorted, so just use the string directly
        ans6 = ans6_str


        # Concatenate the results
        # Ensure all parts are strings before concatenation
        #ans7 = f"{ans0}{ans00}{ans000}{ans1}{ans2}{ans3}{ans4}{ans5}{ans6}"
        ans7 = f"{ans1}{ans2}{ans3}{ans4}{ans5}"
        lists.append(ans7.replace(",","")) # Remove commas from the final concatenated string

    return lists


def get_earthly_branches_from_palace_names(palace_names_list: list, palace_details_list: list) -> list:
    """
    根据宫位名称列表，从宫位详细信息列表中提取对应的地支。

    Args:
        palace_names_list (list): 宫位名称的列表，例如 ['命宫', '兄弟宫']。
        palace_details_list (list): 包含所有宫位详细信息的列表，
                                   每个元素是一个字典，包含 '宫位' 和 '地支' 等键。

    Returns:
        list: 对应宫位的地支列表。
    """
    #print(palace_details_list)
    earthly_branches = []
    for palace_name in palace_names_list:
        #print("*****", palace_details_list)
        #print("*****", palace_name)
        found_palace = False
        for palace_data in palace_details_list:
            #print("*****", palace_data)

            # 确保 palace_data 是一个字典
            if isinstance(palace_data, dict):
                if palace_data.get('名称') == palace_name:

                    found_palace = True
                    if '地支' in palace_data:
                        #print(palace_data['地支'])
                        earthly_branches.append(palace_data['地支'])
                    else:
                        logger.warning(f"宫位 '{palace_name}' 的数据中未找到 '地支' 键。宫位数据: {palace_data}")
                    break  # 找到该宫位后，跳出内层循环，处理下一个宫位名称
            # 如果 palace_details_list 中包含非字典元素，在调用前已过滤，此处不再需要详细警告
        if not found_palace:
            logger.warning(f"在宫位详细信息中未找到宫位 '{palace_name}' 的匹配数据。")
    return earthly_branches


def ziweids4(
        converted_data,
        mingzhu_key,
        shenzhu_key,
        wuxingju_key,
        gongwei_list_key,
        zhuxing_list_key,  # Key for the list of main stars within a palace dict
        dizhi_key,
        rihuayao_key,  # Key for Ri Hua Yao (used by sihua function)
        fuxing_list_key,  # Key for the list of auxiliary/sha stars within a palace dict
        zayao_list_key,  # Key for the list of minor stars within a palace dict
        palace_name_key,  # Key for the palace name and also star name within star dicts
        get_star_names_func,  # The actual function to get star names (i.e., the 'zhuxing' function)
        sihua_func  # The actual function to get Si Hua information (i.e., the 'sihua' function)
):
    # Define sorting orders (still needed if you want to process star lists before checking)
    # These sorting orders are not directly used in the classification logic itself,
    # but might be used if you were to format the star lists within the palace output.
    # Keeping them here for reference if needed elsewhere.
    zhuxing_order = ["紫微", "天机", "太阳", "武曲", "天同", "巨门", "廉贞", "天府", "太阴", "贪狼", "天相", "天梁",
                     "七杀", "破军"]
    fuxing_order = ["天魁", "天钺", "左辅", "右弼", "文昌", "文曲", "禄存", "天马", "火星", "铃星", "擎羊", "陀罗",
                    "地空", "地劫"]
    sihua_order = ['紫微化禄', '紫微化权', '紫微化科', '紫微化忌', '天机化禄', '天机化权', '天机化科', '天机化忌',
                   '太阳化禄', '太阳化权', '太阳化科', '太阳化忌',
                   '武曲化禄', '武曲化权', '武曲化科', '武曲化忌', '天同化禄', '天同化权', '天同化科', '天同化忌',
                   '巨门化禄', '巨门化权', '巨门化科',
                   '巨门化忌', '廉贞化禄', '廉贞化权', '廉贞化科', '廉贞化忌', '天府化禄', '天府化权', '天府化科',
                   '天府化忌', '太阴化禄', '太阴化权',
                   '太阴化科', '太阴化忌', '贪狼化禄', '贪狼化权', '贪狼化科', '贪狼化忌', '天相化禄', '天相化权',
                   '天相化科', '天相化忌', '天梁化禄',
                   '天梁化权', '天梁化科', '天梁化忌', '七杀化禄', '七杀化权', '七杀化科', '七杀化忌', '破军化禄',
                   '破军化权', '破军化科', '破军化忌',
                   '文昌化禄', '文昌化权', '文曲化科', '文曲化忌', '文昌化禄', '文昌化权', '文昌化科', '文昌化忌',
                   '文曲化禄', '文曲化权', '文曲化科', '文曲化忌']

    # Ensure gongwei_list_key exists and is a list
    palaces_data = converted_data.get(gongwei_list_key, [])
    if not isinstance(palaces_data, list) or len(palaces_data) != 12:
        print(
            f"Error: Expected a list of 12 palaces for key '{gongwei_list_key}', but got {type(palaces_data)} with length {len(palaces_data) if isinstance(palaces_data, list) else 'N/A'}")
        return {'bad_palaces': [], 'good_palaces': []}  # Return empty lists in a dict

    # Store keys and functions in dictionaries for easier passing to helper
    keys = {
        'mingzhu_key': mingzhu_key,
        'shenzhu_key': shenzhu_key,
        'wuxingju_key': wuxingju_key,
        'gongwei_list_key': gongwei_list_key,
        'zhuxing_list_key': zhuxing_list_key,
        'dizhi_key': dizhi_key,
        'rihuayao_key': rihuayao_key,
        'fuxing_list_key': fuxing_list_key,
        'zayao_list_key': zayao_list_key,
        'palace_name_key': palace_name_key,
    }
    funcs = {
        'get_star_names_func': get_star_names_func,
        'sihua_func': sihua_func,
    }

    # --- Iteration 1: Initial Classification ---
    print("--- Initial Classification ---")
    initial_bad_stars = ["火星", "铃星", "擎羊", "陀罗", "地空", "地劫"]
    # Standard offsets: Current (0), Opposite (6), +4, +8. Offset 1 is handled separately now.
    initial_standard_offsets = [0, 6, 4, 8]

    bad_palaces_1, good_palaces_1 = classify_palaces_run(
        palaces_data,
        keys,
        funcs,
        initial_bad_stars,
        initial_standard_offsets  # Pass standard offsets
    )
    classification_result = {'bad_palaces': bad_palaces_1, 'good_palaces': good_palaces_1}
    bad_palace_names = [palace_info['name'] for palace_info in classification_result.get('bad_palaces', [])]

    if good_palaces_1:
        print("Initial classification found good palaces.")
        return classification_result, bad_palace_names, good_palaces_1

    # --- Iteration 2: First Fallback (Exclude 地空, 地劫) ---
    print("--- Fallback 1: Exclude 地空, 地劫 ---")
    fallback1_bad_stars = ["火星", "铃星", "擎羊", "陀罗"]  # Exclude 地空, 地劫
    fallback1_standard_offsets = [0, 6, 4, 8]  # Same standard offsets as initial

    bad_palaces_2, good_palaces_2 = classify_palaces_run(
        palaces_data,
        keys,
        funcs,
        fallback1_bad_stars,
        fallback1_standard_offsets  # Pass standard offsets
    )
    classification_result = {'bad_palaces': bad_palaces_2, 'good_palaces': good_palaces_2}
    bad_palace_names = [palace_info['name'] for palace_info in classification_result.get('bad_palaces', [])]

    if good_palaces_2:
        print("Fallback 1 found good palaces.")

        return classification_result, bad_palace_names, good_palaces_2

    # --- Iteration 3: Second Fallback (Exclude i+4, i+8 influence) ---
    print("--- Fallback 2: Exclude i+4, i+8 influence ---")
    fallback2_bad_stars = ["火星", "铃星", "擎羊", "陀罗"]  # Same as fallback 1
    fallback2_standard_offsets = [0, 6]  # Only check Current (0) and Opposite (6)

    bad_palaces_3, good_palaces_3 = classify_palaces_run(
        palaces_data,
        keys,
        funcs,
        fallback2_bad_stars,
        fallback2_standard_offsets  # Pass standard offsets
    )
    classification_result = {'bad_palaces': bad_palaces_3, 'good_palaces': good_palaces_3}
    bad_palace_names = [palace_info['name'] for palace_info in classification_result.get('bad_palaces', [])]

    print("Fallback 2 finished.")
    # Return the result of the last fallback regardless of whether good palaces were found
    return classification_result, bad_palace_names, good_palaces_3
