import pandas as pd
import pymysql
import traceback
import os
import openai
import json
import asyncio
import time as time_module
from datetime import datetime, time, timedelta
from fastapi import FastAPI, HTTPException, status, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool  # 用于在线程池中运行同步代码
import random  # 导入 random 模块用于生成随机数
from dotenv import load_dotenv

# --- FastAPI App Initialization ---
app = FastAPI(
    title="奇门遁甲时辰吉凶API",
    description="根据用户生日和查询日期，提供当天各时辰的宜忌事项分析。",
    version="1.0.0",
)
router = APIRouter()
load_dotenv()

# --- 1. vLLM 和 数据库 配置 ---
print("开始执行基于LLM的奇门遁甲API服务脚本 (V33 - 时辰串行 + 时辰内并发)...")

VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL", "http://192.168.1.201/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME2", "Qwen3-Next-80B-A3B-Instruct-FP8")
VLLM_API_KEY = os.getenv("API_KEY", "not-needed")

try:
    client = openai.AsyncOpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_BASE_URL)
    print(f"异步客户端已创建，目标 vLLM 模型 '{VLLM_MODEL_NAME}' at '{VLLM_API_BASE_URL}'")
except Exception as e:
    print(f"创建AsyncOpenAI客户端失败: {e}")
    exit()

# --- LLM 并发控制 ---
# 这个信号量是全局的
LLM_SEMAPHORE = asyncio.Semaphore(1000) 

DB_CONFIG = {
    'user': os.getenv('DB_USER', 'proxysql'),
    'password': os.getenv('DB_PASSWORD', '87IdR6@1XI5E'),
    'host': os.getenv('DB_HOST', '192.168.1.101'),
    'port': int(os.getenv('DB_PORT', '6033')),
    'database': os.getenv('DB_NAME', 'yingshi'),
    'cursorclass': pymysql.cursors.DictCursor,  # Ensure dict results
    'charset': 'utf8mb4'
}

# --- 2. 加载【全部】核心知识库 ---
KNOWLEDGE_DFS = {}

def robust_read_csv(filepath: str) -> pd.DataFrame:
    """健壮地读取CSV文件，尝试多种编码"""
    for encoding in ['utf-8', 'gbk', 'gb18030', 'utf-8-sig']:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    raise Exception(f"无法读取文件: {filepath}")

def load_all_knowledge_data() -> bool:
    """加载所有奇门遁甲知识库维表"""
    print("\n--- 步骤 1: 正在加载【全部】知识库维表 ---")
    try:
        script_dir = os.path.dirname(__file__)
        assets_dir = os.path.join(script_dir, 'assets')

        KNOWLEDGE_DFS['bamen'] = robust_read_csv(os.path.join(assets_dir, 'bamen.csv'))
        KNOWLEDGE_DFS['bamen_bamen'] = robust_read_csv(os.path.join(assets_dir, 'bamen+bamen.csv'))
        KNOWLEDGE_DFS['bashen'] = robust_read_csv(os.path.join(assets_dir, 'bashen.csv'))
        KNOWLEDGE_DFS['bashen_bamen'] = robust_read_csv(os.path.join(assets_dir, 'bashen+bamen.csv'))
        KNOWLEDGE_DFS['dizhi'] = robust_read_csv(os.path.join(assets_dir, 'dizhi.csv'))
        KNOWLEDGE_DFS['jiugong'] = robust_read_csv(os.path.join(assets_dir, 'jiugong.csv'))
        KNOWLEDGE_DFS['jiuxing'] = robust_read_csv(os.path.join(assets_dir, 'jiuxing.csv'))
        KNOWLEDGE_DFS['qiyi'] = robust_read_csv(os.path.join(assets_dir, 'qiyizuhe.csv'))
        KNOWLEDGE_DFS['shigan_bamen'] = robust_read_csv(os.path.join(assets_dir, 'shigan+bamen.csv'))
        KNOWLEDGE_DFS['tiangan'] = robust_read_csv(os.path.join(assets_dir, 'tiangan.csv'))
        print("所有 10 个奇门知识库维表已全部加载成功。")
        return True
    except Exception as e:
        print(f"加载知识库维表时发生致命错误: {e}")
        traceback.print_exc()
        return False

def get_lookup_info(df_key: str, column_map: dict, target_col_override: str = None) -> str:
    """从知识库中查询信息"""
    df = KNOWLEDGE_DFS.get(df_key)
    if df is None: return f"知识库 '{df_key}' 未加载。"
    valid_column_map = {k: v for k, v in column_map.items() if pd.notna(v)}
    if not valid_column_map: return "查询条件不足"
    condition = pd.Series([True] * len(df))
    for col, val in valid_column_map.items(): condition &= (df[col] == val)
    res = df[condition]
    if res.empty: return "无"
    if target_col_override and target_col_override in res.columns: return str(res.iloc[0][target_col_override])
    for col in ['信息解读', '基础含义', '代表含义', '信息象意']:
        if col in res.columns and pd.notna(res.iloc[0][col]): return str(res.iloc[0][col])
    return "未找到描述信息"

# --- 应用启动时加载知识库 ---
@router.on_event("startup")
async def startup_event():
    if not load_all_knowledge_data():
        raise RuntimeError("Failed to load knowledge data on startup.")
    print("知识库加载完成，FastAPI 应用准备就绪。")


# --- 3. 核心功能：规则引擎、生成描述的LLM Prompt (异步化) ---

def calculate_keynote_result(jixiong: str, gong_status: str) -> str:
    """根据吉凶和宫位关系判断核心结果"""
    if jixiong is None or gong_status is None:
        return "不宜进行"
    if jixiong == '吉' and gong_status in ['吉', '平']: return "有宜进行"
    if jixiong in ['平'] and gong_status == '吉': return "有宜进行"
    return "不宜进行"


async def generate_tag_description(chart_data_row: dict, specific_matter: str, keynote_result: str) -> tuple[str, int, int, float, bool]:
    """
    为给定的事项生成简洁、通俗易懂的描述。
    """
    prompt_content = ""
    
    # --- 构建奇门要素描述段落 ---
    qimen_elements_intro = "结合奇门局分析，当前运势由以下要素影响："
    qimen_elements_list = []
    
    tianpan_xing = chart_data_row.get('tianpanXing_ori', '无')
    shenpan_god = chart_data_row.get('shenpan', '无')
    tianpan_gan = chart_data_row.get('tianpanGan', '无')
    dipan_gan = chart_data_row.get('dipan', '无')
    renpan_door = chart_data_row.get('renpan', '无') 

    tianpan_xing_meaning = get_lookup_info('jiuxing', {'九星': tianpan_xing}) if tianpan_xing != '无' else '无'
    shenpan_god_meaning = get_lookup_info('bashen', {'八神': shenpan_god}) if shenpan_god != '无' else '无'
    renpan_meaning = get_lookup_info('bamen', {'八门': renpan_door}) if renpan_door != '无' else '无'
    qiyi_combo_str = f"{tianpan_gan}+{dipan_gan}" if tianpan_gan != '无' and dipan_gan != '无' else '无+无'
    qiyi_interpretation = get_lookup_info('qiyi', {'天盘天干': tianpan_gan, '地盘天干': dipan_gan}) if tianpan_gan != '无' and dipan_gan != '无' else '无'

    if tianpan_xing != '无' and tianpan_xing_meaning != '无':
        qimen_elements_list.append(f"天盘九星【{tianpan_xing}星】，其象意是：{tianpan_xing_meaning}。")
    if renpan_door != '无' and renpan_meaning != '无':
        qimen_elements_list.append(f"人盘八门【{renpan_door}门】，代表着：{renpan_meaning}。")
    if shenpan_god != '无' and shenpan_god_meaning != '无':
        qimen_elements_list.append(f"神盘八神【{shenpan_god}】，其能量为：{shenpan_god_meaning}。")
    if tianpan_gan != '无' and dipan_gan != '无' and qiyi_interpretation != '无' and qiyi_combo_str != '无+无':
        qimen_elements_list.append(f"奇仪组合【{qiyi_combo_str}】，预示着：{qiyi_interpretation}。")

    full_qimen_analysis_segment = qimen_elements_intro + "\n" + "\n".join(qimen_elements_list) + "\n\n"
    if not qimen_elements_list: 
        full_qimen_analysis_segment = "当前奇门局势如下：" 

    if keynote_result == '有宜进行':
        prompt_content = f"""
你是一位专业的奇门遁甲分析师，现在需要为客户生成一个针对“{specific_matter}”事项的简短、积极的行动建议。
核心基调是“有宜进行”。
{full_qimen_analysis_segment}
请你综合上述奇门要素的含义，以**通俗易懂的直白语言**，说明此事为何顺利，以及会有哪些好结果或积极体验。
务必保持正向语气，使用“可以”“适合”“顺利”“建议去做”等表达，不得出现“不宜”“避免”“谨慎”“阻碍”等消极词。
全程不使用任何比喻，直接给出结论，输出内容控制在150字以内。
范例：
"这个时间去KTV会非常尽兴。当前天盘九星【天辅星】，其象意是：辅助吉利，利于发展。人盘八门【开门】，代表着：顺利开创，事业发展。神盘八神【值符】，其能量为：贵人相助，官方支持。奇仪组合【丙+戊】，预示着：财源广进，合作顺畅。总体而言，此刻条件成熟，放手去做即可获得良好收益。"
要求：
- 不使用比喻，不超过150字。
- 所有奇门专有名词都用【】括起来（如【景门】【值符】等）。
- **重要：只使用重点信息中明确提到的奇门要素，不要添加或编造其他要素。**
"""
    else: 
        prompt_content = f"""
你是一位专业的奇门遁甲分析师，现在需要为客户生成一个针对“{specific_matter}”事项的简短、谨慎的提醒。
核心基调是“不宜进行”。
{full_qimen_analysis_segment}
请你综合上述奇门要素的含义，以**通俗易懂的直白语言**，说明此事可能遇到的阻碍、挑战，或建议避免的原因。
语气需明确表达“谨慎/不宜进行”，可以使用“避免”“谨慎”“风险”“推迟”等词，不能出现鼓励去做的表述。
全程不使用任何比喻，直接给出结论，输出内容控制在150字以内。
范例：
"这个时间进行理发可能不会达到预期效果。当前天盘九星【天柱星】，其象意是：孤高易折，阻碍重重。人盘八门【伤门】，代表着：受伤损耗，竞争激烈。神盘八神【腾蛇】，其能量为：虚假纠缠，反复无常。奇仪组合【戊+己】，预示着：贵人入狱，资金受困。所以，此时不宜仓促行动，容易遇到预想不到的麻烦和不顺，最好推迟，待时机好转再处理。"
要求：
- 不使用比喻，不超过150字。
- 所有奇门专有名词都用【】括起来（如【景门】【值符】等）。
- **重要：只使用重点信息中明确提到的奇门要素，不要添加或编造其他要素。**
"""
    
    try:
        llm_start_time = time_module.perf_counter()
        # 使用全局信号量控制并发数
        async with LLM_SEMAPHORE:
            response = await client.chat.completions.create(
                model=VLLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt_content}],
                temperature=0.7,
                max_tokens=250
            )
        llm_end_time = time_module.perf_counter()
        llm_call_time = llm_end_time - llm_start_time
        
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        return (
            response.choices[0].message.content.strip(),
            prompt_tokens,
            completion_tokens,
            llm_call_time,
            False,
        )
    except Exception as e:
        print(f"LLM生成'{specific_matter}'描述失败: {e}")
        return f"未能生成具体描述。请稍后重试。", 0, 0, 0.0, True


# --- 辅助函数 ---
TIANGAN_LIST = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
REFERENCE_DATE = datetime(1900, 1, 1) 
REFERENCE_STEM_INDEX = 7 

def get_day_stem_from_gregorian_date(target_datetime: datetime) -> str:
    days_diff = (target_datetime.date() - REFERENCE_DATE.date()).days
    day_stem_index = (REFERENCE_STEM_INDEX + days_diff) % 10
    return TIANGAN_LIST[day_stem_index]

def time_to_chinese_hour_segment(time_obj: time) -> tuple[str, str]:
    hour = time_obj.hour
    if 7 <= hour < 9: return "辰时", "7:00"
    elif 9 <= hour < 11: return "巳时", "9:00"
    elif 11 <= hour < 13: return "午时", "11:00"
    elif 13 <= hour < 15: return "未时", "13:00"
    elif 15 <= hour < 17: return "申时", "15:00"
    elif 17 <= hour < 19: return "酉时", "17:00"
    elif 19 <= hour < 21: return "戌时", "19:00"
    elif 21 <= hour < 23: return "亥时", "21:00"
    elif hour == 23: return "亥时", "21:00"
    else: return None, None


# --- 数据库操作函数 (同步) ---
def _get_chart_data_from_db(target_date_dt: datetime, tian_gan_for_person: str, db_config: dict) -> list[dict]:
    conn = None
    all_chart_data = []
    try:
        conn = pymysql.connect(**db_config)
        with conn.cursor() as cursor:
            gong_status_col = f'gong_relation_status_{tian_gan_for_person}'

            query_cols = [
                'id', 'date_str', '吉凶', 'total_score', '具体事项',
                'tianpanGan', 'dipan', 'renpan', 'shenpan', 'tianpanXing_ori',
                gong_status_col
            ]
            query_cols_str = ", ".join(f"`{col}`" for col in query_cols)

            start_datetime = target_date_dt.replace(hour=7, minute=0, second=0)
            end_datetime = target_date_dt.replace(hour=23, minute=59, second=59)

            query = f"""
                SELECT {query_cols_str}
                FROM qimen_interpreted_analysis
                WHERE `date_str` BETWEEN %s AND %s
                ORDER BY `date_str` ASC, `total_score` DESC
            """
            cursor.execute(query, (start_datetime.strftime("%Y-%m-%d %H:%M:%S"), end_datetime.strftime("%Y-%m-%d %H:%M:%S")))
            all_chart_data = cursor.fetchall()
        print(f"从数据库获取到 {len(all_chart_data)} 条记录。")
    except Exception as db_err:
        traceback.print_exc()
        raise RuntimeError(f"数据库查询失败: {str(db_err)}") from db_err
    finally:
        if conn:
            conn.close()
    return all_chart_data


class RequestData(BaseModel):
    gender: str = Field(..., description="性别 (男、女)", example="男")
    birthday: datetime = Field(..., description="公历生日 (yyyy-MM-dd HH:mm:ss格式)", example="1991-01-01 12:30:00")
    date: str = Field(..., description="日期 (yyyy-MM-dd格式)", example="2025-11-24")

class TagInfo(BaseModel):
    tag: str = Field(..., description="具体事项标签")
    tagDesc: str = Field(..., description="事项描述")

class TimeInfo(BaseModel):
    startTime: str = Field(..., description="时辰开始时间，例如 7:00")
    startTimeInChinese: str = Field(..., description="中文时辰名称，例如 辰时")
    suggestions: list[TagInfo] = Field(..., description="宜进行事项列表")
    avoidances: list[TagInfo] = Field(..., description="不宜进行事项列表")

class QiMenTimeCalendarResponse(BaseModel):
    qiMenTimeInfo: list[TimeInfo] = Field(..., description="各时辰详情信息列表 (辰时到亥时 7点-23点)")


# --- 单个时辰处理逻辑 (时辰内并行) ---
async def process_single_hour_segment(
    hour_name: str, 
    display_time: str, 
    hour_data_entries: list,
    gong_status_col: str
) -> dict:
    """
    处理单个时辰的所有逻辑：筛选建议/避免事项 -> 生成LLM任务 -> 并行执行 -> 聚合结果
    """
    current_hour_info = TimeInfo(
        startTime=display_time,
        startTimeInChinese=hour_name,
        suggestions=[],
        avoidances=[]
    )
    
    stats = {
        "input_tokens": 0,
        "output_tokens": 0,
        "llm_time": 0.0,
        "llm_failure": False,
        "hour_info": current_hour_info,
        "llm_count": 0
    }

    if not hour_data_entries:
        current_hour_info.avoidances.clear()
        current_hour_info.avoidances.append(TagInfo(tag="诸事不宜", tagDesc="诸事不宜"))
        return stats

    suggestions_candidates = []
    avoidances_candidates = []

    for entry_row in hour_data_entries:
        person_gong_status = entry_row.get(gong_status_col)
        keynote_res = calculate_keynote_result(entry_row['吉凶'], person_gong_status)
        entry_row['keynote_result_for_person'] = keynote_res
        
        if keynote_res == "有宜进行":
            suggestions_candidates.append(entry_row)
        else:
            avoidances_candidates.append(entry_row)

    llm_tasks = []
    task_metadata = []

    # --- 固定条目数 (3) for suggestions ---
    num_suggestions_to_take = min(3, len(suggestions_candidates))
    sorted_suggestions = sorted(suggestions_candidates, key=lambda x: x['total_score'] if x.get('total_score') is not None else -1, reverse=True)
    seen_tags_sug = set()
    sug_count = 0
    for s_row in sorted_suggestions:
        if sug_count >= num_suggestions_to_take: break
        matter_tag = s_row.get('具体事项')
        if matter_tag and matter_tag not in seen_tags_sug:
            llm_tasks.append(generate_tag_description(s_row, matter_tag, s_row['keynote_result_for_person']))
            task_metadata.append({'type': 'suggestion', 'tag': matter_tag})
            seen_tags_sug.add(matter_tag)
            sug_count += 1
    
    # --- 固定条目数 (3) for avoidances ---
    num_avoidances_to_take = min(3, len(avoidances_candidates))
    sorted_avoidances = sorted(avoidances_candidates, key=lambda x: x['total_score'] if x.get('total_score') is not None else float('inf'))
    seen_tags_avd = set()
    avd_count = 0
    for a_row in sorted_avoidances:
        if avd_count >= num_avoidances_to_take: break
        matter_tag = a_row.get('具体事项')
        if matter_tag and matter_tag not in seen_tags_avd:
            llm_tasks.append(generate_tag_description(a_row, matter_tag, a_row['keynote_result_for_person']))
            task_metadata.append({'type': 'avoidance', 'tag': matter_tag})
            seen_tags_avd.add(matter_tag)
            avd_count += 1

    if llm_tasks:
        stats["llm_count"] = len(llm_tasks)
        print(f"[{hour_name}] 准备执行 {len(llm_tasks)} 个LLM任务...")
        
        # 在时辰内部，依然使用 gather 并发执行 LLM 请求
        all_llm_results = await asyncio.gather(*llm_tasks)
        
        desc_index = 0
        for meta in task_metadata:
            (description, input_tokens, output_tokens, 
             call_time, had_error) = all_llm_results[desc_index]
            
            stats["input_tokens"] += input_tokens
            stats["output_tokens"] += output_tokens
            stats["llm_time"] += call_time
            if had_error:
                stats["llm_failure"] = True

            if meta['type'] == 'suggestion':
                current_hour_info.suggestions.append(TagInfo(tag=meta['tag'], tagDesc=description))
            else: # type == 'avoidance'
                current_hour_info.avoidances.append(TagInfo(tag=meta['tag'], tagDesc=description))
            desc_index += 1

    # --- 兜底逻辑 ---
    if not current_hour_info.suggestions:
        current_hour_info.avoidances.clear()
        current_hour_info.avoidances.append(TagInfo(tag="诸事不宜", tagDesc="诸事不宜"))

    return stats


# --- FastAPI 主路由 (已改回：时辰串行) ---
@router.post("/qiMenTimeCalendar", response_model=QiMenTimeCalendarResponse, summary="获取奇门遁甲时辰宜忌日历")
async def qi_men_time_calendar(request_data: RequestData):
    """
    根据用户提供的生日和查询日期，计算并返回当天各时辰（辰时到亥时）的宜忌事项。
    逻辑：时辰间串行处理，时辰内并发生成描述。
    """
    start_time = time_module.perf_counter()
    llm_total = 0.0
    
    # 统计汇总
    total_accumulated_input_tokens = 0
    total_accumulated_output_tokens = 0
    total_llm_call_time = 0.0 
    total_llm_count = 0 
    llm_failure_detected = False

    try:
        birthday_dt = request_data.birthday
        try:
            target_date_dt = datetime.strptime(request_data.date, "%Y-%m-%d").replace(hour=0, minute=0, second=0, microsecond=0)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"日期格式错误: {request_data.date}")

        tian_gan_for_person = get_day_stem_from_gregorian_date(birthday_dt)
        gong_status_col = f'gong_relation_status_{tian_gan_for_person}' 
        print(f"根据生日 {birthday_dt} 计算得到的日干是: {tian_gan_for_person}")

        try:
            # 数据库查询
            all_chart_data_for_day = await run_in_threadpool(
                _get_chart_data_from_db,
                target_date_dt,
                tian_gan_for_person,
                DB_CONFIG
            )
        except RuntimeError as db_err:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"数据库查询失败: {str(db_err)}")

        # 预处理数据：按时辰分组
        grouped_by_chinese_hour = {}
        for row in all_chart_data_for_day:
            db_datetime_obj = row['date_str']
            if not isinstance(db_datetime_obj, datetime):
                try:
                    db_datetime_obj = datetime.strptime(str(db_datetime_obj), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
            chinese_hour_name, start_time_of_segment = time_to_chinese_hour_segment(db_datetime_obj.time())
            if chinese_hour_name:
                if chinese_hour_name not in grouped_by_chinese_hour:
                    grouped_by_chinese_hour[chinese_hour_name] = {
                        "startTime": start_time_of_segment,
                        "startTimeInChinese": chinese_hour_name,
                        "raw_entries": []
                    }
                grouped_by_chinese_hour[chinese_hour_name]["raw_entries"].append(row)

        chinese_hour_segments = [
            ("辰时", "7:00"), ("巳时", "9:00"), ("午时", "11:00"),
            ("未时", "13:00"), ("申时", "15:00"), ("酉时", "17:00"),
            ("戌时", "19:00"), ("亥时", "21:00")
        ]

        final_qi_men_time_info = []

        print(f"开始串行处理当天 {len(chinese_hour_segments)} 个时辰...")
        for hour_name, display_time in chinese_hour_segments:
            hour_data_entries = grouped_by_chinese_hour.get(hour_name, {}).get("raw_entries", [])
            
            hour_llm_time_start = time_module.perf_counter()

            stat = await process_single_hour_segment(
                hour_name, 
                display_time, 
                hour_data_entries, 
                gong_status_col
            )
            hour_llm_time =  time_module.perf_counter() - hour_llm_time_start
            print(f"{hour_name}-llm Execution Time: {hour_llm_time:.2f} seconds")

            llm_total += hour_llm_time
            # 累加统计数据
            total_accumulated_input_tokens += stat["input_tokens"]
            total_accumulated_output_tokens += stat["output_tokens"]
            total_llm_call_time += stat["llm_time"]
            total_llm_count += stat["llm_count"]
            if stat["llm_failure"]:
                llm_failure_detected = True
            
            final_qi_men_time_info.append(stat["hour_info"])

        elapsed_time = time_module.perf_counter() - start_time
        
        print(f"\n--- API Request Summary ---")
        print(f"Total API Execution Time: {elapsed_time:.2f} seconds")
        print(f"Total API Execution Times: {total_llm_count}")
        print(f"Total LLM Execution Times: {llm_total}seconds")
        print(f"Total LLM Tokens: In={total_accumulated_input_tokens}, Out={total_accumulated_output_tokens}")
        print(f"---------------------------")

        response_payload = QiMenTimeCalendarResponse(qiMenTimeInfo=final_qi_men_time_info)
        if llm_failure_detected:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=response_payload.model_dump(),
            )
        return response_payload
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"服务器内部错误: {str(e)}")

app.include_router(router)