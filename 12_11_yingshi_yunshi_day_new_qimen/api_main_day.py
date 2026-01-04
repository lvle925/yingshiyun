import pandas as pd
import pymysql
import traceback
import os
import openai
import json
import asyncio
import re
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
    title="奇门遁甲择时功能API",
    description="根据用户生日，目标事情和查询日期范围，提供各适宜行动的时辰和原因。",
    version="1.0.0",
)
router = APIRouter()

# --- 1. vLLM 和 数据库 配置 ---
print("开始执行基于LLM的奇门遁甲API服务脚本 (V29 - FastAPI 异步优化版带随机条目数,'诸事不宜'直出及日干计算)...")

# 加载.env中的环境变量，便于本地开发与容器部署
load_dotenv()

VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL", "http://192.168.1.201/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME2", "Qwen3-Next-80B-A3B-Instruct-FP8")
VLLM_API_KEY = os.getenv("API_KEY", "not-needed")

try:
    client = openai.AsyncOpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_BASE_URL)
    print(f"异步客户端已创建，目标 vLLM 模型 '{VLLM_MODEL_NAME}' at '{VLLM_API_BASE_URL}'")
except Exception as e:
    print(f"创建AsyncOpenAI客户端失败: {e}");
    exit()

# --- LLM 并发控制 ---
LLM_SEMAPHORE = asyncio.Semaphore(1000)  # 最多10个并发LLM调用

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


class ChooseGoodDayRequest(BaseModel):
    gender: str = Field(..., description="性别 (男、女)", example="男")
    birthday: datetime = Field(..., description="公历生日 (yyyy-MM-dd HH:mm:ss格式)", example="1991-01-01 12:30:00")
    tag: str = Field(..., description="标签 (例如'买彩票')", example="买彩票")
    startDate: str = Field(..., description="开始日期 (yyyy-MM-dd格式)", example="2025-11-24")
    endDate: str = Field(..., description="截止日期 (yyyy-MM-dd格式)", example="2025-12-24")


class GoodDay(BaseModel):
    dateTime: str = Field(..., description="时间点 (yyyy-MM-dd HH:mm:ss格式)")
    reason: str = Field(..., description="原因")


class ChooseGoodDayResponse(BaseModel):
    goodDayList: list[GoodDay] = Field(..., description="选出的好日子列表")


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
        print("所有 10 个奇门知识库维表已全部加载成功。");
        return True
    except Exception as e:
        print(f"加载知识库维表时发生致命错误: {e}");
        traceback.print_exc();
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


# --- 八门吉凶映射表 ---
BAMEN_JIXIONG_MAP = {
    '开门': '吉',
    '休门': '吉',
    '生门': '吉',
    '景门': '平',
    '死门': '凶',
    '惊门': '凶',
    '伤门': '凶',
    '杜门': '凶',
    '开': '吉',
    '休': '吉',
    '生': '吉',
    '景': '平',
    '死': '凶',
    '惊': '凶',
    '伤': '凶',
    '杜': '凶',
}

# --- 八神吉凶映射表 ---
BASHEN_JIXIONG_MAP = {
    '值符': '吉',
    '太阴': '吉',
    '六合': '吉',
    '九地': '吉',
    '九天': '吉',
    '白虎': '平',
    '玄武': '平',
    '腾蛇': '凶',
}

# --- 九星吉凶映射表 ---
JIUXING_JIXIONG_MAP = {
    '天辅星': '吉',
    '天心星': '吉',
    '天禽星': '吉',
    '天任星': '吉',
    '天冲星': '平',
    '天英星': '平',
    '天蓬星': '凶',
    '天芮星': '凶',
    '天柱星': '凶',
    '天辅': '吉',
    '天心': '吉',
    '天禽': '吉',
    '天任': '吉',
    '天冲': '平',
    '天英': '平',
    '天蓬': '凶',
    '天芮': '凶',
    '天柱': '凶',
}

# --- 奇仪组合吉凶映射表（部分常见组合）---
QIYI_JIXIONG_MAP = {
    '丙+戊': '吉',
    '戊+丙': '吉',
    '乙+丙': '吉',
    '丙+乙': '吉',
    '丁+丙': '吉',
    '丙+丁': '吉',
    '乙+丁': '平',
    '丁+乙': '平',
    '戊+己': '凶',
    '己+戊': '凶',
    '庚+庚': '凶',
    '辛+辛': '凶',
    '壬+壬': '凶',
    '癸+癸': '凶',
    '壬+癸': '凶',
    '癸+壬': '凶',
}


def get_jixiong_from_mapping(element_type: str, element_name: str) -> str:
    """从映射表中查询吉凶信息"""
    if not element_name or element_name == '无':
        return '无'
    if element_type == 'bamen':
        return BAMEN_JIXIONG_MAP.get(element_name, '无')
    if element_type == 'bashen':
        return BASHEN_JIXIONG_MAP.get(element_name, '无')
    if element_type == 'jiuxing':
        return JIUXING_JIXIONG_MAP.get(element_name, '无')
    if element_type == 'qiyi':
        return QIYI_JIXIONG_MAP.get(element_name, '无')
    return '无'


def extract_primary_positive_feature(chart_data_row: dict) -> tuple[str | None, str | None]:
    """
    按优先级挑选一个“吉”象，供 LLM Prompt 与兜底文案使用
    优先级：格局 > 奇仪组合 > 八门 > 八神
    """
    geju_names = chart_data_row.get('geju_names')
    if geju_names and pd.notna(geju_names) and str(geju_names).strip() and str(geju_names) != '无':
        return "格局", f"此时辰为【{geju_names}】，整体局势顺势而为，利于推进{chart_data_row.get('具体事项', '当事事项')}。"

    tianpan_gan = chart_data_row.get('tianpanGan', '无')
    dipan_gan = chart_data_row.get('dipan', '无')
    if tianpan_gan != '无' and dipan_gan != '无':
        combo = f"{tianpan_gan}+{dipan_gan}"
        if get_jixiong_from_mapping('qiyi', combo) == '吉':
            # 从知识库查询天盘天干和地盘天干的组合信息
            qiyi_meaning = get_lookup_info('qiyi', {'天盘天干': tianpan_gan, '地盘天干': dipan_gan})
            if qiyi_meaning and qiyi_meaning != '无' and qiyi_meaning != "查询条件不足" and qiyi_meaning != "未找到描述信息" and not qiyi_meaning.startswith("知识库"):
                return "奇仪组合", f"奇仪组合【{combo}】，象意：{qiyi_meaning}，助力{chart_data_row.get('具体事项', '当事事项')}顺利推进。"
            else:
                # 如果知识库没有信息，回退到使用数据库中的 qiyi_zuhe_name
                qiyi_name = chart_data_row.get('qiyi_zuhe_name')
                if qiyi_name and pd.notna(qiyi_name) and str(qiyi_name).strip() and str(qiyi_name) != '无':
                    sanitized = sanitize_qiyi_name(qiyi_name)
                    return "奇仪组合", f"奇仪组合【{sanitized}】，暗示合作顺畅、资源互补，助力{chart_data_row.get('具体事项', '当事事项')}顺利推进。"

    renpan = chart_data_row.get('renpan', '无')
    if renpan and renpan != '无' and get_jixiong_from_mapping('bamen', renpan) == '吉':
        renpan_meaning = get_lookup_info('bamen', {'八门': renpan})
        if renpan_meaning and renpan_meaning != '无':
            return "人盘八门", f"人盘八门【{renpan}门】，象意：{renpan_meaning}，利于直接行动并获得良好反馈。"

    shenpan = chart_data_row.get('shenpan', '无')
    if shenpan and shenpan != '无' and get_jixiong_from_mapping('bashen', shenpan) == '吉':
        shenpan_meaning = get_lookup_info('bashen', {'八神': shenpan})
        if shenpan_meaning and shenpan_meaning != '无':
            return "神盘八神", f"神盘八神【{shenpan}】，象意：{shenpan_meaning}，说明此时贵人助力、能量正面。"

    return None, None


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


def normalize_qimen_terms(text: str) -> str:
    """
    后处理函数：确保所有奇门遁甲专有名词都用【】括起来
    """
    # 八门名称列表
    bamen_list = ['景门', '开门', '休门', '生门', '伤门', '杜门', '死门', '惊门']
    # 八神名称列表
    bashen_list = ['值符', '腾蛇', '太阴', '六合', '白虎', '玄武', '九地', '九天']
    # 天干列表（用于奇仪组合）
    tiangan_list = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
    # 九星名称列表
    jiuxing_list = ['天蓬星', '天芮星', '天冲星', '天辅星', '天禽星', '天心星', '天柱星', '天任星', '天英星']
    
    # 处理八门：如果不在【】中，则添加
    for bamen in bamen_list:
        # 匹配不在【】中的八门名称
        pattern = r'(?<!【)' + re.escape(bamen) + r'(?!】)'
        text = re.sub(pattern, f'【{bamen}】', text)
    
    # 处理八神：如果不在【】中，则添加
    for bashen in bashen_list:
        pattern = r'(?<!【)' + re.escape(bashen) + r'(?!】)'
        text = re.sub(pattern, f'【{bashen}】', text)
    
    # 处理九星：如果不在【】中，则添加
    for jiuxing in jiuxing_list:
        pattern = r'(?<!【)' + re.escape(jiuxing) + r'(?!】)'
        text = re.sub(pattern, f'【{jiuxing}】', text)
    
    # 处理奇仪组合：格式为 天干+天干，如 癸+乙、戊+丙 等
    for t1 in tiangan_list:
        for t2 in tiangan_list:
            combo = f'{t1}+{t2}'
            # 匹配不在【】中的奇仪组合
            pattern = r'(?<!【)' + re.escape(combo) + r'(?!】)'
            text = re.sub(pattern, f'【{combo}】', text)
    
    return text


# --- 敏感词汇映射表 ---
# 用于替换奇门遁甲术语中可能引起误解的词汇
SENSITIVE_TERMS_MAP = {
    '幼女奸淫': '壬癸相合',
    '奸淫': '相合',
    # 可以根据需要添加其他需要替换的术语
}


def sanitize_qiyi_name(qiyi_name: str) -> str:
    """
    清理奇仪组合名称中的敏感词汇
    将不当术语替换为更合适的描述
    """
    if not qiyi_name or qiyi_name == '无' or pd.isna(qiyi_name):
        return qiyi_name
    
    qiyi_name_str = str(qiyi_name).strip()
    
    # 替换敏感词汇
    for sensitive, replacement in SENSITIVE_TERMS_MAP.items():
        if sensitive in qiyi_name_str:
            qiyi_name_str = qiyi_name_str.replace(sensitive, replacement)
    
    return qiyi_name_str


async def generate_tag_description(
    chart_data_row: dict, specific_matter: str, keynote_result: str = "有宜进行"
) -> tuple[str, int, int, bool]:
    """
    为给定的事项生成简洁、通俗易懂的描述，并返回LLM token使用量。
    该接口只输出“宜进行”结果，因此这里只生成积极语气的描述。
    """
    primary_title, primary_desc = extract_primary_positive_feature(chart_data_row)
    print(f"test根据数据{chart_data_row}通过优先级判断吉凶得出【{primary_title}】有利进行，具体：{primary_desc}其中这里的话语是硬编码，是为了传给大模型让其知道是吉")
    if not primary_desc:
        primary_desc = "此刻整体局势平稳，可顺势推进当前事项。"

    prompt_content = f"""
你是一位专业的奇门遁甲分析师，需要为"{specific_matter}"生成一句积极的行动建议。
核心基调：只有"有宜进行"，请保持鼓励、肯定的语气。
重点信息：{primary_desc}

请用直白语言说明此刻为何顺利、可以获得哪些积极结果。
要求：
- 不使用比喻，不超过150字。
- 所有奇门专有名词都用【】括起来（如【景门】【值符】等）。
- **重要：只使用重点信息中明确提到的奇门要素，不要添加或编造其他要素。**
- 不要使用"动手"这类词，统一用"行动"
"""

    try:
        # 使用信号量控制并发数
        async with LLM_SEMAPHORE:
            response = await client.chat.completions.create(
                model=VLLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt_content}],
                temperature=0.7,
                max_tokens=250
            )
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        generated_text = response.choices[0].message.content.strip()
        
        # 后处理：统一专有名词格式，确保都用【】括起来
        normalized_text = normalize_qimen_terms(generated_text)
        
        return normalized_text, prompt_tokens, completion_tokens, False
    except Exception as e:
        print(f"LLM生成'{specific_matter}'描述失败: {e}")
        return f"未能生成具体描述。请稍后重试。", 0, 0, True


# --- 辅助函数：计算日干 ---
TIANGAN_LIST = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
# 1900年1月1日是辛丑日，辛是TIANGAN_LIST中索引为7的天干 (0-based)
REFERENCE_DATE = datetime(1900, 1, 1)
REFERENCE_STEM_INDEX = 7  # '辛' 的索引


def get_day_stem_from_gregorian_date(target_datetime: datetime) -> str:
    """
    根据公历日期计算对应的日干（天干）。
    基于1900年1月1日是辛日的参考点。
    """
    # 只需要日期部分来计算天数差，时间部分不影响日干
    days_diff = (target_datetime.date() - REFERENCE_DATE.date()).days
    day_stem_index = (REFERENCE_STEM_INDEX + days_diff) % 10
    return TIANGAN_LIST[day_stem_index]


# --- 辅助函数：将时间转换为地支时辰 ---
def time_to_chinese_hour_segment(time_obj: time) -> tuple[str, str]:
    """
    将 datetime.time 对象转换为地支时辰和对应的起始时间字符串。
    限定在辰时到亥时。
    """
    hour = time_obj.hour

    if 7 <= hour < 9:
        return "辰时", "7:00"
    elif 9 <= hour < 11:
        return "巳时", "9:00"
    elif 11 <= hour < 13:
        return "午时", "11:00"
    elif 13 <= hour < 15:
        return "未时", "13:00"
    elif 15 <= hour < 17:
        return "申时", "15:00"
    elif 17 <= hour < 19:
        return "酉时", "17:00"
    elif 19 <= hour < 21:
        return "戌时", "19:00"
    elif 21 <= hour < 23:
        return "亥时", "21:00"
    elif hour == 23:
        return "亥时", "21:00"
    else:
        return None, None


# --- 数据库操作函数 (同步，将在线程池中运行) ---
def _get_chart_data_from_db(target_date_dt: datetime, tian_gan_for_person: str, db_config: dict) -> list[dict]:
    """
    同步函数，从数据库获取奇门局数据。
    此函数将被 run_in_threadpool 调用，不会阻塞ASGI事件循环。
    """
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
            cursor.execute(query,
                           (start_datetime.strftime("%Y-%m-%d %H:%M:%S"), end_datetime.strftime("%Y-%m-%d %H:%M:%S")))
            all_chart_data = cursor.fetchall()
        print(f"从数据库获取到 {len(all_chart_data)} 条记录。")
    except Exception as db_err:
        traceback.print_exc()
        raise RuntimeError(f"数据库查询失败: {str(db_err)}") from db_err
    finally:
        if conn:
            conn.close()
    return all_chart_data


def _get_good_days_from_db(
    start_date: datetime, 
    end_date: datetime, 
    tag: str, 
    tian_gan_for_person: str, 
    db_config: dict
) -> list[dict]:
    """
    从数据库获取符合条件的好日子数据（用于奇门择日功能）
    只查询早上7点到晚上11点之间的时间
    此函数将被 run_in_threadpool 调用，不会阻塞ASGI事件循环。
    """
    conn = None
    good_days_data = []
    try:
        conn = pymysql.connect(**db_config)
        with conn.cursor() as cursor:
            gong_status_col = f'gong_relation_status_{tian_gan_for_person}'
            
            query_cols = [
                'id', 'date_str', '吉凶', 'total_score', '具体事项',
                'tianpanGan', 'dipan', 'renpan', 'shenpan', 'tianpanXing_ori',
                'geju_names', 'qiyi_zuhe_name',  # 用于生成reason的字段
                gong_status_col
            ]
            query_cols_str = ", ".join(f"`{col}`" for col in query_cols)
            
            # 查询时间范围：start_date 00:00:00 到 end_date 23:59:59（再在 SQL 中按小时筛选）
            start_datetime = start_date.replace(hour=0, minute=0, second=0)
            end_datetime = end_date.replace(hour=23, minute=59, second=59)
            
            query = f"""
                SELECT {query_cols_str}
                FROM qimen_interpreted_analysis
                WHERE `具体事项` = %s
                AND `date_str` BETWEEN %s AND %s
                AND HOUR(`date_str`) >= 7
                AND HOUR(`date_str`) < 23
                ORDER BY `total_score` DESC, `date_str` ASC
            """
            cursor.execute(query, (
                tag,
                start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                end_datetime.strftime("%Y-%m-%d %H:%M:%S")
            ))
            all_data = cursor.fetchall()
            
            # 筛选出适宜的时间（吉凶=吉 且 gong_status in ['吉', '平']）
            for row in all_data:
                person_gong_status = row.get(gong_status_col)
                keynote_res = calculate_keynote_result(row['吉凶'], person_gong_status)
                if keynote_res == "有宜进行":
                    good_days_data.append(row)
                    
        print(f"从数据库获取到 {len(good_days_data)} 条适宜记录（tag: {tag}）。")
    except Exception as db_err:
        traceback.print_exc()
        raise RuntimeError(f"数据库查询失败: {str(db_err)}") from db_err
    finally:
        if conn:
            conn.close()
    return good_days_data


def generate_reason(chart_data_row: dict) -> str:
    """
    使用与 LLM 相同的优先级逻辑，生成兜底描述
    """
    _, desc = extract_primary_positive_feature(chart_data_row)
    if desc:
        return desc
    return "此时辰运势良好，适宜进行。"


# --- 奇门择日接口 ---
@router.post("/chooseGoodDay", response_model=ChooseGoodDayResponse, summary="奇门择日接口")
async def qimen_choose_good_day(request_data: ChooseGoodDayRequest):
    """
    根据用户提供的生日、标签和时间范围，查询适宜的日期时间。
    返回最多20个适宜的时间点，时间范围为早上7点到晚上11点（07:00-23:00）。
    """
    try:
        birthday_dt = request_data.birthday
        try:
            start_date = datetime.strptime(request_data.startDate, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"开始日期格式错误，期望格式: YYYY-MM-DD，收到: {request_data.startDate}"
            )
        try:
            end_date = datetime.strptime(request_data.endDate, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"截止日期格式错误，期望格式: YYYY-MM-DD，收到: {request_data.endDate}"
            )
        tag = request_data.tag

        # 计算日干
        tian_gan_for_person = get_day_stem_from_gregorian_date(birthday_dt)
        print(f"根据生日 {birthday_dt} 计算得到的日干是: {tian_gan_for_person}")

        # 从数据库查询适宜的时间
        try:
            good_days_data = await run_in_threadpool(
                _get_good_days_from_db,
                start_date,
                end_date,
                tag,
                tian_gan_for_person,
                DB_CONFIG
            )
        except RuntimeError as db_err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"数据库查询失败: {str(db_err)}"
            )

        # 对结果进行去重：同一个date_str只保留total_score最高的一条
        seen_datetimes = {}  # key: 格式化后的时间字符串, value: 对应的记录
        for row in good_days_data:
            date_time_str = row['date_str']
            if not isinstance(date_time_str, datetime):
                try:
                    date_time_str = datetime.strptime(str(date_time_str), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    print(f"无法解析date_str: {row['date_str']}")
                    continue

            date_time_formatted = date_time_str.strftime("%Y-%m-%d %H:%M:%S")
            
            # 如果这个时间点已经存在，比较total_score，只保留分数更高的
            if date_time_formatted in seen_datetimes:
                existing_score = seen_datetimes[date_time_formatted].get('total_score', 0) or 0
                current_score = row.get('total_score', 0) or 0
                if current_score > existing_score:
                    # 替换为分数更高的记录
                    seen_datetimes[date_time_formatted] = row
            else:
                seen_datetimes[date_time_formatted] = row
        
        # 转换为列表，按total_score降序排序，取前10个
        unique_good_days = list(seen_datetimes.values())
        unique_good_days.sort(key=lambda x: (x.get('total_score') or 0), reverse=True)
        unique_good_days = unique_good_days[:10]
        
        # 生成结果列表（最多10个），并准备调用LLM生成reason
        processed_entries: list[dict] = []
        for row in unique_good_days:
            date_time_str = row['date_str']
            if not isinstance(date_time_str, datetime):
                try:
                    date_time_str = datetime.strptime(str(date_time_str), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    print(f"无法解析date_str: {row['date_str']}")
                    continue

            # 格式化日期时间
            date_time_formatted = date_time_str.strftime("%Y-%m-%d %H:%M:%S")

            # 奇门择日仅返回宜进行的时间段，直接标记为"有宜进行"
            row['keynote_result_for_person'] = "有宜进行"
            specific_matter = row.get('具体事项') or tag

            processed_entries.append({
                "row": row,
                "date_time_formatted": date_time_formatted,
                "specific_matter": specific_matter
            })

        good_day_list = []
        total_input_tokens = 0
        total_output_tokens = 0

        llm_failure_detected = False
        if processed_entries:
            llm_tasks = [
                generate_tag_description(
                    entry["row"],
                    entry["specific_matter"],
                    entry["row"].get('keynote_result_for_person', "有宜进行")
                )
                for entry in processed_entries
            ]
            llm_time_start = time_module.perf_counter()

            llm_results = await asyncio.gather(*llm_tasks)

            llm_time =  time_module.perf_counter() - llm_time_start

            print(f"llm Execution Time: {llm_time:.2f} seconds")

            for entry, (description, input_tokens, output_tokens, had_llm_error) in zip(processed_entries, llm_results):
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                if had_llm_error:
                    llm_failure_detected = True

                # 如果LLM生成失败则回落到模板reason
                if not description or description.strip() == "" or "未能生成具体描述" in description:
                    description = generate_reason(entry["row"])

                good_day_list.append(GoodDay(
                    dateTime=entry["date_time_formatted"],
                    reason=description
                ))

        print(f"奇门择日查询完成，返回 {len(good_day_list)} 个适宜时间点。")
        print(f"LLM Tokens - Input: {total_input_tokens}, Output: {total_output_tokens}")
        response_payload = ChooseGoodDayResponse(goodDayList=good_day_list)
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"服务器内部错误: {str(e)}"
        )


app.include_router(router)

