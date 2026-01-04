from fastapi import FastAPI, Body, APIRouter
import pymysql
from datetime import datetime
from dotenv import load_dotenv
import os
import logging
from typing import List, Dict, Any
from starlette.concurrency import run_in_threadpool

logging.basicConfig(level=logging.INFO)

load_dotenv()

app = FastAPI()
router = APIRouter()

# 数据库配置（保持不变）
DB_CONFIG = {
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'host': os.getenv("DB_HOST"),
    'port': int(os.getenv("DB_PORT")),
    'database': os.getenv("DB_NAME"),
    'cursorclass': pymysql.cursors.DictCursor,
    'charset': 'utf8mb4'
}


def time_to_chinese_hour_segment(hour: int) -> tuple[str, str]:
    """
    将小时转换为地支时辰和对应的起始时间字符串。
    限定在辰时到亥时（7点到23点）。
    """
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


def _get_auspicious_info_from_db(start_datetime: datetime, end_datetime: datetime, db_config: dict) -> list:
    """同步函数，从数据库获取吉信息数据，将在线程池中运行"""
    conn = None
    try:
        conn = pymysql.connect(**db_config)
        with conn.cursor() as cur:
            query = """
                SELECT 
                    date_str,
                    fangxiang, 
                    吉色, 
                    吉数, 
                    吉物, 
                    total_score 
                FROM qimen_interpreted_analysis
                WHERE date_str BETWEEN %s AND %s
            """
            cur.execute(query, (
                start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                end_datetime.strftime("%Y-%m-%d %H:%M:%S")
            )) 
            rows = cur.fetchall()
        return rows
    except pymysql.Error as e:
        logging.error(f"Database error: {e}")
        raise RuntimeError(f"Database query failed: {e}")
    finally:
        if conn:
            conn.close()


@router.post("/qiMenAuspiciousInfo")
async def qiMenAuspiciousInfo(date: str = Body(..., embed=True)) -> Dict[str, List[Dict[str, Any]]]:
    """
    根据输入的日期（YYYY-MM-DD格式）查询，并基于 total_score 筛选合并结果。
    """
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        start_datetime = dt.replace(hour=7, minute=0, second=0)
        end_datetime = dt.replace(hour=23, minute=0, second=0)  # 晚上11点
    except ValueError:
        return {"error": "Invalid date format. Expected: YYYY-MM-DD"}

    try:
        rows = await run_in_threadpool(
            _get_auspicious_info_from_db,
            start_datetime,
            end_datetime,
            DB_CONFIG
        )
    except RuntimeError as e:
        return {"error": str(e)}

    if not rows:
        return {"auspiciousInfo": []}

    # 定义8个时辰的时间段
    chinese_hour_segments = [
        ("辰时", "7:00"), ("巳时", "9:00"), ("午时", "11:00"),
        ("未时", "13:00"), ("申时", "15:00"), ("酉时", "17:00"),
        ("戌时", "19:00"), ("亥时", "21:00")
    ]
    
    # 按时辰分组数据
    grouped_by_hour = {}
    for row in rows:
        date_str = row.get('date_str')
        if not date_str:
            continue
            
        # 解析时间
        if isinstance(date_str, datetime):
            hour = date_str.hour
        else:
            try:
                dt = datetime.strptime(str(date_str), "%Y-%m-%d %H:%M:%S")
                hour = dt.hour
            except ValueError:
                continue
        
        # 获取时辰名称
        hour_name, _ = time_to_chinese_hour_segment(hour)
        if not hour_name:
            continue
        
        # 按时辰分组
        if hour_name not in grouped_by_hour:
            grouped_by_hour[hour_name] = []
        grouped_by_hour[hour_name].append(row)
    
    # 对每个时辰取最高分的一条，并处理数据
    result_list = []
    for hour_name, display_time in chinese_hour_segments:
        hour_rows = grouped_by_hour.get(hour_name, [])
        
        if not hour_rows:
            # 如果该时辰没有数据，返回空结果
            result_list.append({
                "startTimeInChinese": hour_name,
                "auspiciousPosition": "",
                "auspiciousItem": "",
                "auspiciousNumber": "",
                "auspiciousColor": ""
            })
            continue
        
        # 找到该时辰最高分的记录
        try:
            max_score = max(float(r.get("total_score", -9999)) for r in hour_rows if r.get("total_score") is not None)
            best_rows = [r for r in hour_rows if float(r.get("total_score", -9999)) == max_score]
        except (KeyError, ValueError):
            # 如果没有有效的total_score，取第一条
            best_rows = [hour_rows[0]]
        
        # 合并最高分记录的数据
        auspicious_positions = set()
        auspicious_colors = set()
        auspicious_numbers = set()
        auspicious_items = set()
        
        for r in best_rows:
            for val, target_set in [
                (r.get("fangxiang"), auspicious_positions),
                (r.get("吉色"), auspicious_colors),
                (r.get("吉数"), auspicious_numbers),
                (r.get("吉物"), auspicious_items)
            ]:
                if val is not None:
                    for item in str(val).split(','):
                        stripped_item = item.strip()
                        if stripped_item:
                            target_set.add(stripped_item)
        
        result_list.append({
            "startTimeInChinese": hour_name,
            "auspiciousPosition": ", ".join(sorted(list(auspicious_positions))),
            "auspiciousItem": ", ".join(sorted(list(auspicious_items))),
            "auspiciousNumber": ", ".join(sorted(list(auspicious_numbers))),
            "auspiciousColor": ", ".join(sorted(list(auspicious_colors)))
        })
    
    return {"auspiciousInfo": result_list}


app.include_router(router)