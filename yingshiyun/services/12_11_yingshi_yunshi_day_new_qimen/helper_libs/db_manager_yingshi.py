import os
import aiomysql
from collections import defaultdict
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager 
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
from app.config import DB_CONFIG 

load_dotenv()

logger = logging.getLogger(__name__)


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


db_pool: Optional[aiomysql.Pool] = None


async def init_db_pool():
    """在应用启动时初始化数据库连接池。"""
    global db_pool
    if db_pool is None:
        try:
            db_pool = await aiomysql.create_pool(
                **DB_CONFIG,
                minsize=5, 
                maxsize=20, 
                pool_recycle=3600 
            )
            logger.info("数据库连接池初始化成功。")
            return db_pool
        except Exception as e:
            logger.error(f"数据库连接池初始化失败: {e}")
            db_pool = None


async def close_db_pool():
    """在应用关闭时关闭数据库连接池。"""
    if db_pool:
        db_pool.close()
        await db_pool.wait_closed()
        logger.info("数据库连接池已关闭。")


DIZHI_DUIGONG_MAP = {"子": "午", "丑": "未", "寅": "申", "卯": "酉", "辰": "戌", "巳": "亥", "午": "子", "未": "丑", "申": "寅", "酉": "卯", "戌": "辰", "亥": "巳"}


async def query_natal_chart_info_batch(queries: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """
    【分表批量查询】本命盘数据，内置对宫查找和动态表名逻辑。
    :param queries: 一个查询字典的列表，每个字典包含 'xingxi', 'dizhi', 'gongwei', 'niangan' 等键。
    :return: 一个字典，键是唯一的本宫查询标识符，值是匹配的 'xianxiang' 文本列表。
    """
    if not db_pool or not queries:
        return {}

    queries_by_table = defaultdict(list)

    for q in queries:
        bengong_id = f"{q['xingxi']}|{q['dizhi']}|{q['gongwei']}"

        combo_key = f"{q['xingxi']},{q['dizhi']}"
        target_dizhi = q['dizhi']
        if combo_key not in VALID_XINGXI_DIZHI_COMBOS:
            target_dizhi = DIZHI_DUIGONG_MAP.get(q['dizhi'], None)

        if not target_dizhi:
            logger.warning(f"无法确定目标地支或对宫地支，跳过查询: {q}")
            continue

        xingxi_for_tablename = q['xingxi'].replace('，', '_')
        table_name = f"{xingxi_for_tablename}_{target_dizhi}_{q['gongwei']}_all_time_combinations"

        queries_by_table[table_name].append({
            'bengong_id': bengong_id,
            'time_params': {
                'niangan': q['niangan'],
                'nianzhi': q['nianzhi'],
                'yuezhi': q['yuezhi'],
                'shichen': q['shichen']
            }
        })

    final_results = defaultdict(list)

    if not queries_by_table:
        return {}

    async with db_pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            for table_name, query_list_for_table in queries_by_table.items():
                if not query_list_for_table:
                    continue

                local_time_to_id_map = {
                    f"{item['time_params']['niangan']}|{item['time_params']['nianzhi']}|{item['time_params']['yuezhi']}|{item['time_params']['shichen']}":
                        item['bengong_id']
                    for item in query_list_for_table
                }
                conditions = []
                params = []

                for item in query_list_for_table:
                    details = item['time_params']
                    conditions.append("(niangan = %s AND nianzhi = %s AND yuezhi = %s AND shichen = %s)")
                    params.extend([details['niangan'], details['nianzhi'], details['yuezhi'], details['shichen']])

                unique_conditions = list(
                    set(zip(conditions, [tuple(params[i:i + 4]) for i in range(0, len(params), 4)])))

                final_conditions = [c[0] for c in unique_conditions]
                final_params = [p for c in unique_conditions for p in c[1]]

                where_clause = " OR ".join(final_conditions)

                sql = f"""
                        SELECT niangan, nianzhi, yuezhi, shichen, xianxiang
                        FROM `{table_name}`
                        WHERE ({where_clause}) AND active = 1
                    """

                async with DbQueryTimer(sql, tuple(final_params)):
                    try:
                        await cursor.execute(sql, tuple(final_params))
                        rows = await cursor.fetchall()

                        for row in rows:

                            time_key = f"{row['niangan']}|{row['nianzhi']}|{row['yuezhi']}|{row['shichen']}"

                            bengong_id = local_time_to_id_map.get(time_key)

                            if bengong_id and row.get('xianxiang'):
                                final_results[bengong_id].append(row['xianxiang'])

                    except Exception as e:
                        logger.error(f"查询表 '{table_name}' 时出错: {e}. SQL: {sql[:500]}...")
                        continue


    return dict(final_results)


async def query_horoscope_info_batch(queries: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """
    【批量查询】运势数据，内置对宫查找逻辑。
    """
    if not db_pool or not queries:
        return {}

    
    bengong_params_list = []
    duigong_params_list = []

    for q in queries:
        normalized_xingxi = q['xingxi']
        combo_key = f"{normalized_xingxi},{q['dizhi']}"

        bengong_params = [q['xingxi'], q['dizhi'], q['gongwei'], q['pipeixingyao'], q['huaxing']]

        if combo_key in VALID_XINGXI_DIZHI_COMBOS:
            bengong_params_list.append(bengong_params)
        else:

            logger.debug(f"组合 '{combo_key}' 不在合法清单中，将直接查询其对宫。")
            duigong_dizhi = DIZHI_DUIGONG_MAP.get(q['dizhi'])
            if duigong_dizhi:
                duigong_params_list.append([q['xingxi'], duigong_dizhi, q['gongwei'], q['pipeixingyao'], q['huaxing']])

    all_conditions = []
    all_params = []

    for params in bengong_params_list:
        all_conditions.append("(trim(xingxi) = %s AND trim(dizhi) = %s AND trim(gongwei) = %s AND trim(pipeixingyao) = %s AND trim(huaxing) = %s)")
        all_params.extend(params)

    for params in duigong_params_list:
        all_conditions.append("(trim(xingxi) = %s AND trim(dizhi) = %s AND trim(gongwei) = %s AND trim(pipeixingyao) = %s AND trim(huaxing) = %s)")
        all_params.extend(params)

    if not all_conditions:
        return {}


    where_clause = " OR ".join(all_conditions)
    sql = f"""
            SELECT xingxi, dizhi, gongwei, pipeixingyao, huaxing, huizongxianxiang 
            FROM group_data 
            WHERE ({where_clause}) AND active = 1
        """
    batch_results = {}
    async with DbQueryTimer(sql, tuple(all_params)):
        try:
            async with db_pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(sql, all_params)
                    rows = await cursor.fetchall()
                    for row in rows:
                        result_id = f"{row[0]}|{row[1]}|{row[2]}|{row[3]}|{row[4]}"
                        huizongxianxiang = row[5]
                        if result_id not in batch_results: batch_results[result_id] = []
                        batch_results[result_id].append(huizongxianxiang)
        except Exception as e:
            logger.error(f"批量查询运势数据时出错: {e}")

    final_results = {}
    for q in queries:
        # 构建本宫ID
        bengong_id = f"{q['xingxi']}|{q['dizhi']}|{q['gongwei']}|{q['pipeixingyao']}|{q['huaxing']}"

        normalized_xingxi = q['xingxi']
        combo_key = f"{normalized_xingxi},{q['dizhi']}"

        if combo_key in VALID_XINGXI_DIZHI_COMBOS:
            # 如果在清单里，那么结果一定是在 batch_results 中以本宫ID为键
            if bengong_id in batch_results:
                final_results[bengong_id] = batch_results[bengong_id]
        else:
            # 如果不在清单里，我们需要去 batch_results 中找对宫的结果
            duigong_dizhi = DIZHI_DUIGONG_MAP.get(q['dizhi'])
            if duigong_dizhi:
                duigong_id = f"{q['xingxi']}|{duigong_dizhi}|{q['gongwei']}|{q['pipeixingyao']}|{q['huaxing']}"
                if duigong_id in batch_results:
                    # 【关键】: 即使是从对宫找到的，也用本宫的ID作为键存起来
                    final_results[bengong_id] = batch_results[duigong_id]

    #print("final_results",final_results)
    return final_results


async def upsert_api_usage_stats(session_id: str, app_id: str):
    """
    更新或插入API用量统计。
    如果记录已存在，则请求次数+1；如果不存在，则创建新记录。
    """
    if not db_pool:
        logger.warning("数据库连接池未初始化，跳过用量统计。")
        return
    current_month = datetime.now().strftime('%Y-%m')

    sql = """
        INSERT INTO api_usage_stats (session_id, app_id, request_month, request_count)
        VALUES (%s, %s, %s, 1)
        ON DUPLICATE KEY UPDATE request_count = request_count + 1;
    """
    params = (session_id, app_id, current_month)

    try:
        async with db_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, params)
    except Exception as e:
        logger.error(f"更新API用量统计时出错: {e}", exc_info=True)


async def log_qa_record(session_id: str, app_id: str, user_query: str, final_response: str):
    """
    向 qa_logs 表中插入一条新的问答记录。
    """
    if not db_pool:
        logger.warning("数据库连接池未初始化，跳过问答日志记录。")
        return

    sql = """
        INSERT INTO qa_logs (session_id, app_id, user_query, final_response)
        VALUES (%s, %s, %s, %s);
    """
    params = (session_id, app_id, user_query, final_response)

    try:
        async with db_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, params)
    except Exception as e:
        logger.error(f"记录问答日志时出错: {e}", exc_info=True)




class DbQueryTimer:
    def __init__(self, sql: str, params: tuple):
        self.sql = sql
        self.params = params
        self.start_time = 0.0

    async def __aenter__(self):
        """进入 async with 块时执行"""
        self.start_time = time.monotonic()
        # 这里可以返回 self，但我们不需要
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出 async with 块时执行"""
        end_time = time.monotonic()
        duration_ms = (end_time - self.start_time) * 1000

        log_level = logging.INFO
        if exc_type:  # 如果块内发生了异常
            log_level = logging.ERROR
            logger.error(f"查询块内发生异常: {exc_type.__name__}: {exc_val}")
        # 返回 False 表示如果发生了异常，我们不“吞掉”它，让它继续向上传播
        return False
