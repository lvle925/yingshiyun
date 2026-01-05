# db_manager.py

import aiomysql
from collections import defaultdict
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager # 导入异步上下文管理器工具
import time
from datetime import datetime, timedelta
from ziwei_ai_function import VALID_XINGXI_DIZHI_COMBOS
#import pandas as pd
logger = logging.getLogger(__name__)
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# 导入 config.py 中的配置
from config import DB_CONFIG

# 全局连接池变量
db_pool: Optional[aiomysql.Pool] = None


async def init_db_pool():
    """在应用启动时初始化数据库连接池。"""
    global db_pool
    if db_pool is None:
        try:
            db_pool = await aiomysql.create_pool(
                **DB_CONFIG,
                minsize=5,  # 最小连接数
                maxsize=20,  # 最大连接数
                pool_recycle=3600  # 连接回收时间（秒）
            )
            logger.info("数据库连接池初始化成功。")
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

    # --- 步骤 1: 将所有查询按目标表名进行分组 ---
    # queries_by_table 的结构: { '表名1': [查询详情1, 查询详情2], '表名2': [...] }
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

        # 将本宫ID与查询参数一起存储
        queries_by_table[table_name].append({
            'bengong_id': bengong_id,
            'time_params': {
                'niangan': q['niangan'],
                'nianzhi': q['nianzhi'],
                'yuezhi': q['yuezhi'],
                'shichen': q['shichen']
            }
        })

    # --- 步骤 2: 遍历每个目标表，执行批量查询并使用局部映射 ---
    final_results = defaultdict(list)

    if not queries_by_table:
        return {}

    async with db_pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            # 遍历每个需要查询的表
            for table_name, query_list_for_table in queries_by_table.items():
                if not query_list_for_table:
                    continue

                # --- 核心修正：为当前表创建一个局部的、明确的映射 ---
                # local_time_to_id_map 的结构: { '时间key': '对应的本宫ID' }
                local_time_to_id_map = {
                    f"{item['time_params']['niangan']}|{item['time_params']['nianzhi']}|{item['time_params']['yuezhi']}|{item['time_params']['shichen']}":
                        item['bengong_id']
                    for item in query_list_for_table
                }

                # 构建SQL的WHERE子句
                conditions = []
                params = []
                # 使用 item['time_params'] 来构建查询
                for item in query_list_for_table:
                    details = item['time_params']
                    conditions.append("(niangan = %s AND nianzhi = %s AND yuezhi = %s AND shichen = %s)")
                    params.extend([details['niangan'], details['nianzhi'], details['yuezhi'], details['shichen']])

                # 去重，避免因对宫等逻辑导致重复的查询条件
                unique_conditions = list(
                    set(zip(conditions, [tuple(params[i:i + 4]) for i in range(0, len(params), 4)])))

                final_conditions = [c[0] for c in unique_conditions]
                final_params = [p for c in unique_conditions for p in c[1]]

                where_clause = " OR ".join(final_conditions)

                # 假设 active 列存在于分表中
                sql = f"""
                        SELECT niangan, nianzhi, yuezhi, shichen, xianxiang
                        FROM `{table_name}`
                        WHERE ({where_clause}) AND active = 1
                    """

                # 在查询前检查表是否存在
                check_table_sql = f"""
                    SELECT COUNT(*) as count
                    FROM information_schema.tables
                    WHERE table_schema = DATABASE()
                    AND table_name = %s
                """
                
                try:
                    await cursor.execute(check_table_sql, (table_name,))
                    table_exists = await cursor.fetchone()
                    
                    if not table_exists or table_exists.get('count', 0) == 0:
                        logger.warning(f"表 '{table_name}' 不存在，跳过查询")
                        continue
                except Exception as e:
                    logger.warning(f"检查表 '{table_name}' 是否存在时出错: {e}，继续尝试查询")
                    # 如果检查表存在性失败，继续尝试查询

                async with DbQueryTimer(sql, tuple(final_params)):
                    try:
                        await cursor.execute(sql, tuple(final_params))
                        rows = await cursor.fetchall()

                        # --- 步骤 3: 使用局部映射将结果赋给正确的本宫ID ---
                        for row in rows:
                            # 从返回的行构建时间key
                            time_key = f"{row['niangan']}|{row['nianzhi']}|{row['yuezhi']}|{row['shichen']}"

                            # 从局部映射中查找对应的本宫ID
                            bengong_id = local_time_to_id_map.get(time_key)

                            if bengong_id and row.get('xianxiang'):
                                # 直接将结果添加到正确的bengong_id下
                                final_results[bengong_id].append(row['xianxiang'])

                    except Exception as e:
                        # 如果是表不存在的错误，记录为警告；其他错误记录为错误
                        error_msg = str(e)
                        if "doesn't exist" in error_msg or "不存在" in error_msg:
                            logger.warning(f"表 '{table_name}' 不存在，跳过查询: {error_msg}")
                        else:
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

        # 准备本宫参数
        bengong_params = [q['xingxi'], q['dizhi'], q['gongwei'], q['pipeixingyao'], q['huaxing']]

        if combo_key in VALID_XINGXI_DIZHI_COMBOS:
            bengong_params_list.append(bengong_params)
        else:
            # 如果不在清单里，直接准备其对宫的查询参数
            logger.debug(f"组合 '{combo_key}' 不在合法清单中，将直接查询其对宫。")
            duigong_dizhi = DIZHI_DUIGONG_MAP.get(q['dizhi'])
            if duigong_dizhi:
                duigong_params_list.append([q['xingxi'], duigong_dizhi, q['gongwei'], q['pipeixingyao'], q['huaxing']])

        # --- 步骤 2: 构建并执行一个统一的批量查询 ---
    all_conditions = []
    all_params = []

    # 添加本宫的查询条件和参数
    for params in bengong_params_list:
        all_conditions.append("(trim(xingxi) = %s AND trim(dizhi) = %s AND trim(gongwei) = %s AND trim(pipeixingyao) = %s AND trim(huaxing) = %s)")
        all_params.extend(params)

    # 添加对宫的查询条件和参数
    for params in duigong_params_list:
        all_conditions.append("(trim(xingxi) = %s AND trim(dizhi) = %s AND trim(gongwei) = %s AND trim(pipeixingyao) = %s AND trim(huaxing) = %s)")
        all_params.extend(params)

    if not all_conditions:
        return {}

    # --- 执行一次大的批量查询 ---
    where_clause = " OR ".join(all_conditions)
    # 返回的字段需要能构建唯一的ID
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
                        #print("row*********************",row)
                        # 用返回的字段构建基于【实际地支】的ID
                        result_id = f"{row[0]}|{row[1]}|{row[2]}|{row[3]}|{row[4]}"
                        huizongxianxiang = row[5]
                        if result_id not in batch_results: batch_results[result_id] = []
                        batch_results[result_id].append(huizongxianxiang)
        except Exception as e:
            logger.error(f"批量查询运势数据时出错: {e}")

    # --- 步骤 3: 组装最终结果，确保键是基于【本宫地支】 ---
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
    print("连接成功",flush=True)
    # 获取当前月份，格式为 "YYYY-MM"
    current_month = datetime.now().strftime('%Y-%m')

    # 这条 SQL 语句是核心，利用了 MySQL 的 ON DUPLICATE KEY UPDATE 特性
    sql = """
        INSERT INTO api_usage_stats (session_id, app_id, request_month, request_count)
        VALUES (%s, %s, %s, 1)
        ON DUPLICATE KEY UPDATE request_count = request_count + 1;
    """
    params = (session_id, app_id, current_month)

    print("获取成功",flush=True)

    try:
        print(0)
        async with db_pool.acquire() as conn:
            print(1)
            async with conn.cursor() as cursor:
                print(2)
                await cursor.execute(sql, params)
                print(3)
            # 对于 INSERT/UPDATE 操作，我们需要提交事务（如果 autocommit=False）
            # 因为我们在 DB_CONFIG 中设置了 autocommit=True，所以这里可以省略 conn.commit()
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




