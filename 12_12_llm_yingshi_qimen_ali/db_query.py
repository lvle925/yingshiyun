# -*- coding: utf-8 -*-
"""
数据库查询模块
包含时间映射、数据库查询等逻辑
"""

import asyncio
import pymysql
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from time import time
from starlette.concurrency import run_in_threadpool
from config import DB_CONFIG
from user_info_extractor import get_day_stem_from_gregorian_date

logger = logging.getLogger(__name__)

_tags_cache: List[str] = []
_tags_cache_time: float = 0.0
_tags_cache_lock = asyncio.Lock()
_TAGS_TTL_SECONDS = 86400  # 24小时

# --- 吉凶映射（参考 api_main_day.py） ---
BAMEN_JIXIONG_MAP = {
    '开门': '吉', '休门': '吉', '生门': '吉', '景门': '平',
    '死门': '凶', '惊门': '凶', '伤门': '凶', '杜门': '凶',
    '开': '吉', '休': '吉', '生': '吉', '景': '平',
    '死': '凶', '惊': '凶', '伤': '凶', '杜': '凶',
}

BASHEN_JIXIONG_MAP = {
    '值符': '吉', '太阴': '吉', '六合': '吉', '九地': '吉', '九天': '吉',
    '白虎': '平', '玄武': '平', '腾蛇': '凶',
}

JIUXING_JIXIONG_MAP = {
    '天辅星': '吉', '天心星': '吉', '天禽星': '吉', '天任星': '吉',
    '天冲星': '平', '天英星': '平',
    '天蓬星': '凶', '天芮星': '凶', '天柱星': '凶',
    '天辅': '吉', '天心': '吉', '天禽': '吉', '天任': '吉',
    '天冲': '平', '天英': '平', '天蓬': '凶', '天芮': '凶', '天柱': '凶',
}

QIYI_JIXIONG_MAP = {
    '丙+戊': '吉', '戊+丙': '吉', '乙+丙': '吉', '丙+乙': '吉',
    '丁+丙': '吉', '丙+丁': '吉',
    '乙+丁': '平', '丁+乙': '平',
    '戊+己': '凶', '己+戊': '凶',
    '庚+庚': '凶', '辛+辛': '凶', '壬+壬': '凶', '癸+癸': '凶',
    '壬+癸': '凶', '癸+壬': '凶',
}


def time_to_chinese_hour_segment(time_obj) -> Tuple[Optional[str], Optional[str]]:
    """
    将时间转换为地支时辰和对应的起始时间字符串
    限定在辰时到亥时（7点到23点）
    """
    if isinstance(time_obj, str):
        try:
            time_obj = datetime.strptime(time_obj, "%H:%M:%S").time()
        except:
            try:
                time_obj = datetime.strptime(time_obj.split()[-1], "%H:%M:%S").time()
            except:
                return None, None
    
    hour = time_obj.hour if hasattr(time_obj, 'hour') else time_obj
    
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


def map_time_to_db_hour(target_datetime: datetime) -> datetime:
    """
    将具体时间点映射到数据库的时辰格式
    数据库存储的时间是通过时辰的方式以两小时间隔的
    例如：7:00-9:00为辰时，存储为8:00
    """
    hour = target_datetime.hour
    
    # 映射到对应时辰的起始时间
    if 1 <= hour < 3:
        mapped_hour = 2
    elif 3 <= hour < 5:
        mapped_hour = 4
    elif 5 <= hour < 7:
        mapped_hour = 6
    elif 7 <= hour < 9:
        mapped_hour = 8
    elif 9 <= hour < 11:
        mapped_hour = 10
    elif 11 <= hour < 13:
        mapped_hour = 12
    elif 13 <= hour < 15:
        mapped_hour = 14
    elif 15 <= hour < 17:
        mapped_hour = 16
    elif 17 <= hour < 19:
        mapped_hour = 18
    elif 19 <= hour < 21:
        mapped_hour = 20
    elif 21 <= hour < 23:
        mapped_hour = 22
    else:
        mapped_hour = 0
    
    return target_datetime.replace(hour=mapped_hour, minute=0, second=0, microsecond=0)


async def get_all_tags_from_db_async(db_pool) -> List[str]:
    """
    查询数据库所有标签的类型（异步版本，使用连接池，带10分钟缓存）
    数据库表：qimen_interpreted_analysis，字段：具体事项
    """
    if not db_pool:
        logger.error("数据库连接池不可用")
        return []
    
    async with _tags_cache_lock:
        now = time()
        if _tags_cache and now - _tags_cache_time < _TAGS_TTL_SECONDS:
            return _tags_cache
        
        try:
            async with db_pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    query = """
                        SELECT DISTINCT `具体事项`
                        FROM qimen_interpreted_analysis
                        WHERE `具体事项` IS NOT NULL
                        AND `具体事项` != ''
                        ORDER BY `具体事项`
                    """
                    await cursor.execute(query)
                    results = await cursor.fetchall()
                    tags = [row[0] for row in results if row[0]]
                    logger.info(f"从数据库获取到 {len(tags)} 个标签")
                    _tags_cache.clear()
                    _tags_cache.extend(tags)
                    globals()['_tags_cache_time'] = now
                    return tags
        except Exception as e:
            logger.error(f"查询标签失败: {e}", exc_info=True)
            return _tags_cache or []


def _derive_jixiong_from_elements(row: Dict[str, Any]) -> Optional[str]:
    """
    根据奇仪/八门/八神/九星的映射推导吉凶（优先级：奇仪组合 > 八门(人盘) > 八神(神盘) > 九星(天盘星)）
    """
    tianpan_gan = row.get("tianpanGan")
    dipan_gan = row.get("dipan")
    if tianpan_gan and dipan_gan:
        combo = f"{tianpan_gan}+{dipan_gan}"
        jx = QIYI_JIXIONG_MAP.get(combo)
        if jx:
            return jx

    renpan = row.get("renpan")
    if renpan:
        jx = BAMEN_JIXIONG_MAP.get(renpan)
        if jx:
            return jx

    shenpan = row.get("shenpan")
    if shenpan:
        jx = BASHEN_JIXIONG_MAP.get(shenpan)
        if jx:
            return jx

    tianpan_xing = row.get("tianpanXing_ori")
    if tianpan_xing:
        jx = JIUXING_JIXIONG_MAP.get(tianpan_xing)
        if jx:
            return jx

    return None


def _evaluate_keynote_result(row: Dict[str, Any], gong_status_col: str) -> str:
    """
    按旧版规则使用行内“吉凶” + 宫位关系列进行判定，返回 "吉" / "凶" / "未知"
    规则（收紧后，仅两种情况算“吉”）：
    - 行内吉凶为“吉” 且 宫位关系为“吉”或“平” -> “吉”
    - 行内吉凶为“平” 且 宫位关系为“吉” -> “吉”
    其他一律视为“未知”（不再把“行内吉凶为吉”单独判吉，也不再把“凶”直接判凶）
    """
    jixiong = row.get("吉凶")
    gong_status = row.get(gong_status_col)
    if jixiong == "吉" and gong_status in ["吉", "平"]:
        return "吉"
    if jixiong == "平" and gong_status == "吉":
        return "吉"
    return "未知"


def _filter_and_rank_records(
    records: List[Dict[str, Any]],
    gong_status_col: str,
    jixiong_preference: Optional[str],
    apply_jixiong_filter: bool,
    max_count: int = 10
) -> List[Dict[str, Any]]:
    """
    按吉凶偏好筛选并排序，最终限制返回条数
    jixiong_preference:
        - "吉"  : 只保留 _evaluate_keynote_result 判为“吉”的行
        - "凶"  : 只保留 _evaluate_keynote_result 判为“非吉”的行（即过滤掉“吉”）
        - "吉凶"/None : 不做吉凶筛选
    apply_jixiong_filter: 仅当 True 且偏好为吉/凶时执行筛选；否则仅做基础排序+去重
    排序（有吉/凶筛选时）：按 total_score 降序，再按 date_str 升序；再按时间去重（保留同 date_str 最高分）
    排序（无吉凶筛选时）：date_str 升序，total_score 降序；再去重截断
    """
    if apply_jixiong_filter and jixiong_preference in ["吉", "凶"]:
        filtered = []
        for row in records:
            keynote = _evaluate_keynote_result(row, gong_status_col)
            row["__keynote_result__"] = keynote
            if jixiong_preference == "吉":
                if keynote == "吉":
                    filtered.append(row)
            else:  # 偏好“凶”：过滤掉“吉”，保留非吉（未知/非吉）
                if keynote != "吉":
                    filtered.append(row)
        # 有筛选时，按照分数优先，再按时间排序
        filtered.sort(
            key=lambda r: (
                -(r.get("total_score") or 0),
                r.get("date_str")
            )
        )
    else:
        # 不做吉凶筛选：仅按时间升序、分数降序
        filtered = sorted(
            records,
            key=lambda r: (
                r.get("date_str"),
                -(r.get("total_score") or 0)
            )
        )

    # 去重：同一 date_str 只保留排序后第一条（最高分/优先级）
    dedup = []
    seen = set()
    for r in filtered:
        ds = r.get("date_str")
        if ds in seen:
            continue
        seen.add(ds)
        dedup.append(r)
        if len(dedup) >= max_count:
            break
    return dedup


def _query_type1_data(
    specific_event: str,
    start_time: datetime,
    end_time: datetime,
    tian_gan_for_person: str,
    jixiong_preference: Optional[str],
    apply_jixiong_filter: bool,
    db_config: dict
) -> List[Dict[str, Any]]:
    """
    查询类型1的数据：具体时间点做具体事件是否合适
    先取较多数据，再按吉/凶偏好及得分筛选出最多10条
    """
    conn = None
    try:
        conn = pymysql.connect(**db_config)
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            gong_status_col = f'gong_relation_status_{tian_gan_for_person}'
            
            query_cols = [
                'id', 'date_str', '吉凶', 'total_score', '具体事项','fangxiang', 
                'tianpanGan', 'dipan', 'renpan', 'shenpan', 'tianpanXing_ori',
                'geju_names', 'qiyi_zuhe_name', gong_status_col
            ]
            query_cols_str = ", ".join(f"`{col}`" for col in query_cols)
            
            query = f"""
                SELECT {query_cols_str}
                FROM qimen_interpreted_analysis
                WHERE `具体事项` = %s
                AND `date_str` BETWEEN %s AND %s
                ORDER BY `date_str` ASC, `total_score` DESC
                LIMIT 200
            """
            cursor.execute(query, (
                specific_event,
                start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end_time.strftime("%Y-%m-%d %H:%M:%S")
            ))
            results = cursor.fetchall()
            filtered = _filter_and_rank_records(
                results,
                gong_status_col,
                jixiong_preference,
                apply_jixiong_filter
            )
            logger.info(f"类型1查询返回 {len(results)} 条，筛选后 {len(filtered)} 条")
            return filtered
    except Exception as e:
        logger.error(f"类型1数据库查询失败: {e}", exc_info=True)
        return []
    finally:
        if conn:
            conn.close()


def _query_type2_data(
    specific_event: str,
    time_range_start: datetime,
    time_range_end: datetime,
    tian_gan_for_person: str,
    jixiong_preference: Optional[str],
    apply_jixiong_filter: bool,
    db_config: dict
) -> List[Dict[str, Any]]:
    """
    查询类型2的数据：什么时间做具体事件
    先取较多数据，再按吉/凶偏好及得分筛选出最多10条
    """
    conn = None
    try:
        conn = pymysql.connect(**db_config)
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            gong_status_col = f'gong_relation_status_{tian_gan_for_person}'
            
            query_cols = [
                'id', 'date_str', '吉凶', 'total_score', '具体事项',
                'tianpanGan', 'dipan', 'renpan', 'shenpan', 'tianpanXing_ori',
                'geju_names', 'qiyi_zuhe_name', gong_status_col
            ]
            query_cols_str = ", ".join(f"`{col}`" for col in query_cols)
            
            query = f"""
                SELECT {query_cols_str}
                FROM qimen_interpreted_analysis
                WHERE `具体事项` = %s
                AND HOUR(`date_str`) >= 7
                AND HOUR(`date_str`) < 23
                AND `date_str` BETWEEN %s AND %s
                ORDER BY `date_str` ASC, `total_score` DESC
                LIMIT 200
            """
            cursor.execute(query, (
                specific_event,
                time_range_start.strftime("%Y-%m-%d %H:%M:%S"),
                time_range_end.strftime("%Y-%m-%d %H:%M:%S")
            ))
            results = cursor.fetchall()
            filtered = _filter_and_rank_records(
                results,
                gong_status_col,
                jixiong_preference,
                apply_jixiong_filter
            )
            logger.info(f"类型2查询返回 {len(results)} 条，筛选后 {len(filtered)} 条")
            return filtered
    except Exception as e:
        logger.error(f"类型2数据库查询失败: {e}", exc_info=True)
        return []
    finally:
        if conn:
            conn.close()


def _query_type3_data(
    start_time: datetime,
    end_time: datetime,
    tian_gan_for_person: str,
    jixiong_preference: Optional[str],
    apply_jixiong_filter: bool,
    db_config: dict
) -> List[Dict[str, Any]]:
    """
    查询类型3的数据：具体时间点做什么事件
    先取较多数据，再按吉/凶偏好及得分筛选出最多10条
    """
    conn = None
    try:
        conn = pymysql.connect(**db_config)
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            gong_status_col = f'gong_relation_status_{tian_gan_for_person}'
            
            query_cols = [
                'id', 'date_str', '吉凶', 'total_score', '具体事项',
                'tianpanGan', 'dipan', 'renpan', 'shenpan', 'tianpanXing_ori',
                gong_status_col
            ]
            query_cols_str = ", ".join(f"`{col}`" for col in query_cols)
            
            query = f"""
                SELECT {query_cols_str}
                FROM qimen_interpreted_analysis
                WHERE `date_str` BETWEEN %s AND %s
                ORDER BY `date_str` ASC, `total_score` DESC
                LIMIT 200
            """
            cursor.execute(query, (
                start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end_time.strftime("%Y-%m-%d %H:%M:%S")
            ))
            results = cursor.fetchall()
            filtered = _filter_and_rank_records(
                results,
                gong_status_col,
                jixiong_preference,
                apply_jixiong_filter
            )
            logger.info(f"类型3查询返回 {len(results)} 条，筛选后 {len(filtered)} 条")
            return filtered
    except Exception as e:
        logger.error(f"类型3数据库查询失败: {e}", exc_info=True)
        return []
    finally:
        if conn:
            conn.close()


async def query_qimen_data(
    qimen_type: str,
    specific_event: Optional[str],
    time_range_start: Optional[str],
    time_range_end: Optional[str],
    jixiong_preference: Optional[str],
    user_info: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    根据奇门问题类型查询数据库
    
    Args:
        qimen_type: 奇门问题类型 ("type1", "type2", "type3")
        specific_event: 具体事件标签
        time_range_start: 时间范围开始时间，格式：YYYY-MM-DD HH:MM:SS
        time_range_end: 时间范围结束时间，格式：YYYY-MM-DD HH:MM:SS（对于具体时辰，与time_range_start相同）
        jixiong_preference: 吉/凶/吉凶，用于吉凶筛选，默认吉凶（不过滤）
        user_info: 用户信息字典
    
    Returns:
        查询结果列表
    """
    # 计算用户的日干
    birthday_dt = datetime.strptime(user_info['birth_datetime'], "%Y-%m-%d %H:%M:%S")
    tian_gan_for_person = get_day_stem_from_gregorian_date(birthday_dt)
    preference = jixiong_preference if jixiong_preference in ["吉", "凶"] else "吉凶"
    print("用户天干是",tian_gan_for_person)
    
    if qimen_type == "type1":
        # 具体时间点做具体事件是否合适
        if not time_range_start or not time_range_end or not specific_event:
            logger.error("类型1缺少必要参数")
            return []
        
        start_dt = datetime.strptime(time_range_start, "%Y-%m-%d %H:%M:%S")
        start_dt = map_time_to_db_hour(start_dt)
        
        end_dt = datetime.strptime(time_range_end, "%Y-%m-%d %H:%M:%S")
        end_dt = map_time_to_db_hour(end_dt)
        print("end_dt",end_dt)
        print("start_dt",start_dt)

        # 仅当用户明确要吉/凶，且是时间范围（而非单点）时执行吉凶筛选
        apply_jixiong_filter = preference in ["吉", "凶"] and start_dt != end_dt
        
        return await run_in_threadpool(
            _query_type1_data,
            specific_event,
            start_dt,
            end_dt,
            tian_gan_for_person,
            preference,
            apply_jixiong_filter,
            DB_CONFIG
        )
    
    elif qimen_type == "type2":
        # 什么时间做具体事件
        if not specific_event:
            logger.error("类型2缺少必要参数")
            return []

        if time_range_start and time_range_end:
            start_dt = datetime.strptime(time_range_start, "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(time_range_end, "%Y-%m-%d %H:%M:%S")
        else:
            # 如果未提及时间范围，设置为从此刻到一年后
            now = datetime.now()
            start_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_dt = (now + timedelta(days=365)).replace(hour=23, minute=59, second=59, microsecond=0)

        # 不对已经过去的时间进行判断：
        # 如果开始时间早于当前时间，则将开始时间提升为“此刻”
        now = datetime.now()
        if start_dt < now:
            start_dt = now
            # 如果时间范围被完全推到过去导致结束时间早于开始时间，让数据库查询自然返回空结果

        # type2 为时间范围，只有明确吉/凶才筛
        apply_jixiong_filter = preference in ["吉", "凶"]
        
        return await run_in_threadpool(
            _query_type2_data,
            specific_event,
            start_dt,
            end_dt,
            tian_gan_for_person,
            preference,
            apply_jixiong_filter,
            DB_CONFIG
        )
    
    elif qimen_type == "type3":
        # 具体时间点做什么事件
        if not time_range_start or not time_range_end:
            logger.error("类型3缺少必要参数")
            return []

        start_dt = datetime.strptime(time_range_start, "%Y-%m-%d %H:%M:%S")

        # 不对已经过去的时间进行判断：
        # 当用户问“这个月哪天适合做什么”这类问题时，第二层会给出当月月初作为开始时间，
        # 这里需要将已经过去的时间段裁掉，只从当前时间开始判断。
        now = datetime.now()
        if start_dt < now:
            start_dt = now

        start_dt = map_time_to_db_hour(start_dt)

        end_dt = datetime.strptime(time_range_end, "%Y-%m-%d %H:%M:%S")
        end_dt = map_time_to_db_hour(end_dt)

        # 仅当明确吉/凶且为范围时筛；单点或吉凶（无偏好）不筛
        apply_jixiong_filter = preference in ["吉", "凶"] and start_dt != end_dt
        
        return await run_in_threadpool(
            _query_type3_data,
            start_dt,
            end_dt,
            tian_gan_for_person,
            preference,
            apply_jixiong_filter,
            DB_CONFIG
        )
    
    else:
        logger.error(f"未知的奇门问题类型: {qimen_type}")
        return []

