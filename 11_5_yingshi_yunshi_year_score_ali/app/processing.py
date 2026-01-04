import asyncio
import json
import logging
import requests
import aiohttp
import time
from typing import Dict, Any
import math

# 导入配置和辅助函数
from .config import ASTRO_API_URL
from .utils import scale_score_to_100
from .utils import transform_horoscope_scope_data, transform_palace_data, parse_palace_data, calculate_score_by_rules1 as calculate_score_by_rules
from .astro_api_client import AstroAPIClient

logger = logging.getLogger(__name__)


async def _compute_scores(dateStr: str, type: str, timeIndex: int, gender: str, horoscopeDate: str, log_details: bool = True) -> dict:
    start_time = time.time()
    if log_details:
        logger.info(f"开始处理年份: {horoscopeDate}")
    
    payload = {
        "dateStr": dateStr,
        "type": type,
        "timeIndex": timeIndex,
        "gender": gender,
        "horoscopeDate": horoscopeDate
    }
    
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = await asyncio.to_thread(
            requests.post, ASTRO_API_URL, 
            data=json.dumps(payload), 
            headers=headers, 
            timeout=30
        )
        response.raise_for_status()
        response_data = response.json()
        
        astrolabe_palaces = response_data['data']['astrolabe']['palaces']
        transformed_horoscope_scopes = {
            "大限盘": transform_horoscope_scope_data(
                response_data['data']['horoscope'].get('decadal', {}),
                astrolabe_palaces),
            "流年盘": transform_horoscope_scope_data(
                response_data['data']['horoscope'].get('yearly', {}),
                astrolabe_palaces),
        }
        
        liu_nian_chart = transformed_horoscope_scopes.get("流年盘", [])
        da_xian_chart = transformed_horoscope_scopes.get("大限盘", [])
        
        if not liu_nian_chart:
            return {"error": "无法获取流年盘数据"}
        
        raw_astrolabe_palaces = transform_palace_data({
            "palaces": astrolabe_palaces,
            "chineseDate": response_data['data']['astrolabe'].get('chineseDate', "")
        })
        liu_nian_parsed = parse_palace_data(liu_nian_chart)
        da_xian_parsed = parse_palace_data(da_xian_chart) if da_xian_chart else []
        
        
        final_results_list = []
        palace_details = []
        for palace_info_dict in liu_nian_parsed:
            palace_name = palace_info_dict.get('palace_name', '未知宫位')
            score, source_scores, summary_dict, negative_score_sum, detail_events = calculate_score_by_rules(
                palace_info_dict,
                liu_nian_parsed,
                da_xian_parsed
            )
            final_results_list.append({
                "palace_name": palace_name,
                "final_score": score
            })

            palace_details.append({
                "palace_name": palace_name,
                "score": score,
                "source_scores": source_scores,
                "negative_score_sum": negative_score_sum,
                "main_stars": palace_info_dict.get('main_stars', []),
                "aux_stars": palace_info_dict.get('aux_stars', []),
                "si_hua": palace_info_dict.get('si_hua', []),
                "events": detail_events
            })
        
        palace_weights = {
            "官禄宫": 1.0,
            "财帛宫": 1.0,
            "夫妻宫": 1.0,
            "迁移宫": 0.5,
            "疾厄宫": 0.5,
            "仆役宫": 0.5,
        }
        
        scaled_scores_map = {}
        weighted_total_score = 0
        raw_scores_map = {item['palace_name']: item['final_score'] for item in final_results_list}
        
        for palace in palace_weights.keys():
            raw_score = raw_scores_map.get(palace, 0)
            scaled_score = scale_score_to_100(raw_score)
            scaled_scores_map[f"{palace}_100"] = scaled_score
            if palace in ["官禄宫","财帛宫","夫妻宫"]:
                weighted_total_score += scaled_score
        
        composite_score = int(weighted_total_score / 3)
        
        if log_details:
            logger.info(f"计算完成，耗时: {time.time() - start_time:.2f}秒")
        
        return {
            "scaled_scores_map": scaled_scores_map,
            "composite_score": composite_score,
            "raw_scores_map": raw_scores_map,
            "palace_details": palace_details,
            "astrolabe_palaces": raw_astrolabe_palaces,
            "liu_nian_palaces": liu_nian_parsed,
            "da_xian_palaces": da_xian_parsed
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"请求发生错误: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"处理数据时发生意外错误: {e}", exc_info=True)
        return {"error": f"内部错误: {e}"}


async def get_yearly_scores_only(dateStr: str, type: str, timeIndex: int, gender: str, horoscopeDate: str) -> dict:
    return await _compute_scores(dateStr, type, timeIndex, gender, horoscopeDate, log_details=True)


async def get_monthly_scores(dateStr: str, type: str, timeIndex: int, gender: str, year: int) -> list:
    client = AstroAPIClient(
        api_url=ASTRO_API_URL,
        birth_info={
            "dateStr": dateStr,
            "type": type,
            "timeIndex": timeIndex,
            "gender": gender
        },
        astro_type="heaven",
        query_year=year,
        save_results=False
    )
    async with aiohttp.ClientSession() as session:
        monthly_results = await client.run_query_async(session=session, year=year, concurrency=5)
    
    monthly_outputs = []
    for item in monthly_results:
        api_result = item.get("api_result", {})
        if not api_result.get("success"):
            logger.error("获取农历月份数据失败: %s", api_result.get("error"))
            continue
        monthly_data = api_result.get("data", {})
        try:
            data_block = monthly_data.get('data', {})
            horoscope_block = data_block.get('horoscope', {})
            astrolabe_block = data_block.get('astrolabe', {})
            decadal_scope = horoscope_block.get('decadal', {})
            monthly_scope = horoscope_block.get('monthly', {})
            if not isinstance(decadal_scope, dict):
                decadal_scope = {}
            if not isinstance(monthly_scope, dict):
                monthly_scope = {}
                logger.warning("月度数据缺少 'monthly' 盘，使用空结构")
            transformed_scopes = {
                "大限盘": transform_horoscope_scope_data(
                    decadal_scope,
                    astrolabe_block.get('palaces', [])
                ),
                "流年盘": transform_horoscope_scope_data(
                    monthly_scope,
                    astrolabe_block.get('palaces', [])
                ),
            }
            liu_nian_chart = transformed_scopes.get("流年盘", [])
            da_xian_chart = transformed_scopes.get("大限盘", [])
            liu_nian_parsed = parse_palace_data(liu_nian_chart)
            da_xian_parsed = parse_palace_data(da_xian_chart) if da_xian_chart else []
            final_results_list = []
            for palace_info_dict in liu_nian_parsed:
                palace_name = palace_info_dict.get('palace_name', '未知宫位')
                score, _, _, _, _ = calculate_score_by_rules(
                    palace_info_dict,
                    liu_nian_parsed,
                    da_xian_parsed
                )
                final_results_list.append({
                    "palace_name": palace_name,
                    "final_score": score
                })
            scaled_scores_map = {}
            raw_scores_map = {entry['palace_name']: entry['final_score'] for entry in final_results_list}
            for palace in ["官禄宫", "财帛宫", "夫妻宫", "迁移宫", "疾厄宫", "仆役宫"]:
                scaled_scores_map[f"{palace}_100"] = scale_score_to_100(raw_scores_map.get(palace, 0))
            monthly_outputs.append({
                "month": item.get("lunar_month"),
                "horoscopeDate": item.get("horoscope_date"),
                "scaled_scores_map": scaled_scores_map,
                "raw_scores_map": raw_scores_map,
                "composite_score": scale_score_to_100(
                    sum(raw_scores_map.get(p, 0) for p in ["官禄宫", "财帛宫", "夫妻宫"]) / 3
                )
            })
        except Exception as e:
            logger.error("解析农历月份数据时发生异常: %s", e, exc_info=True)
            continue
    monthly_outputs.sort(key=lambda x: x["month"])
    return monthly_outputs

