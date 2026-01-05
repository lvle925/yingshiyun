# -*- coding: utf-8 -*-
"""
LLM响应生成模块
用于生成最终的奇门分析结果
"""

import aiohttp
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from threading import Lock
from config import VLLM_API_BASE_URL, VLLM_MODEL_NAME, API_KEY

logger = logging.getLogger(__name__)

# 提示词模板管理
_prompt_template: Optional[str] = None
_prompt_template_lock = Lock()
_prompts_dir = Path(__file__).parent / "prompts"


def load_prompt_template(filename: str = "final_response_prompt.xml") -> Optional[str]:
    """从XML文件加载提示词模板"""
    try:
        prompt_file = _prompts_dir / filename
        if not prompt_file.exists():
            logger.error(f"提示词文件不存在: {prompt_file}")
            return None
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"成功加载提示词文件: {filename}")
        return content
    except Exception as e:
        logger.error(f"加载提示词文件 {filename} 失败: {e}", exc_info=True)
        return None


def reload_prompt_template():
    """重新加载提示词模板（支持热更新）"""
    global _prompt_template
    
    with _prompt_template_lock:
        logger.info("开始重新加载提示词模板...")
        _prompt_template = load_prompt_template("final_response_prompt.xml")
        if _prompt_template:
            logger.info("✓ 提示词模板加载成功")
        else:
            logger.error("✗ 提示词模板加载失败")


def get_prompt_template() -> Optional[str]:
    """获取提示词模板"""
    with _prompt_template_lock:
        if _prompt_template is None:
            reload_prompt_template()
        return _prompt_template


def _db_hour_to_time_range(hour: int) -> str:
    """
    将数据库存储的时辰整点映射回对应的时间范围
    数据库存储格式：2点表示1:00-3:00，8点表示7:00-9:00
    """
    mapping = {
        0: "23:00-01:00",
        2: "01:00-03:00",
        4: "03:00-05:00",
        6: "05:00-07:00",
        8: "07:00-09:00",
        10: "09:00-11:00",
        12: "11:00-13:00",
        14: "13:00-15:00",
        16: "15:00-17:00",
        18: "17:00-19:00",
        20: "19:00-21:00",
        22: "21:00-23:00",
    }
    return mapping.get(hour, f"{hour:02d}:00-{hour+2:02d}:00")


def format_qimen_data_for_prompt(qimen_data: List[Dict[str, Any]]) -> str:
    """
    将奇门数据格式化为提示词中的文本
    """
    if not qimen_data:
        return "您所提问的时间段诸事不宜，请另外择时。"
    
    formatted_lines = []
    for idx, row in enumerate(qimen_data, 1):
        date_str_raw = row.get('date_str', '未知时间')
        date_str = date_str_raw
        time_range = "未知时间段"
        if isinstance(date_str_raw, str):
            try:
                dt = datetime.strptime(date_str_raw, "%Y-%m-%d %H:%M:%S")
                date_str = dt.strftime("%Y年%m月%d日")
                time_range = _db_hour_to_time_range(dt.hour)
            except:
                pass
        
        jixiong = row.get('吉凶', '未知')
        total_score = row.get('total_score', 0)
        specific_matter = row.get('具体事项', '未知事项')
        
        tianpan_gan = row.get('tianpanGan', '无')
        dipan = row.get('dipan', '无')
        renpan = row.get('renpan', '无')
        shenpan = row.get('shenpan', '无')
        tianpan_xing = row.get('tianpanXing_ori', '无')
        geju_names = row.get('geju_names', '无')
        qiyi_zuhe_name = row.get('qiyi_zuhe_name', '无')
        
        # 判断是否合适
        gong_status_col = [k for k in row.keys() if k.startswith('gong_relation_status_')]
        gong_status = row.get(gong_status_col[0], '无') if gong_status_col else '无'
        
        is_suitable = "适合" if (jixiong == '吉' and gong_status in ['吉', '平']) or (jixiong == '平' and gong_status == '吉') else "不适合"
        
        line = f"""
数据 {idx}:
- 日期: {date_str} {time_range}
- 具体事项: {specific_matter}
- 是否合适: {is_suitable}
- 吉凶: {jixiong}
- 综合评分: {total_score}
- 天盘天干: {tianpan_gan}
- 地盘: {dipan}
- 人盘: {renpan}
- 神盘: {shenpan}
- 天盘星: {tianpan_xing}
- 格局: {geju_names}
- 奇仪组合: {qiyi_zuhe_name}
- 宫位关系: {gong_status}
"""
        formatted_lines.append(line)
    
    return "\n".join(formatted_lines)


async def generate_final_response(
    user_question: str,
    qimen_data: List[Dict[str, Any]],
    qimen_type: str,
    original_time_text: str,
    jixiong_preference: str,
    async_client: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore
) -> str:
    """
    生成最终的奇门分析结果
    
    Args:
        user_question: 用户原始问题
        qimen_data: 奇门局数据列表
        qimen_type: 奇门问题类型（type1/type2/type3）
        original_time_text: 用户原始时间表述
        jixiong_preference: 第二层识别出的吉凶偏好（"吉"|"凶"|"吉凶"）
        async_client: aiohttp客户端
        semaphore: 并发控制信号量
    
    Returns:
        最终的分析结果文本
    """
    template = get_prompt_template()
    if not template:
        return "抱歉，系统错误，无法生成分析结果。"
    
    # 格式化奇门数据
    formatted_data = format_qimen_data_for_prompt(qimen_data)
    
    # 替换占位符
    prompt = template.replace("{user_question}", user_question)
    prompt = prompt.replace("{qimen_data}", formatted_data)
    prompt = prompt.replace("{qimen_type}", qimen_type or "type1")
    prompt = prompt.replace("{original_time_text}", original_time_text or "")
    prompt = prompt.replace("{jixiong_preference}", jixiong_preference or "吉凶")

    
    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": False
    }
    
    url = f"{VLLM_API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=1500)
        try:
            timeout = aiohttp.ClientTimeout(total=500)
            async with async_client.post(url, json=payload, headers=headers, timeout=timeout) as response:
                response.raise_for_status()
                json_response = await response.json()
                content = json_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                # 调试输出：打印大模型原始content（便于排查，后续可删除）
                try:
                    logger.info("[DEBUG_LLM_RAW_CONTENT_START]\n%s\n[DEBUG_LLM_RAW_CONTENT_END]", content)
                except Exception:
                    pass
                if not content:
                    return "抱歉，无法生成分析结果，请稍后重试。"
                return content.strip()
        finally:
            semaphore.release()
    except Exception as e:
        logger.error(f"生成最终响应失败: {e}", exc_info=True)
        return "抱歉，生成分析结果时发生错误，请稍后重试。"


def start_prompt_file_watcher():
    """
    启动提示词文件监控（支持热更新）
    注意：需要 watchdog 库支持
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class PromptFileHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if event.src_path.endswith('.xml'):
                    logger.info(f"检测到提示词文件变化: {event.src_path}")
                    reload_prompt_template()
        
        observer = Observer()
        event_handler = PromptFileHandler()
        observer.schedule(event_handler, path=str(_prompts_dir), recursive=False)
        observer.start()
        logger.info(f"✓ 提示词文件监控已启动，监控目录: {_prompts_dir.absolute()}")
        return observer
    except ImportError:
        logger.warning("未安装 watchdog 库，热更新功能不可用。修改提示词需要重启服务。")
        return None
    except Exception as e:
        logger.error(f"启动提示词文件监控失败: {e}", exc_info=True)
        return None


# 初始化：在模块加载时预加载提示词
reload_prompt_template()

