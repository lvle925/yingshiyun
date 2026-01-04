# -*- coding: utf-8 -*-
"""
用户信息提取模块
从用户输入中提取出生年月日、性别等信息，并将身份信息和问题意图剥离
"""

import re
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# 匹配用户信息格式的正则表达式
# 格式: (公历|农历)? YYYY-MM-DD HH:MM:SS 性别
BIRTH_INFO_PATTERN = re.compile(
    r"(公历|农历)?\s*(\d{4}-\d{1,2}-\d{1,2})\s+(\d{2}:\d{2}:\d{2})\s+(男|女)"
)

# 额外的日期格式匹配（用于更彻底地清理）
# 匹配各种可能的日期时间格式
ADDITIONAL_DATE_PATTERNS = [
    re.compile(r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日\s]+\d{1,2}[:：]\d{2}[:：]?\d{0,2}\s*[男女]'),  # 2024-01-01 12:00:00 男
    re.compile(r'\d{4}年\d{1,2}月\d{1,2}日\s+\d{1,2}[:：]\d{2}[:：]?\d{0,2}\s*[男女]'),  # 2024年1月1日 12:00 男
    re.compile(r'\d{4}\.\d{1,2}\.\d{1,2}\s+\d{1,2}[:：]\d{2}[:：]?\d{0,2}\s*[男女]'),  # 2024.1.1 12:00 男
]


def extract_user_info(prompt: str) -> Tuple[Optional[Dict[str, str]], str]:
    """
    从用户输入中提取用户信息和清理后的问题
    
    Args:
        prompt: 用户输入的原始问题
        
    Returns:
        Tuple[用户信息字典, 清理后的问题]
        用户信息格式: {
            'calendar_type': '公历' or '农历',
            'birthday': 'YYYY-MM-DD',
            'birth_time': 'HH:MM:SS',
            'gender': '男' or '女',
            'birth_datetime': 'YYYY-MM-DD HH:MM:SS'
        }
    """
    match = BIRTH_INFO_PATTERN.search(prompt)
    
    # 初始化清理后的问题
    cleaned_prompt = prompt
    
    if match:
        # 找到标准格式的用户信息
        calendar_type = match.group(1) or "公历"  # 默认为公历
        birthday = match.group(2)
        birth_time = match.group(3)
        gender = match.group(4)
        
        # 构建完整的出生日期时间字符串
        birth_datetime_str = f"{birthday} {birth_time}"
        
        user_info = {
            'calendar_type': calendar_type,
            'birthday': birthday,
            'birth_time': birth_time,
            'gender': gender,
            'birth_datetime': birth_datetime_str
        }
        
        # 从问题中移除用户信息
        cleaned_prompt = BIRTH_INFO_PATTERN.sub("", cleaned_prompt)
        
        logger.info(f"提取到用户信息: {user_info}, 清理后问题: {cleaned_prompt}")
    else:
        # 没有找到标准格式的用户信息，返回None
        user_info = None
    
    # 无论是否找到标准格式，都使用额外的模式进一步清理可能的日期格式
    # 这样可以确保所有可能的出生年月日格式都被清理
    for pattern in ADDITIONAL_DATE_PATTERNS:
        cleaned_prompt = pattern.sub("", cleaned_prompt)
    
    # 移除开头和结尾可能留下的空格、中文逗号、英文逗号
    cleaned_prompt = re.sub(r'^[\s,，]+|[\s,，]+$', '', cleaned_prompt).strip()
    
    # 移除可能残留的多个连续空格
    cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt).strip()
    
    if user_info:
        logger.info(f"最终清理后问题: {cleaned_prompt}")
    
    return user_info, cleaned_prompt


def get_day_stem_from_gregorian_date(birthday_dt: datetime) -> str:
    """
    根据公历日期计算对应的日干（天干）
    基于1900年1月1日是辛日的参考点
    """
    TIANGAN_LIST = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
    REFERENCE_DATE = datetime(1900, 1, 1)
    REFERENCE_STEM_INDEX = 7  # '辛' 的索引
    
    days_diff = (birthday_dt.date() - REFERENCE_DATE.date()).days
    day_stem_index = (REFERENCE_STEM_INDEX + days_diff) % 10
    return TIANGAN_LIST[day_stem_index]

