from typing import Optional, List
from pydantic import BaseModel, Field, validator
from datetime import datetime
import ast

# --- 紫微斗数模型 ---
class YearlyAnalysisRequest(BaseModel):
    gender: str = Field(..., description="性别 (男/女)", example="男")
    birthday: str = Field(..., description="生辰 (yyyy-MM-dd HH:mm:ss格式)", example="1995-05-06 14:30:00")
    year: int = Field(..., description="年份 (例如2026)", example=2026)

    @validator('gender')
    def gender_must_be_valid(cls, v):
        if v not in ['男', '女']:
            raise ValueError('性别必须是 "男" 或 "女"')
        return v
        
    @validator('birthday')
    def birthday_format_must_be_valid(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            raise ValueError('生辰格式必须是 "yyyy-MM-dd HH:mm:ss"')
        return v

# --- 雷诺曼 LLM 模型 ---
class ClientRequest(BaseModel):
    appid: str = Field(..., description="应用ID")
    prompt: str = Field(..., description="用户的问题")
    format: str = Field("json", description="响应格式，默认为json")
    ftime: int = Field(..., description="时间戳 (整数)")
    sign: str = Field(..., description="请求签名")
    session_id: Optional[str] = Field(None, description="会话ID")
    hl_ymd: Optional[str] = Field(None, description="可选的日期参数")
    card_number_pool: Optional[List[int]] = Field(
        None, description="可选的卡牌编号列表"
    )
    score_level: str = Field(
        "0", description="占卜风格等级：'0', '1', '2'"
    )

    @validator('card_number_pool', pre=True, always=True)
    def parse_and_validate_card_number_pool(cls, v):
        if v is None: return None
        if isinstance(v, str):
            try: v = ast.literal_eval(v)
            except: raise ValueError("card_number_pool: 无效字符串")
        if not isinstance(v, list) or len(v) < 3 or not all(isinstance(i, int) for i in v):
            raise ValueError('card_number_pool 必须是至少3个整数的列表')
        return v

    @validator('score_level', pre=True, always=True)
    def validate_score_level(cls, v):
        if v not in ["0", "1", "2"]:
            raise ValueError('score_level 必须是 "0", "1", 或 "2"')
        return v