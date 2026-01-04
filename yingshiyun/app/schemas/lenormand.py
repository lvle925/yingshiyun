from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class DailyRequest(BaseModel):
    prompt: str = Field(..., description="User question for daily reading")
    score_level: str = Field("0", description="Style level")
    card_number_pool: Optional[List[int]] = None


class LLMRequest(BaseModel):
    appid: str
    prompt: str
    ftime: int
    sign: str
    session_id: Optional[str] = None


class BasicResponse(BaseModel):
    message: str
    detail: Optional[Dict] = None
