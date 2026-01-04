from pydantic import BaseModel, Field


class YearScoreRequest(BaseModel):
    birthday: str = Field(..., description="出生时间 yyyy-MM-dd HH:mm:ss")
    gender: str = Field(..., description="男/女")
    year: int = Field(..., description="目标年份")


class YearScoreResponse(BaseModel):
    summary: str
