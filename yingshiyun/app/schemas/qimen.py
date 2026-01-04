from pydantic import BaseModel, Field


class QimenDayRequest(BaseModel):
    date: str = Field(..., description="日期")
    question: str = Field(..., description="问题")


class QimenDayResponse(BaseModel):
    answer: str
