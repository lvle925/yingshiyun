from pydantic import BaseModel, Field


class SummaryRequest(BaseModel):
    text: str = Field(..., description="待总结文本")


class SummaryResponse(BaseModel):
    summary: str
