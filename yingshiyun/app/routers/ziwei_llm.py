from fastapi import APIRouter

from app.schemas.ziwei import YearScoreRequest, YearScoreResponse
from app.services import ziwei_llm as service

router = APIRouter()


@router.post("/chat", response_model=YearScoreResponse)
async def chat(body: YearScoreRequest) -> YearScoreResponse:
    return await service.analyze_llm(body)
