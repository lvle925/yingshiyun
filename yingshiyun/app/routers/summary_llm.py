from fastapi import APIRouter

from app.schemas.summary import SummaryRequest, SummaryResponse
from app.services import summary_llm as service

router = APIRouter()


@router.post("/generate", response_model=SummaryResponse)
async def generate(body: SummaryRequest) -> SummaryResponse:
    return await service.generate_summary(body)
