from fastapi import APIRouter

from app.schemas.ziwei import YearScoreRequest, YearScoreResponse
from app.services import ziwei_report_year as service

router = APIRouter()


@router.post("/report", response_model=YearScoreResponse)
async def report(body: YearScoreRequest) -> YearScoreResponse:
    return await service.generate_report(body)
