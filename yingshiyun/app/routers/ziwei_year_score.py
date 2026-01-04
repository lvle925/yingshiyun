from fastapi import APIRouter

from app.schemas.ziwei import YearScoreRequest, YearScoreResponse
from app.services import ziwei_year_score as service

router = APIRouter()


@router.post("/score", response_model=YearScoreResponse)
async def year_score(body: YearScoreRequest) -> YearScoreResponse:
    return await service.compute_year_score(body)
