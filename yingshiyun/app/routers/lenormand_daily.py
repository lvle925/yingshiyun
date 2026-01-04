from fastapi import APIRouter, Depends

from app.schemas.lenormand import DailyRequest, BasicResponse
from app.services import lenormand_daily as service

router = APIRouter()


@router.post("/daily", response_model=BasicResponse)
async def daily_fortune(body: DailyRequest) -> BasicResponse:
    return await service.run_daily_reading(body)
