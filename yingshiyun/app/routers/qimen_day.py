from fastapi import APIRouter

from app.schemas.qimen import QimenDayRequest, QimenDayResponse
from app.services import qimen_day as service

router = APIRouter()


@router.post("/analyze", response_model=QimenDayResponse)
async def analyze(body: QimenDayRequest) -> QimenDayResponse:
    return await service.analyze_day(body)
