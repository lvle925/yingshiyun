from fastapi import APIRouter

from app.schemas.qimen import QimenDayRequest, QimenDayResponse
from app.services import qimen_llm as service

router = APIRouter()


@router.post("/chat", response_model=QimenDayResponse)
async def chat(body: QimenDayRequest) -> QimenDayResponse:
    return await service.analyze_llm(body)
