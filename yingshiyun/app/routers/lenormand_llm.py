from fastapi import APIRouter

from app.schemas.lenormand import LLMRequest, BasicResponse
from app.services import lenormand_llm as service

router = APIRouter()


@router.post("/chat", response_model=BasicResponse)
async def llm_chat(body: LLMRequest) -> BasicResponse:
    return await service.run_llm_reading(body)
