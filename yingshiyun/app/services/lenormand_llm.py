from app.schemas.lenormand import LLMRequest, BasicResponse


async def run_llm_reading(request: LLMRequest) -> BasicResponse:
    return BasicResponse(message="lenormand llm placeholder", detail=request.model_dump())
