from app.schemas.qimen import QimenDayRequest, QimenDayResponse


async def analyze_llm(request: QimenDayRequest) -> QimenDayResponse:
    return QimenDayResponse(answer=f"placeholder qimen llm for {request.question}")
