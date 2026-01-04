from app.schemas.qimen import QimenDayRequest, QimenDayResponse


async def analyze_day(request: QimenDayRequest) -> QimenDayResponse:
    return QimenDayResponse(answer=f"placeholder qimen day for {request.date}")
