from app.schemas.lenormand import DailyRequest, BasicResponse


async def run_daily_reading(request: DailyRequest) -> BasicResponse:
    return BasicResponse(message="lenormand daily placeholder", detail=request.model_dump())
