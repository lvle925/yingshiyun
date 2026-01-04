from app.schemas.ziwei import YearScoreRequest, YearScoreResponse


async def analyze_llm(request: YearScoreRequest) -> YearScoreResponse:
    return YearScoreResponse(summary=f"placeholder ziwei llm for {request.birthday}")
