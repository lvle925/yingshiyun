from app.schemas.ziwei import YearScoreRequest, YearScoreResponse


async def compute_year_score(request: YearScoreRequest) -> YearScoreResponse:
    return YearScoreResponse(summary=f"placeholder score for {request.year}")
