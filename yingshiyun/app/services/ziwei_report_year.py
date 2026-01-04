from app.schemas.ziwei import YearScoreRequest, YearScoreResponse


async def generate_report(request: YearScoreRequest) -> YearScoreResponse:
    return YearScoreResponse(summary=f"placeholder report for year {request.year}")
