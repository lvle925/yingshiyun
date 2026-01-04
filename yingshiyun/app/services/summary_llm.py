from app.schemas.summary import SummaryRequest, SummaryResponse


async def generate_summary(request: SummaryRequest) -> SummaryResponse:
    return SummaryResponse(summary=f"summary placeholder: {request.text[:50]}")
