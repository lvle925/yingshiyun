from app.schemas.report_prediction import ReportPredictionRequest, ReportPredictionResponse


async def generate_report(request: ReportPredictionRequest) -> ReportPredictionResponse:
    return ReportPredictionResponse(report=request.payload)
