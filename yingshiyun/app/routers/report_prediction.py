from fastapi import APIRouter

from app.schemas.report_prediction import ReportPredictionRequest, ReportPredictionResponse
from app.services import report_prediction as service

router = APIRouter()


@router.post("/predict", response_model=ReportPredictionResponse)
async def predict(body: ReportPredictionRequest) -> ReportPredictionResponse:
    return await service.generate_report(body)
