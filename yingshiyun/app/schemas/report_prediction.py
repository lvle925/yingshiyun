from typing import Dict, Any
from pydantic import BaseModel


class ReportPredictionRequest(BaseModel):
    payload: Dict[str, Any]


class ReportPredictionResponse(BaseModel):
    report: Dict[str, Any]
