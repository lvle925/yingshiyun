import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import (
    lenormand_daily,
    lenormand_llm,
    ziwei_year_score,
    qimen_day,
    qimen_llm,
    ziwei_llm,
    summary_llm,
    report_prediction,
    ziwei_report_year,
)
from app.core.config import get_settings

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    app = FastAPI(
        title="yingshiyun",
        description="Unified Router-Service-Schema FastAPI stack for Yingshi services",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers (keep tags to mirror original services)
    app.include_router(lenormand_daily.router, prefix="/lenormand", tags=["lenormand_daily"])
    app.include_router(lenormand_llm.router, prefix="/lenormand_llm", tags=["lenormand_llm"])
    app.include_router(ziwei_year_score.router, prefix="/ziwei_year", tags=["ziwei_year_score"])
    app.include_router(qimen_day.router, prefix="/qimen_day", tags=["qimen_day"])
    app.include_router(qimen_llm.router, prefix="/qimen_llm", tags=["qimen_llm"])
    app.include_router(ziwei_llm.router, prefix="/ziwei_llm", tags=["ziwei_llm"])
    app.include_router(summary_llm.router, prefix="/summary", tags=["summary_llm"])
    app.include_router(report_prediction.router, prefix="/report_prediction", tags=["report_prediction"])
    app.include_router(ziwei_report_year.router, prefix="/ziwei_report_year", tags=["ziwei_report_year"])

    return app


app = create_app()


@app.get("/health", tags=["meta"])
async def healthcheck():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
