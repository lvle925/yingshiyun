from fastapi import FastAPI

from api_main_attributes import router as attributes_router
from api_main_calendar import router as calendar_router
from api_main_day import router as day_router
from app.api import (
    router as daily_router,
    init_daily_analysis_service,
    shutdown_daily_analysis_service,
)


app = FastAPI(
    title="奇门遁甲聚合服务",
    description="统一承载多套奇门遁甲 API 接口。",
    version="1.0.0",
)

app.include_router(attributes_router, tags=["Auspicious Info"])
app.include_router(calendar_router, tags=["Time Calendar"])
app.include_router(day_router, tags=["Choose Good Day"])
app.include_router(
    daily_router,
    tags=["Daily Fortune"],
)


@app.on_event("startup")
async def startup_event():
    await init_daily_analysis_service()


@app.on_event("shutdown")
async def shutdown_event():
    await shutdown_daily_analysis_service()

