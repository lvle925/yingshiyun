from fastapi import FastAPI

from .routers import api_router
from .services.registry import iter_service_definitions


app = FastAPI(
    title="yingshiyun unified services",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
app.include_router(api_router)


@app.on_event("startup")
async def startup_event():
    for service in iter_service_definitions():
        service_app = service.load_app()
        service_app.state = app.state
        startup = service.load_startup()
        if startup is not None:
            await startup()


@app.on_event("shutdown")
async def shutdown_event():
    for service in iter_service_definitions():
        shutdown = service.load_shutdown()
        if shutdown is not None:
            await shutdown()
