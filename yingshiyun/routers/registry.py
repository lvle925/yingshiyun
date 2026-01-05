from fastapi import APIRouter

from ..services.registry import iter_service_definitions

api_router = APIRouter()

for service in iter_service_definitions():
    service_app = service.load_app()
    api_router.include_router(service_app.router)
