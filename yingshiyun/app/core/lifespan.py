from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await db.init_db_pool()
    try:
        yield
    finally:
        # Shutdown
        await db.close_db_pool()
