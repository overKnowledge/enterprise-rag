import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1 import chat, documents, health
from app.core.config import get_settings
from app.core.exceptions import unhandled_exception_handler
from app.core.logging import setup_logging
from app.db.session import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    init_db()
    logger.info("Starting %s (%s)", settings.app_name, settings.environment)
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    return app


app = create_app()