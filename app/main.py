from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger

from app.api.middleware import (
    APIKeyMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
    RequestLoggingMiddleware,
)
from app.api.routes import router as api_router
from app.core.config import settings
from app.core.dependencies import get_metrics
from app.core.logging import setup_logging
from app.db.database import init_db

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} [{settings.ENV}]")
    init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade LLM monitoring and evaluation pipeline with hallucination detection, toxicity scoring, and experiment tracking.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware (order matters — outermost first)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=settings.RATE_LIMIT_PER_MINUTE)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(api_router, prefix=settings.API_V1_PREFIX, tags=["LLM"])


@app.get("/health", tags=["System"])
def health_check():
    return JSONResponse(content={
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.ENV,
    })


@app.get("/metrics", tags=["System"], response_class=PlainTextResponse)
def prometheus_metrics():
    metrics = get_metrics()
    return metrics.get_prometheus_metrics()
