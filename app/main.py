from fastapi import FastAPI
from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import router

logger = setup_logging()

app = FastAPI(title=settings.APP_NAME)

app.include_router(router)

@app.get("/health")
def health_check():
    logger.info("Health check called")
    return {"status": "ok", "env": settings.ENV}
