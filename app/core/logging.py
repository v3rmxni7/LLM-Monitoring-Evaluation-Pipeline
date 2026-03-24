import json
import sys

from loguru import logger

from app.core.config import settings


def json_sink(message):
    record = message.record
    log_entry = {
        "timestamp": str(record["time"]),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }
    if record.get("extra"):
        log_entry["extra"] = {
            k: v for k, v in record["extra"].items() if k != "request_id"
        }
        if "request_id" in record["extra"]:
            log_entry["request_id"] = record["extra"]["request_id"]
    print(json.dumps(log_entry), flush=True)


def setup_logging():
    logger.remove()

    if settings.ENV == "prod":
        logger.add(json_sink, level="INFO")
    else:
        logger.add(
            sys.stdout,
            level="DEBUG" if settings.DEBUG else "INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}",
        )

    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {module}:{function}:{line} | {message}",
    )

    return logger
