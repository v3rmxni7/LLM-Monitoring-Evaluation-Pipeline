from functools import lru_cache

from app.core.config import settings
from app.evaluation.evaluator import LLMEvaluator
from app.llm.client import LocalHFClient
from app.monitoring.metrics import MetricsCollector
from app.monitoring.mlflow_tracker import MLflowTracker


@lru_cache
def get_llm_client() -> LocalHFClient:
    return LocalHFClient(
        model_name=settings.LLM_MODEL_NAME,
        max_length=settings.LLM_MAX_LENGTH,
        temperature=settings.LLM_TEMPERATURE,
        device=settings.LLM_DEVICE,
    )


@lru_cache
def get_evaluator() -> LLMEvaluator:
    return LLMEvaluator(
        relevance_model=settings.RELEVANCE_MODEL_NAME,
        hallucination_threshold=settings.HALLUCINATION_RELEVANCE_THRESHOLD,
        toxicity_threshold=settings.TOXICITY_THRESHOLD,
    )


@lru_cache
def get_tracker() -> MLflowTracker:
    return MLflowTracker(
        tracking_uri=settings.MLFLOW_TRACKING_URI,
        experiment_name=settings.MLFLOW_EXPERIMENT_NAME,
    )


@lru_cache
def get_metrics() -> MetricsCollector:
    return MetricsCollector()
