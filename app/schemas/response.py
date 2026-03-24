
from pydantic import BaseModel

from app.evaluation.result import EvaluationResult


class GenerateResponse(BaseModel):
    request_id: str
    prompt: str
    raw_output: str
    model_name: str
    evaluation: EvaluationResult
    latency_ms: float
    token_count: int


class BatchGenerateResponse(BaseModel):
    results: list[GenerateResponse]
    total_latency_ms: float
    batch_size: int


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    models_loaded: bool


class HistoryRecord(BaseModel):
    id: str
    request_id: str
    prompt: str
    model_name: str
    raw_output: str
    relevance_score: float | None
    hallucination_flag: bool | None
    toxicity_score: float | None
    confidence_score: float | None
    latency_ms: float | None
    token_count: int | None
    status: str
    created_at: str | None


class HistoryResponse(BaseModel):
    total: int
    page: int
    page_size: int
    records: list[HistoryRecord]


class AnalyticsResponse(BaseModel):
    total_requests: int
    avg_relevance_score: float | None
    avg_latency_ms: float | None
    hallucination_rate: float | None
    avg_toxicity_score: float | None
    avg_confidence_score: float | None
    model_breakdown: dict


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
