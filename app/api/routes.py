import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy import Float, func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.dependencies import get_evaluator, get_llm_client, get_metrics, get_tracker
from app.db.database import get_db
from app.db.models import LLMRequest
from app.evaluation.evaluator import LLMEvaluator
from app.llm.client import LocalHFClient
from app.monitoring.metrics import MetricsCollector
from app.monitoring.mlflow_tracker import MLflowTracker
from app.schemas.request import BatchGenerateRequest, GenerateRequest
from app.schemas.response import (
    AnalyticsResponse,
    BatchGenerateResponse,
    GenerateResponse,
    HistoryRecord,
    HistoryResponse,
)

router = APIRouter()


def _process_single_request(
    prompt: str,
    client: LocalHFClient,
    evaluator: LLMEvaluator,
    tracker: MLflowTracker,
    metrics: MetricsCollector,
    db: Session,
    model_name: str | None = None,
    max_length: int | None = None,
    temperature: float | None = None,
) -> GenerateResponse:
    request_id = str(uuid.uuid4())
    start_time = time.time()

    effective_model = model_name or settings.LLM_MODEL_NAME
    gen_kwargs = {}
    if max_length is not None:
        gen_kwargs["max_length"] = max_length
    if temperature is not None:
        gen_kwargs["temperature"] = temperature

    try:
        raw_output = client.generate(prompt, **gen_kwargs)
        latency_ms = round((time.time() - start_time) * 1000, 2)
        token_count = client.count_tokens(raw_output)

        evaluation = evaluator.evaluate(prompt, raw_output)

        # Log to MLflow (non-blocking, errors caught internally)
        tracker.log_run(
            prompt=prompt,
            raw_output=raw_output,
            model_name=effective_model,
            relevance_score=evaluation.relevance_score,
            hallucination_flag=evaluation.hallucination_flag,
            toxicity_score=evaluation.toxicity_score,
            confidence_score=evaluation.confidence_score,
            latency_ms=latency_ms,
            token_count=token_count,
            request_id=request_id,
        )

        # Record metrics
        metrics.record_request(
            model_name=effective_model,
            latency_ms=latency_ms,
            relevance_score=evaluation.relevance_score,
            hallucination_flag=evaluation.hallucination_flag,
            toxicity_score=evaluation.toxicity_score,
        )

        # Persist to database
        db_record = LLMRequest(
            request_id=request_id,
            prompt=prompt,
            model_name=effective_model,
            raw_output=raw_output,
            relevance_score=evaluation.relevance_score,
            hallucination_flag=evaluation.hallucination_flag,
            toxicity_score=evaluation.toxicity_score,
            confidence_score=evaluation.confidence_score,
            latency_ms=latency_ms,
            token_count=token_count,
            status="success",
        )
        db.add(db_record)
        db.commit()

        return GenerateResponse(
            request_id=request_id,
            prompt=prompt,
            raw_output=raw_output,
            model_name=effective_model,
            evaluation=evaluation,
            latency_ms=latency_ms,
            token_count=token_count,
        )

    except Exception as e:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(f"Request {request_id} failed: {e}")

        metrics.record_request(
            model_name=effective_model,
            latency_ms=latency_ms,
            relevance_score=0.0,
            hallucination_flag=False,
            toxicity_score=0.0,
            success=False,
        )

        db_record = LLMRequest(
            request_id=request_id,
            prompt=prompt,
            model_name=effective_model,
            raw_output="",
            latency_ms=latency_ms,
            status="error",
            error_message=str(e),
        )
        db.add(db_record)
        db.commit()

        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=GenerateResponse)
def generate_text(
    request: GenerateRequest,
    client: LocalHFClient = Depends(get_llm_client),
    evaluator: LLMEvaluator = Depends(get_evaluator),
    tracker: MLflowTracker = Depends(get_tracker),
    metrics: MetricsCollector = Depends(get_metrics),
    db: Session = Depends(get_db),
):
    return _process_single_request(
        prompt=request.prompt,
        client=client,
        evaluator=evaluator,
        tracker=tracker,
        metrics=metrics,
        db=db,
        model_name=request.model_name,
        max_length=request.max_length,
        temperature=request.temperature,
    )


@router.post("/generate/batch", response_model=BatchGenerateResponse)
def batch_generate(
    request: BatchGenerateRequest,
    client: LocalHFClient = Depends(get_llm_client),
    evaluator: LLMEvaluator = Depends(get_evaluator),
    tracker: MLflowTracker = Depends(get_tracker),
    metrics: MetricsCollector = Depends(get_metrics),
    db: Session = Depends(get_db),
):
    batch_start = time.time()
    results = []

    for prompt in request.prompts:
        result = _process_single_request(
            prompt=prompt,
            client=client,
            evaluator=evaluator,
            tracker=tracker,
            metrics=metrics,
            db=db,
            model_name=request.model_name,
            max_length=request.max_length,
            temperature=request.temperature,
        )
        results.append(result)

    total_latency = round((time.time() - batch_start) * 1000, 2)
    return BatchGenerateResponse(
        results=results,
        total_latency_ms=total_latency,
        batch_size=len(results),
    )


@router.get("/history", response_model=HistoryResponse)
def get_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    model_name: str | None = Query(None),
    hallucination_only: bool = Query(False),
    db: Session = Depends(get_db),
):
    query = db.query(LLMRequest)

    if model_name:
        query = query.filter(LLMRequest.model_name == model_name)
    if hallucination_only:
        query = query.filter(LLMRequest.hallucination_flag.is_(True))

    total = query.count()
    records = (
        query.order_by(LLMRequest.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return HistoryResponse(
        total=total,
        page=page,
        page_size=page_size,
        records=[
            HistoryRecord(**r.to_dict()) for r in records
        ],
    )


@router.get("/analytics", response_model=AnalyticsResponse)
def get_analytics(
    model_name: str | None = Query(None),
    db: Session = Depends(get_db),
):
    query = db.query(LLMRequest).filter(LLMRequest.status == "success")

    if model_name:
        query = query.filter(LLMRequest.model_name == model_name)

    total = query.count()

    if total == 0:
        return AnalyticsResponse(
            total_requests=0,
            avg_relevance_score=None,
            avg_latency_ms=None,
            hallucination_rate=None,
            avg_toxicity_score=None,
            avg_confidence_score=None,
            model_breakdown={},
        )

    stats = db.query(
        func.avg(LLMRequest.relevance_score),
        func.avg(LLMRequest.latency_ms),
        func.avg(func.cast(LLMRequest.hallucination_flag, Float)),
        func.avg(LLMRequest.toxicity_score),
        func.avg(LLMRequest.confidence_score),
    ).filter(LLMRequest.status == "success")

    if model_name:
        stats = stats.filter(LLMRequest.model_name == model_name)

    row = stats.first()

    model_counts = (
        db.query(LLMRequest.model_name, func.count(LLMRequest.id))
        .filter(LLMRequest.status == "success")
        .group_by(LLMRequest.model_name)
        .all()
    )

    return AnalyticsResponse(
        total_requests=total,
        avg_relevance_score=round(row[0], 4) if row[0] else None,
        avg_latency_ms=round(row[1], 2) if row[1] else None,
        hallucination_rate=round(row[2], 4) if row[2] else None,
        avg_toxicity_score=round(row[3], 4) if row[3] else None,
        avg_confidence_score=round(row[4], 4) if row[4] else None,
        model_breakdown={name: count for name, count in model_counts},
    )
