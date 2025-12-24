from fastapi import APIRouter
from app.schemas.request import GenerateRequest
from app.schemas.response import GenerateResponse
from app.llm.client import LocalHFClient
from app.evaluation.evaluator import LLMEvaluator
from app.monitoring.mlflow_tracker import MLflowTracker

tracker = MLflowTracker()

router = APIRouter()

client = LocalHFClient()
evaluator = LLMEvaluator()

@router.get("/")
def root():
    return {"message": "LLM Monitoring & Evaluation API"}

@router.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    output = client.generate(request.prompt)
    evaluation = evaluator.evaluate(request.prompt, output)

    tracker.log_run(
        prompt=request.prompt,
        raw_output=output,
        relevance_score=evaluation.relevance_score,
        hallucination_flag=evaluation.hallucination_flag,
        model_name="distilgpt2"
    )

    return GenerateResponse(
        raw_output=output,
        evaluation=evaluation
    )
