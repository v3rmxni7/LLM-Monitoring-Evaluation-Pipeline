from pydantic import BaseModel
from app.evaluation.result import EvaluationResult

class GenerateResponse(BaseModel):
    raw_output: str
    evaluation: EvaluationResult
