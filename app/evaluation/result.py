from pydantic import BaseModel

class EvaluationResult(BaseModel):
    schema_valid: bool
    relevance_score: float
    hallucination_flag: bool
    notes: str
