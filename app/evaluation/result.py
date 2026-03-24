
from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity between prompt and output")
    hallucination_flag: bool = Field(..., description="Whether hallucination was detected")
    toxicity_score: float = Field(0.0, ge=0.0, le=1.0, description="Toxicity probability score")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall confidence in the output quality")
    hallucination_reasons: list[str] = Field(default_factory=list, description="Reasons for hallucination flag")
    notes: str = Field("auto-evaluated", description="Evaluation notes")
