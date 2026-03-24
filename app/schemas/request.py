
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The input prompt for LLM generation",
        examples=["Explain LLM hallucination in one sentence."],
    )
    model_name: str | None = Field(
        None,
        description="Override model name for this request",
    )
    max_length: int | None = Field(
        None,
        ge=10,
        le=1000,
        description="Override max generation length",
    )
    temperature: float | None = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Override sampling temperature",
    )


class BatchGenerateRequest(BaseModel):
    prompts: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of prompts for batch generation",
    )
    model_name: str | None = None
    max_length: int | None = Field(None, ge=10, le=1000)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
