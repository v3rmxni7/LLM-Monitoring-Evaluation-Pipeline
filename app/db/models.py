import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text

from app.db.database import Base


class LLMRequest(Base):
    __tablename__ = "llm_requests"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id = Column(String, unique=True, index=True, nullable=False)
    prompt = Column(Text, nullable=False)
    model_name = Column(String(100), nullable=False)
    raw_output = Column(Text, nullable=False)

    # Evaluation
    relevance_score = Column(Float, nullable=True)
    hallucination_flag = Column(Boolean, nullable=True)
    toxicity_score = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)

    # Metadata
    latency_ms = Column(Float, nullable=True)
    token_count = Column(Integer, nullable=True)
    status = Column(String(20), default="success")
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "request_id": self.request_id,
            "prompt": self.prompt,
            "model_name": self.model_name,
            "raw_output": self.raw_output,
            "relevance_score": self.relevance_score,
            "hallucination_flag": self.hallucination_flag,
            "toxicity_score": self.toxicity_score,
            "confidence_score": self.confidence_score,
            "latency_ms": self.latency_ms,
            "token_count": self.token_count,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
