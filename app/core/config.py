
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "LLM Monitoring & Evaluation Pipeline"
    APP_VERSION: str = "2.0.0"
    ENV: str = "dev"
    DEBUG: bool = False

    # API
    API_V1_PREFIX: str = "/api/v1"
    API_KEY: str | None = None
    RATE_LIMIT_PER_MINUTE: int = 60

    # LLM
    LLM_MODEL_NAME: str = "distilgpt2"
    LLM_MAX_LENGTH: int = 150
    LLM_TEMPERATURE: float = 1.0
    LLM_NUM_RETURN_SEQUENCES: int = 1
    LLM_DEVICE: int = -1  # -1 = CPU, 0+ = GPU index

    # Evaluation
    RELEVANCE_MODEL_NAME: str = "all-MiniLM-L6-v2"
    HALLUCINATION_RELEVANCE_THRESHOLD: float = 0.3
    TOXICITY_THRESHOLD: float = 0.7

    # MLflow
    MLFLOW_TRACKING_URI: str = "file:./mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "llm-monitoring"

    # Database
    DATABASE_URL: str = "sqlite:///./llm_monitor.db"

    # Prometheus
    ENABLE_METRICS: bool = True

    model_config = {"env_file": ".env", "case_sensitive": True}


settings = Settings()
