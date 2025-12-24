from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "LLM Monitoring & Evaluation Service"
    ENV: str = "dev"

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_MODEL: str = "llama3"

    class Config:
        env_file = ".env"

settings = Settings()
