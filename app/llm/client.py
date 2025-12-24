from transformers import pipeline
from app.core.logging import setup_logging

logger = setup_logging()

class LocalHFClient:
    def __init__(self, model_name: str = "distilgpt2"):
        logger.info(f"Loading local HF model: {model_name}")
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=-1  # CPU
        )

    def generate(self, prompt: str) -> str:
        logger.info("Generating response from local HF model")
        output = self.generator(
            prompt,
            max_length=100,
            num_return_sequences=1
        )
        return output[0]["generated_text"]
