from loguru import logger
from transformers import AutoTokenizer, pipeline

from app.llm.base import BaseLLMClient


class LocalHFClient(BaseLLMClient):
    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_length: int = 150,
        temperature: float = 1.0,
        device: int = -1,
    ):
        self._model_name = model_name
        self._max_length = max_length
        self._temperature = temperature
        self._device = device
        self._generator = None
        self._tokenizer = None

    def _load_model(self):
        if self._generator is not None:
            return
        logger.info(f"Loading HF model: {self._model_name} (device={self._device})")
        try:
            self._generator = pipeline(
                "text-generation",
                model=self._model_name,
                device=self._device,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            logger.info(f"Model {self._model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self._model_name}: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def generate(self, prompt: str, **kwargs) -> str:
        self._load_model()
        max_length = kwargs.get("max_length", self._max_length)
        temperature = kwargs.get("temperature", self._temperature)

        logger.info(f"Generating | model={self._model_name} max_length={max_length}")
        try:
            output = self._generator(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=temperature > 0,
            )
            return output[0]["generated_text"]
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"LLM generation failed: {e}") from e

    def count_tokens(self, text: str) -> int:
        self._load_model()
        try:
            return len(self._tokenizer.encode(text))
        except Exception:
            return len(text.split())

    def get_model_name(self) -> str:
        return self._model_name
