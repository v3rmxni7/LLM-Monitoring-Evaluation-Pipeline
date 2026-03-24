from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RelevanceScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        logger.info(f"Loading embedding model: {self._model_name}")
        try:
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Embedding model loading failed: {e}") from e

    def score(self, prompt: str, output: str) -> float:
        self._load_model()
        try:
            embeddings = self._model.encode([prompt, output])
            similarity = cosine_similarity(
                [embeddings[0]], [embeddings[1]]
            )[0][0]
            score = float(max(0.0, min(1.0, similarity)))
            logger.debug(f"Relevance score: {score:.4f}")
            return score
        except Exception as e:
            logger.error(f"Relevance scoring failed: {e}")
            return 0.0
