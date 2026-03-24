from loguru import logger

from app.evaluation.hallucination import HallucinationDetector
from app.evaluation.relevance import RelevanceScorer
from app.evaluation.result import EvaluationResult
from app.evaluation.toxicity import ToxicityDetector


class LLMEvaluator:
    def __init__(
        self,
        relevance_model: str = "all-MiniLM-L6-v2",
        hallucination_threshold: float = 0.3,
        toxicity_threshold: float = 0.7,
    ):
        self.relevance_scorer = RelevanceScorer(model_name=relevance_model)
        self.hallucination_detector = HallucinationDetector(relevance_threshold=hallucination_threshold)
        self.toxicity_detector = ToxicityDetector(threshold=toxicity_threshold)

    def evaluate(self, prompt: str, output: str) -> EvaluationResult:
        logger.info("Starting evaluation pipeline")

        relevance_score = self.relevance_scorer.score(prompt, output)
        hallucination_flag, hallucination_reasons = self.hallucination_detector.detect(
            prompt, output, relevance_score
        )
        toxicity_score = self.toxicity_detector.score(output)

        # Confidence = weighted combination of signals
        confidence = self._compute_confidence(
            relevance_score, hallucination_flag, toxicity_score
        )

        result = EvaluationResult(
            relevance_score=round(relevance_score, 4),
            hallucination_flag=hallucination_flag,
            toxicity_score=round(toxicity_score, 4),
            confidence_score=round(confidence, 4),
            hallucination_reasons=hallucination_reasons,
            notes="auto-evaluated",
        )

        logger.info(
            f"Evaluation complete | relevance={result.relevance_score} "
            f"hallucination={result.hallucination_flag} "
            f"toxicity={result.toxicity_score} "
            f"confidence={result.confidence_score}"
        )
        return result

    def _compute_confidence(
        self,
        relevance: float,
        hallucination: bool,
        toxicity: float,
    ) -> float:
        score = relevance * 0.5
        if not hallucination:
            score += 0.3
        score += (1.0 - toxicity) * 0.2
        return max(0.0, min(1.0, score))
