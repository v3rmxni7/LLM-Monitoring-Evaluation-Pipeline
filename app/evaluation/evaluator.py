from app.evaluation.relevance import RelevanceScorer
from app.evaluation.hallucination import HallucinationDetector
from app.evaluation.result import EvaluationResult

class LLMEvaluator:
    def __init__(self):
        self.relevance = RelevanceScorer()
        self.hallucination = HallucinationDetector()

    def evaluate(self, prompt: str, output: str) -> EvaluationResult:
        relevance_score = self.relevance.score(prompt, output)
        hallucination_flag = self.hallucination.detect(
            prompt, output, relevance_score
        )

        return EvaluationResult(
            schema_valid=True,  # raw text always valid for now
            relevance_score=relevance_score,
            hallucination_flag=hallucination_flag,
            notes="auto-evaluated"
        )
