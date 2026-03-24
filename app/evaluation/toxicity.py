import re

from loguru import logger


class ToxicityDetector:
    TOXIC_PATTERNS = [
        r"\b(hate|kill|attack|destroy|threat|abuse|harass)\b",
        r"\b(stupid|idiot|moron|dumb)\b",
        r"\b(racist|sexist|bigot)\b",
    ]

    SEVERITY_WEIGHTS = [0.8, 0.4, 0.9]

    def __init__(self, threshold: float = 0.7):
        self._threshold = threshold
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.TOXIC_PATTERNS]

    def score(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0

        total_score = 0.0
        text_lower = text.lower()
        word_count = max(len(text_lower.split()), 1)

        for pattern, weight in zip(self._compiled_patterns, self.SEVERITY_WEIGHTS):
            matches = pattern.findall(text_lower)
            if matches:
                density = len(matches) / word_count
                total_score += weight * min(density * 10, 1.0)

        final_score = min(total_score, 1.0)
        logger.debug(f"Toxicity score: {final_score:.4f}")
        return round(final_score, 4)
