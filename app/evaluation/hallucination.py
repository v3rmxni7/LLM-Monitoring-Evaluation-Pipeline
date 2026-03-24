import re

from loguru import logger


class HallucinationDetector:
    SUSPICIOUS_PATTERNS = [
        r"\b(updated on|written by|published|article|copyright)\b",
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b",
        r"\bhttps?://\S+\b",
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # phone numbers
        r"\b[A-Z][a-z]+ [A-Z][a-z]+(?:,\s*(?:PhD|MD|CEO|CTO|Prof\.))\b",  # fabricated attributions
    ]

    def __init__(self, relevance_threshold: float = 0.3):
        self._relevance_threshold = relevance_threshold
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.SUSPICIOUS_PATTERNS]

    def detect(self, prompt: str, output: str, relevance_score: float) -> tuple[bool, list[str]]:
        reasons = []

        # 1. Prompt echoing — output just repeats the prompt
        cleaned_output = output[len(prompt):].strip() if output.startswith(prompt) else output
        if not cleaned_output or len(cleaned_output) < 5:
            reasons.append("Output is empty or trivially echoes the prompt")

        # 2. Low semantic relevance
        if relevance_score < self._relevance_threshold:
            reasons.append(f"Low relevance score ({relevance_score:.3f} < {self._relevance_threshold})")

        # 3. Excessive repetition
        words = cleaned_output.split() if cleaned_output else []
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                reasons.append(f"Excessive repetition detected (unique ratio: {unique_ratio:.2f})")

        # 4. Suspicious patterns (fabricated metadata, URLs, etc.)
        for i, pattern in enumerate(self._compiled_patterns):
            if pattern.search(cleaned_output):
                reasons.append(f"Suspicious pattern detected: {self.SUSPICIOUS_PATTERNS[i]}")
                break

        # 5. Incoherent sentence structure — sentences that are too long without punctuation
        sentences = re.split(r'[.!?]+', cleaned_output)
        for sentence in sentences:
            if len(sentence.split()) > 50:
                reasons.append("Incoherent output: excessively long sentence without punctuation")
                break

        flagged = len(reasons) > 0
        if flagged:
            logger.warning(f"Hallucination detected: {reasons}")
        else:
            logger.debug("No hallucination detected")

        return flagged, reasons
