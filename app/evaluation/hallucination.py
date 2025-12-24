class HallucinationDetector:
    def detect(self, prompt: str, output: str, relevance_score: float) -> bool:
        # 1. Prompt echoing
        if output.lower().startswith(prompt.lower()):
            return True

        # 2. Very low relevance
        if relevance_score < 0.4:
            return True

        # 3. Suspicious metadata hallucinations
        keywords = ["updated on", "written by", "article", "published"]
        if any(k in output.lower() for k in keywords):
            return True

        return False
