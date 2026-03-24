from app.evaluation.hallucination import HallucinationDetector
from app.evaluation.toxicity import ToxicityDetector


class TestHallucinationDetector:
    def setup_method(self):
        self.detector = HallucinationDetector(relevance_threshold=0.3)

    def test_no_hallucination(self):
        flagged, reasons = self.detector.detect(
            prompt="What is Python?",
            output="Python is a high-level programming language known for its simplicity.",
            relevance_score=0.8,
        )
        assert flagged is False
        assert len(reasons) == 0

    def test_low_relevance_triggers(self):
        flagged, reasons = self.detector.detect(
            prompt="What is Python?",
            output="The weather is nice today in Paris.",
            relevance_score=0.1,
        )
        assert flagged is True
        assert any("relevance" in r.lower() for r in reasons)

    def test_prompt_echoing_detected(self):
        prompt = "What is Python?"
        flagged, reasons = self.detector.detect(
            prompt=prompt,
            output=prompt,
            relevance_score=0.9,
        )
        assert flagged is True
        assert any("echo" in r.lower() for r in reasons)

    def test_repetition_detected(self):
        flagged, reasons = self.detector.detect(
            prompt="Test",
            output="word word word word word word word word word word word word word word",
            relevance_score=0.5,
        )
        assert flagged is True
        assert any("repetition" in r.lower() for r in reasons)

    def test_suspicious_url_detected(self):
        flagged, reasons = self.detector.detect(
            prompt="Tell me about AI",
            output="According to https://fake-news.com/article AI is dangerous to all humans.",
            relevance_score=0.6,
        )
        assert flagged is True
        assert any("pattern" in r.lower() for r in reasons)

    def test_empty_output(self):
        flagged, reasons = self.detector.detect(
            prompt="Question",
            output="Question",
            relevance_score=0.9,
        )
        assert flagged is True


class TestToxicityDetector:
    def setup_method(self):
        self.detector = ToxicityDetector(threshold=0.7)

    def test_clean_text(self):
        score = self.detector.score("The weather is beautiful today.")
        assert score == 0.0

    def test_toxic_text(self):
        score = self.detector.score("I hate you, you stupid idiot moron.")
        assert score > 0.0

    def test_empty_text(self):
        score = self.detector.score("")
        assert score == 0.0

    def test_score_bounded(self):
        score = self.detector.score("hate kill attack destroy abuse harass")
        assert 0.0 <= score <= 1.0
