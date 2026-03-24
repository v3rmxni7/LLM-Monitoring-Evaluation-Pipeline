from app.monitoring.metrics import MetricsCollector


class TestMetricsCollector:
    def setup_method(self):
        self.collector = MetricsCollector()

    def test_initial_state(self):
        summary = self.collector.get_summary()
        assert summary["total_requests"] == 0
        assert summary["successful_requests"] == 0
        assert summary["failed_requests"] == 0

    def test_record_successful_request(self):
        self.collector.record_request(
            model_name="distilgpt2",
            latency_ms=150.0,
            relevance_score=0.8,
            hallucination_flag=False,
            toxicity_score=0.1,
        )
        summary = self.collector.get_summary()
        assert summary["total_requests"] == 1
        assert summary["successful_requests"] == 1
        assert summary["failed_requests"] == 0
        assert summary["model_breakdown"]["distilgpt2"] == 1

    def test_record_failed_request(self):
        self.collector.record_request(
            model_name="distilgpt2",
            latency_ms=50.0,
            relevance_score=0.0,
            hallucination_flag=False,
            toxicity_score=0.0,
            success=False,
        )
        summary = self.collector.get_summary()
        assert summary["total_requests"] == 1
        assert summary["failed_requests"] == 1

    def test_hallucination_tracking(self):
        self.collector.record_request(
            model_name="distilgpt2",
            latency_ms=100.0,
            relevance_score=0.3,
            hallucination_flag=True,
            toxicity_score=0.0,
        )
        self.collector.record_request(
            model_name="distilgpt2",
            latency_ms=100.0,
            relevance_score=0.9,
            hallucination_flag=False,
            toxicity_score=0.0,
        )
        summary = self.collector.get_summary()
        assert summary["hallucination_rate"] == 0.5

    def test_prometheus_format(self):
        self.collector.record_request(
            model_name="distilgpt2",
            latency_ms=200.0,
            relevance_score=0.7,
            hallucination_flag=False,
            toxicity_score=0.05,
        )
        output = self.collector.get_prometheus_metrics()
        assert "llm_requests_total 1" in output
        assert "llm_requests_successful 1" in output
        assert "llm_avg_relevance_score" in output
        assert 'llm_model_requests{model="distilgpt2"}' in output

    def test_multiple_models(self):
        self.collector.record_request("model_a", 100, 0.5, False, 0.0)
        self.collector.record_request("model_b", 200, 0.7, False, 0.0)
        self.collector.record_request("model_a", 150, 0.6, True, 0.1)

        summary = self.collector.get_summary()
        assert summary["model_breakdown"]["model_a"] == 2
        assert summary["model_breakdown"]["model_b"] == 1
