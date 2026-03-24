from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class RequestMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    hallucinations_detected: int = 0
    total_relevance_score: float = 0.0
    total_toxicity_score: float = 0.0
    model_request_counts: dict = field(default_factory=lambda: defaultdict(int))


class MetricsCollector:
    def __init__(self):
        self._metrics = RequestMetrics()
        self._lock = Lock()

    def record_request(
        self,
        model_name: str,
        latency_ms: float,
        relevance_score: float,
        hallucination_flag: bool,
        toxicity_score: float,
        success: bool = True,
    ):
        with self._lock:
            self._metrics.total_requests += 1
            self._metrics.total_latency_ms += latency_ms
            self._metrics.model_request_counts[model_name] += 1

            if success:
                self._metrics.successful_requests += 1
                self._metrics.total_relevance_score += relevance_score
                self._metrics.total_toxicity_score += toxicity_score
                if hallucination_flag:
                    self._metrics.hallucinations_detected += 1
            else:
                self._metrics.failed_requests += 1

    def get_prometheus_metrics(self) -> str:
        with self._lock:
            m = self._metrics
            total = max(m.total_requests, 1)
            successful = max(m.successful_requests, 1)

            lines = [
                "# HELP llm_requests_total Total number of LLM requests",
                "# TYPE llm_requests_total counter",
                f"llm_requests_total {m.total_requests}",
                "",
                "# HELP llm_requests_successful Successful LLM requests",
                "# TYPE llm_requests_successful counter",
                f"llm_requests_successful {m.successful_requests}",
                "",
                "# HELP llm_requests_failed Failed LLM requests",
                "# TYPE llm_requests_failed counter",
                f"llm_requests_failed {m.failed_requests}",
                "",
                "# HELP llm_avg_latency_ms Average request latency in milliseconds",
                "# TYPE llm_avg_latency_ms gauge",
                f"llm_avg_latency_ms {m.total_latency_ms / total:.2f}",
                "",
                "# HELP llm_hallucination_rate Rate of hallucination detections",
                "# TYPE llm_hallucination_rate gauge",
                f"llm_hallucination_rate {m.hallucinations_detected / successful:.4f}",
                "",
                "# HELP llm_avg_relevance_score Average relevance score",
                "# TYPE llm_avg_relevance_score gauge",
                f"llm_avg_relevance_score {m.total_relevance_score / successful:.4f}",
                "",
                "# HELP llm_avg_toxicity_score Average toxicity score",
                "# TYPE llm_avg_toxicity_score gauge",
                f"llm_avg_toxicity_score {m.total_toxicity_score / successful:.4f}",
                "",
            ]

            for model, count in m.model_request_counts.items():
                lines.append(f'llm_model_requests{{model="{model}"}} {count}')

            return "\n".join(lines) + "\n"

    def get_summary(self) -> dict:
        with self._lock:
            m = self._metrics
            total = max(m.total_requests, 1)
            successful = max(m.successful_requests, 1)
            return {
                "total_requests": m.total_requests,
                "successful_requests": m.successful_requests,
                "failed_requests": m.failed_requests,
                "avg_latency_ms": round(m.total_latency_ms / total, 2),
                "hallucination_rate": round(m.hallucinations_detected / successful, 4),
                "avg_relevance_score": round(m.total_relevance_score / successful, 4),
                "avg_toxicity_score": round(m.total_toxicity_score / successful, 4),
                "model_breakdown": dict(m.model_request_counts),
            }
