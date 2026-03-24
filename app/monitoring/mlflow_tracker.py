import mlflow
from loguru import logger


class MLflowTracker:
    def __init__(
        self,
        tracking_uri: str = "file:./mlruns",
        experiment_name: str = "llm-monitoring",
    ):
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._initialized = False

    def _init(self):
        if self._initialized:
            return
        try:
            mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(self._experiment_name)
            self._initialized = True
            logger.info(f"MLflow initialized | uri={self._tracking_uri} experiment={self._experiment_name}")
        except Exception as e:
            logger.error(f"MLflow initialization failed: {e}")
            raise

    def log_run(
        self,
        prompt: str,
        raw_output: str,
        model_name: str,
        relevance_score: float,
        hallucination_flag: bool,
        toxicity_score: float = 0.0,
        confidence_score: float = 0.0,
        latency_ms: float = 0.0,
        token_count: int = 0,
        request_id: str = "",
    ):
        self._init()
        try:
            with mlflow.start_run():
                mlflow.set_tag("request_id", request_id)
                mlflow.set_tag("model_name", model_name)

                mlflow.log_param("prompt", prompt[:250])
                mlflow.log_param("model", model_name)

                mlflow.log_metric("relevance_score", relevance_score)
                mlflow.log_metric("hallucination_flag", int(hallucination_flag))
                mlflow.log_metric("toxicity_score", toxicity_score)
                mlflow.log_metric("confidence_score", confidence_score)
                mlflow.log_metric("latency_ms", latency_ms)
                mlflow.log_metric("token_count", token_count)

                mlflow.log_text(raw_output, "raw_output.txt")
                mlflow.log_text(prompt, "prompt.txt")

                logger.info(f"MLflow run logged | request_id={request_id}")
        except Exception as e:
            logger.error(f"MLflow logging failed: {e}")
