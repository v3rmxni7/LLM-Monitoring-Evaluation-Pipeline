import mlflow
from app.core.logging import setup_logging

logger = setup_logging()

class MLflowTracker:
    def __init__(self, experiment_name: str = "llm-monitoring"):
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment_name)

    def log_run(
        self,
        prompt: str,
        raw_output: str,
        relevance_score: float,
        hallucination_flag: bool,
        model_name: str
    ):
        with mlflow.start_run():
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("prompt", prompt)

            mlflow.log_metric("relevance_score", relevance_score)
            mlflow.log_metric(
                "hallucination_flag",
                int(hallucination_flag)
            )

            mlflow.log_text(raw_output, "raw_output.txt")

            logger.info("Logged run to MLflow")
