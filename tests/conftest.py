import os
from unittest.mock import MagicMock

import pytest

os.environ["DATABASE_URL"] = "sqlite:////tmp/test_llm_monitor.db"
os.environ["ENV"] = "test"
os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/test_mlruns"

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.dependencies import get_evaluator, get_llm_client, get_metrics, get_tracker
from app.db.database import Base, get_db
from app.evaluation.evaluator import LLMEvaluator
from app.evaluation.result import EvaluationResult
from app.llm.client import LocalHFClient
from app.main import app
from app.monitoring.metrics import MetricsCollector
from app.monitoring.mlflow_tracker import MLflowTracker

TEST_DATABASE_URL = "sqlite:////tmp/test_llm_monitor.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=LocalHFClient)
    client.generate.return_value = "This is a generated response about the topic."
    client.count_tokens.return_value = 10
    client.get_model_name.return_value = "distilgpt2"
    return client


@pytest.fixture
def mock_evaluator():
    evaluator = MagicMock(spec=LLMEvaluator)
    evaluator.evaluate.return_value = EvaluationResult(
        relevance_score=0.75,
        hallucination_flag=False,
        toxicity_score=0.05,
        confidence_score=0.82,
        hallucination_reasons=[],
        notes="auto-evaluated",
    )
    return evaluator


@pytest.fixture
def mock_tracker():
    return MagicMock(spec=MLflowTracker)


@pytest.fixture
def mock_metrics():
    return MetricsCollector()


@pytest.fixture
def client(mock_llm_client, mock_evaluator, mock_tracker, mock_metrics):
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_llm_client] = lambda: mock_llm_client
    app.dependency_overrides[get_evaluator] = lambda: mock_evaluator
    app.dependency_overrides[get_tracker] = lambda: mock_tracker
    app.dependency_overrides[get_metrics] = lambda: mock_metrics

    with TestClient(app) as tc:
        yield tc

    app.dependency_overrides.clear()


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
