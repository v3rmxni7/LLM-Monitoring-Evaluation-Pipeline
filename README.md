# LLM Monitoring & Evaluation Pipeline

A **production-grade LLM monitoring and evaluation system** that detects hallucinations, measures semantic relevance, scores toxicity, and tracks quality drift across model iterations — powered by FastAPI, MLflow, Prometheus metrics, and a SQLite-backed history/analytics layer.

---

## Key Features

### LLM Inference API
- FastAPI REST service with **versioned API** (`/api/v1/`)
- Pluggable LLM backend via abstract base class
- **Single and batch** generation endpoints
- Request/response schema validation with Pydantic v2
- Per-request overrides (model, temperature, max length)

### Multi-Dimensional Evaluation Pipeline
- **Relevance scoring** — Sentence-BERT embeddings + cosine similarity
- **Hallucination detection** — Multi-signal heuristic engine (prompt echoing, low relevance, repetition detection, suspicious patterns, incoherent structure)
- **Toxicity scoring** — Pattern-based toxicity analysis with severity weighting
- **Confidence scoring** — Weighted composite of all evaluation signals
- Every response includes full evaluation metadata

### MLOps & Observability
- **MLflow experiment tracking** — params, metrics, artifacts per run
- **Prometheus-compatible `/metrics` endpoint** for Grafana integration
- **Structured JSON logging** (production) with request ID correlation
- Log rotation with compression

### Production Infrastructure
- **SQLite database** with full request/response history
- **Analytics endpoint** — aggregated quality metrics, model breakdowns
- **Paginated history** with filtering (model, hallucination-only)
- **Dependency injection** via FastAPI `Depends()` — fully testable
- **Rate limiting**, **API key auth**, **CORS**, **request ID** middleware
- **Multi-stage Docker** build with health checks
- **docker-compose** with MLflow server
- **GitHub Actions CI** — lint, test, Docker build verification

---

## Architecture

```
Client Request
      |
      v
Middleware Stack (Request ID → Logging → Rate Limit → API Key → CORS)
      |
      v
FastAPI Router (/api/v1/generate)
      |
      v
LLM Client (HuggingFace Transformers)
      |
      v
Evaluation Pipeline
  ├── RelevanceScorer (sentence-transformers)
  ├── HallucinationDetector (multi-signal heuristics)
  └── ToxicityDetector (pattern-based scoring)
      |
      v
EvaluationResult (relevance, hallucination, toxicity, confidence)
      |
      ├──> MLflow Tracking (params, metrics, artifacts)
      ├──> Prometheus Metrics (counters, gauges)
      ├──> SQLite Database (full history)
      |
      v
JSON Response with evaluation metadata
```

---

## Project Structure

```
├── app/
│   ├── main.py                    # Application entrypoint with lifespan
│   ├── api/
│   │   ├── routes.py              # API endpoints (generate, batch, history, analytics)
│   │   └── middleware.py          # Request ID, logging, rate limit, API key
│   ├── core/
│   │   ├── config.py              # Pydantic settings (all configurable via env)
│   │   ├── logging.py             # Structured logging with JSON + rotation
│   │   └── dependencies.py        # FastAPI dependency injection
│   ├── db/
│   │   ├── database.py            # SQLAlchemy engine and session
│   │   └── models.py              # LLMRequest model
│   ├── evaluation/
│   │   ├── evaluator.py           # Evaluation orchestrator
│   │   ├── relevance.py           # Embedding-based relevance scoring
│   │   ├── hallucination.py       # Multi-signal hallucination detection
│   │   ├── toxicity.py            # Toxicity scoring
│   │   └── result.py              # EvaluationResult schema
│   ├── llm/
│   │   ├── base.py                # Abstract LLM client interface
│   │   └── client.py              # HuggingFace local client
│   ├── monitoring/
│   │   ├── mlflow_tracker.py      # MLflow experiment tracking
│   │   └── metrics.py             # Prometheus metrics collector
│   └── schemas/
│       ├── request.py             # Request validation schemas
│       └── response.py            # Response schemas
├── tests/
│   ├── conftest.py                # Fixtures with dependency overrides
│   ├── test_api.py                # API endpoint tests
│   ├── test_evaluation.py         # Evaluation pipeline tests
│   ├── test_llm_client.py         # LLM client tests
│   └── test_monitoring.py         # Metrics collector tests
├── .github/workflows/ci.yml      # CI pipeline (lint → test → Docker)
├── Dockerfile                     # Multi-stage production build
├── docker-compose.yml             # App + MLflow server
├── pyproject.toml                 # Project config, pytest, ruff
├── requirements.txt               # Pinned dependencies
└── .env.example                   # All configuration variables
```

---

## API Endpoints

### System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check with version info |
| GET | `/metrics` | Prometheus-compatible metrics |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc documentation |

### LLM Operations (`/api/v1/`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/generate` | Generate + evaluate single prompt |
| POST | `/generate/batch` | Batch generate (up to 10 prompts) |
| GET | `/history` | Paginated request history with filters |
| GET | `/analytics` | Aggregated quality metrics |

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain LLM hallucination in one sentence."}'
```

### Example Response

```json
{
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "prompt": "Explain LLM hallucination in one sentence.",
  "raw_output": "...",
  "model_name": "distilgpt2",
  "evaluation": {
    "relevance_score": 0.61,
    "hallucination_flag": true,
    "toxicity_score": 0.0,
    "confidence_score": 0.505,
    "hallucination_reasons": ["Low relevance score (0.210 < 0.3)"],
    "notes": "auto-evaluated"
  },
  "latency_ms": 245.32,
  "token_count": 42
}
```

---

## Running the Project

### Local Development

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Start the server
uvicorn app.main:app --reload

# Run tests
pytest

# Run tests with coverage
pytest --cov=app --cov-report=html
```

### Docker

```bash
# Build and start all services
docker compose up --build

# App:    http://localhost:8000
# MLflow: http://localhost:5000
# Docs:   http://localhost:8000/docs
```

---

## Configuration

All settings are configurable via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `dev` | Environment (dev/prod) |
| `API_KEY` | None | API key for auth (disabled if unset) |
| `RATE_LIMIT_PER_MINUTE` | 60 | Max requests per IP per minute |
| `LLM_MODEL_NAME` | `distilgpt2` | HuggingFace model name |
| `LLM_MAX_LENGTH` | 150 | Max generation length |
| `LLM_TEMPERATURE` | 1.0 | Sampling temperature |
| `LLM_DEVICE` | -1 | Device (-1=CPU, 0+=GPU) |
| `HALLUCINATION_RELEVANCE_THRESHOLD` | 0.3 | Relevance threshold for hallucination |
| `DATABASE_URL` | `sqlite:///./llm_monitor.db` | Database connection string |
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | MLflow tracking URI |

---

## Evaluation Metrics

### Relevance Score (0.0 - 1.0)
Cosine similarity between prompt and output embeddings using `all-MiniLM-L6-v2`. Lower scores indicate semantic drift.

### Hallucination Detection
Multi-signal heuristic engine:
- **Prompt echoing** — output trivially repeats the input
- **Low relevance** — output semantically diverges from prompt
- **Excessive repetition** — low unique word ratio
- **Suspicious patterns** — fabricated URLs, dates, attributions
- **Incoherent structure** — excessively long sentences without punctuation

### Toxicity Score (0.0 - 1.0)
Pattern-based analysis with severity-weighted scoring across multiple toxicity categories.

### Confidence Score (0.0 - 1.0)
Weighted composite: `relevance * 0.5 + no_hallucination * 0.3 + (1 - toxicity) * 0.2`

---

## Why distilgpt2?

This project intentionally uses a weak, non-instruction-tuned model. It **frequently hallucinates**, making it ideal for stress-testing the monitoring and evaluation pipeline. The goal is to detect and measure failures — not hide them behind a strong model.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI, Pydantic v2, Uvicorn |
| LLM | HuggingFace Transformers |
| Embeddings | sentence-transformers (MiniLM) |
| Evaluation | Custom multi-signal pipeline |
| Tracking | MLflow |
| Metrics | Prometheus-compatible |
| Database | SQLAlchemy + SQLite |
| Logging | Loguru (structured JSON) |
| Testing | pytest with dependency injection |
| CI/CD | GitHub Actions |
| Container | Docker (multi-stage) |
