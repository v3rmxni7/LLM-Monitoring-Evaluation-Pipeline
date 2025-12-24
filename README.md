# LLM Monitoring & Evaluation Pipeline (MLOps)

A **production-grade LLM monitoring and evaluation system** built to detect hallucinations, relevance degradation, and reliability issues in Large Language Model (LLM outputs).
The system exposes a FastAPI service for LLM inference, evaluates responses using embedding-based metrics and heuristics, and tracks experiments using MLflow to enable regression analysis across model and prompt iterations.

---

## 🚀 Key Features

* **LLM Inference API**

  * FastAPI-based REST service
  * Pluggable LLM backend (open-source Hugging Face models)
  * Schema-validated request and response contracts

* **Automated LLM Evaluation**

  * Embedding-based relevance scoring (cosine similarity)
  * Heuristic-based hallucination detection
  * Structured evaluation results returned with each response

* **MLOps & Monitoring**

  * MLflow experiment tracking (file-based, no external infrastructure)
  * Logging of prompts, models, metrics, and raw outputs
  * Regression analysis across prompt and model iterations

* **Production-Oriented Design**

  * Modular architecture
  * Clear separation of concerns
  * Easily extensible for new metrics, models, or evaluators

---

## 🧠 Motivation

In real-world GenAI systems, **LLM outputs cannot be blindly trusted**.
This project focuses on **detecting failures**, not hiding them.

Instead of optimizing for “perfect answers”, it:

* Surfaces hallucinations
* Measures semantic relevance
* Tracks quality drift over time

This mirrors how **production GenAI and MLOps teams** monitor LLM reliability.

---

## 🏗️ Architecture Overview

```
Client (Swagger / API)
        |
        v
FastAPI (/generate)
        |
        v
LLM Client (Local Hugging Face Model)
        |
        v
Raw LLM Output
        |
        v
Evaluation Pipeline
   ├── Relevance Scoring (Embeddings)
   ├── Hallucination Detection (Heuristics)
        |
        v
EvaluationResult
        |
        v
MLflow Tracking (Metrics + Artifacts)
```

---

## 📂 Project Structure

```
llm-monitoring-evaluation-pipeline/
│
├── app/
│   ├── api/
│   │   └── routes.py            # FastAPI endpoints
│   ├── core/
│   │   ├── config.py            # Configuration handling
│   │   └── logging.py           # Logging setup
│   ├── llm/
│   │   └── client.py            # LLM backend abstraction
│   ├── schemas/
│   │   ├── request.py           # Request schemas
│   │   └── response.py          # Response schemas
│   ├── evaluation/
│   │   ├── evaluator.py         # Evaluation orchestrator
│   │   ├── relevance.py         # Relevance scoring
│   │   ├── hallucination.py     # Hallucination detection
│   │   └── result.py            # Evaluation result schema
│   ├── monitoring/
│   │   └── mlflow_tracker.py    # MLflow logging
│   └── main.py                  # Application entrypoint
│
├── mlruns/                       # MLflow runs (local)
├── tests/
├── requirements.txt
├── README.md
└── .env.example
```

---

## 🤖 LLM Used

* **Model:** `distilgpt2`
* **Framework:** Hugging Face `transformers`
* **Inference:** Local CPU (no API keys, no paid services)

### Why `distilgpt2`?

* Small and fast
* Not instruction-tuned
* High hallucination tendency

This makes it **ideal for testing monitoring and evaluation logic**, rather than hiding failures with a strong model.

---

## 📊 Evaluation Metrics

### 1️⃣ Relevance Score

* Computed using sentence embeddings
* Cosine similarity between prompt and generated output
* Range: `[0, 1]`
* Lower score indicates semantic drift or irrelevance

### 2️⃣ Hallucination Detection

Heuristic-based detection including:

* Prompt echoing
* Low relevance score
* Suspicious metadata hallucinations (e.g., fabricated articles, dates)

### 3️⃣ Schema Validity

* Ensures responses remain structurally valid and API-safe

---

## 📈 MLflow Experiment Tracking

Each `/generate` request logs:

* **Parameters**

  * Prompt
  * Model name
* **Metrics**

  * Relevance score
  * Hallucination flag
* **Artifacts**

  * Raw LLM output

This enables:

* Prompt regression testing
* Model comparison
* Drift and failure trend analysis

---

## 🔌 API Endpoints

### Health Check

```
GET /health
```

### Generate & Evaluate

```
POST /generate
```

#### Request

```json
{
  "prompt": "Explain LLM hallucination in one sentence."
}
```

#### Response

```json
{
  "raw_output": "...",
  "evaluation": {
    "schema_valid": true,
    "relevance_score": 0.61,
    "hallucination_flag": true,
    "notes": "auto-evaluated"
  }
}
```

Swagger UI is available at:

```
/docs
```

---

## ▶️ Running the Project

### 1️⃣ Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Start FastAPI server

```bash
python -m uvicorn app.main:app --reload
```

### 4️⃣ Start MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

---

## 🔁 Model & Prompt Regression Testing

Because the LLM client is abstracted:

* Models can be swapped with minimal changes
* The same prompts can be re-evaluated
* MLflow metrics enable detection of quality regressions

This mirrors **real-world LLM validation workflows**.

---

## 📌 Future Enhancements

* JSON-only LLM outputs with strict schema enforcement
* LLM-as-a-judge hallucination scoring
* Drift alerts and dashboards
* Dockerized deployment
* CI-based evaluation regression tests


