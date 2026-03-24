

class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "environment" in data


class TestGenerateEndpoint:
    def test_generate_success(self, client):
        response = client.post(
            "/api/v1/generate",
            json={"prompt": "Explain LLM hallucination."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "raw_output" in data
        assert "evaluation" in data
        assert "latency_ms" in data
        assert "token_count" in data
        assert data["model_name"] == "distilgpt2"

    def test_generate_empty_prompt_rejected(self, client):
        response = client.post(
            "/api/v1/generate",
            json={"prompt": ""},
        )
        assert response.status_code == 422

    def test_generate_prompt_too_long(self, client):
        response = client.post(
            "/api/v1/generate",
            json={"prompt": "x" * 2001},
        )
        assert response.status_code == 422

    def test_generate_with_overrides(self, client):
        response = client.post(
            "/api/v1/generate",
            json={
                "prompt": "Test prompt",
                "max_length": 50,
                "temperature": 0.5,
            },
        )
        assert response.status_code == 200

    def test_generate_invalid_temperature(self, client):
        response = client.post(
            "/api/v1/generate",
            json={"prompt": "Test", "temperature": 3.0},
        )
        assert response.status_code == 422


class TestBatchEndpoint:
    def test_batch_generate(self, client):
        response = client.post(
            "/api/v1/generate/batch",
            json={"prompts": ["Prompt 1", "Prompt 2"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert data["batch_size"] == 2
        assert "total_latency_ms" in data

    def test_batch_empty_rejected(self, client):
        response = client.post(
            "/api/v1/generate/batch",
            json={"prompts": []},
        )
        assert response.status_code == 422

    def test_batch_too_many_prompts(self, client):
        response = client.post(
            "/api/v1/generate/batch",
            json={"prompts": [f"p{i}" for i in range(11)]},
        )
        assert response.status_code == 422


class TestHistoryEndpoint:
    def test_history_empty(self, client):
        response = client.get("/api/v1/history")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["records"] == []

    def test_history_after_generate(self, client):
        client.post("/api/v1/generate", json={"prompt": "Test"})
        response = client.get("/api/v1/history")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["records"]) == 1

    def test_history_pagination(self, client):
        for i in range(5):
            client.post("/api/v1/generate", json={"prompt": f"Test {i}"})
        response = client.get("/api/v1/history?page=1&page_size=2")
        data = response.json()
        assert data["total"] == 5
        assert len(data["records"]) == 2


class TestAnalyticsEndpoint:
    def test_analytics_empty(self, client):
        response = client.get("/api/v1/analytics")
        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 0

    def test_analytics_with_data(self, client):
        client.post("/api/v1/generate", json={"prompt": "Test"})
        response = client.get("/api/v1/analytics")
        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 1
        assert data["avg_relevance_score"] is not None


class TestMetricsEndpoint:
    def test_prometheus_metrics(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "llm_requests_total" in response.text
