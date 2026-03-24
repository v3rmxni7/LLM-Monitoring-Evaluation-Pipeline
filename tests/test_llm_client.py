from unittest.mock import MagicMock, patch

import pytest

from app.llm.client import LocalHFClient


class TestLocalHFClient:
    def test_init_does_not_load_model(self):
        client = LocalHFClient(model_name="distilgpt2")
        assert client._generator is None
        assert client._tokenizer is None

    def test_get_model_name(self):
        client = LocalHFClient(model_name="test-model")
        assert client.get_model_name() == "test-model"

    @patch("app.llm.client.pipeline")
    @patch("app.llm.client.AutoTokenizer")
    def test_generate_loads_model_on_first_call(self, mock_tokenizer, mock_pipeline):
        mock_gen = MagicMock()
        mock_gen.return_value = [{"generated_text": "response"}]
        mock_pipeline.return_value = mock_gen

        client = LocalHFClient(model_name="distilgpt2")
        result = client.generate("test prompt")

        assert result == "response"
        mock_pipeline.assert_called_once()

    @patch("app.llm.client.pipeline")
    @patch("app.llm.client.AutoTokenizer")
    def test_generate_reuses_model(self, mock_tokenizer, mock_pipeline):
        mock_gen = MagicMock()
        mock_gen.return_value = [{"generated_text": "response"}]
        mock_pipeline.return_value = mock_gen

        client = LocalHFClient(model_name="distilgpt2")
        client.generate("prompt 1")
        client.generate("prompt 2")

        # Model loaded only once
        mock_pipeline.assert_called_once()

    @patch("app.llm.client.pipeline")
    @patch("app.llm.client.AutoTokenizer")
    def test_generate_failure_raises(self, mock_tokenizer, mock_pipeline):
        mock_pipeline.side_effect = RuntimeError("Model not found")

        client = LocalHFClient(model_name="nonexistent-model")
        with pytest.raises(RuntimeError, match="Model loading failed"):
            client.generate("test")

    @patch("app.llm.client.pipeline")
    @patch("app.llm.client.AutoTokenizer")
    def test_count_tokens(self, mock_tokenizer_cls, mock_pipeline):
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        client = LocalHFClient(model_name="distilgpt2")
        client._generator = MagicMock()  # skip model loading
        client._tokenizer = mock_tokenizer

        count = client.count_tokens("hello world test")
        assert count == 5
