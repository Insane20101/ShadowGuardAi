
import httpx
import os

from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

TEST_API_KEY = "test-shadowguard-api-key-123456789"
os.environ["SHADOWGUARD_API_KEY"] = TEST_API_KEY

from app.main import app

import pytest

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

API_HEADERS = {"X-API-Key": TEST_API_KEY}








# --------------------------------------------------
# Shared mock helpers
# --------------------------------------------------

def mock_groq_response(status_code=200, json_data=None):
    response = MagicMock()
    response.raise_for_status.return_value = None

    response.json.return_value = json_data or {
        "choices": [
            {"message": {"content": "Hello! How can I help?"}}
        ]
    }

    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Provider error",
            request=httpx.Request(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
            ),
            response=httpx.Response(status_code),
        )

    return response


def mock_async_client(mock_response=None, side_effect=None):
    """Mock the AsyncClient context manager used by /chat."""
    provider_client = AsyncMock()

    if side_effect is not None:
        provider_client.post.side_effect = side_effect
    else:
        provider_client.post.return_value = mock_response

    context_manager = MagicMock()
    context_manager.__aenter__ = AsyncMock(
        return_value=provider_client
    )
    context_manager.__aexit__ = AsyncMock(
        return_value=False
    )

    return provider_client, context_manager


# --------------------------------------------------
# Health endpoint tests
# --------------------------------------------------

def test_health_is_public(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# --------------------------------------------------
# /sanitize endpoint tests
# --------------------------------------------------

def test_sanitize_rejects_missing_api_key(client):
    response = client.post(
        "/sanitize",
        json={"text": "Contact alice@example.com"},
    )

    assert response.status_code == 401


def test_sanitize_rejects_invalid_api_key(client):
    response = client.post(
        "/sanitize",
        headers={"X-API-Key": "wrong-key"},
        json={"text": "Contact alice@example.com"},
    )

    assert response.status_code == 401


def test_sanitize_masks_email(client):
    response = client.post(
        "/sanitize",
        headers=API_HEADERS,
        json={"text": "Contact alice@example.com"},
    )

    assert response.status_code == 200

    data = response.json()

    assert "alice@example.com" not in data["sanitized_text"]
    assert "[EMAIL_1]" in data["sanitized_text"]
    assert data["entity_counts"]["EMAIL"] == 1
    assert "mapping" not in data


def test_sanitize_rejects_empty_text(client):
    response = client.post(
        "/sanitize",
        headers=API_HEADERS,
        json={"text": ""},
    )

    assert response.status_code == 422


def test_sanitize_rejects_oversized_text(client):
    response = client.post(
        "/sanitize",
        headers=API_HEADERS,
        json={"text": "a" * 12001},
    )

    assert response.status_code == 422


# --------------------------------------------------
# /chat endpoint tests
# --------------------------------------------------

def test_chat_rejects_missing_api_key(client):
    response = client.post(
        "/chat",
        json={"text": "Hello"},
    )

    assert response.status_code == 401


def test_chat_returns_answer_and_sends_sanitized_prompt(
    monkeypatch,client
):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    provider_response = mock_groq_response()

    provider_client, context_manager = mock_async_client(
        mock_response=provider_response
    )

    with patch(
        "app.main.httpx.AsyncClient",
        return_value=context_manager,
    ):
        response = client.post(
            "/chat",
            headers=API_HEADERS,
            json={"text": "Email me at alice@example.com"},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "Hello! How can I help?"
    assert data["entity_counts"]["EMAIL"] == 1
    assert data["model"]

    # Confirm that the original email was not sent to Groq.
    sent_payload = provider_client.post.await_args.kwargs["json"]
    sent_text = sent_payload["messages"][0]["content"]

    assert "alice@example.com" not in sent_text
    assert "[EMAIL_1]" in sent_text

    # The API response must not expose the original-value mapping.
    assert "mapping" not in data


def test_chat_handles_provider_rate_limit(monkeypatch, client):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    provider_response = mock_groq_response(status_code=429)

    _, context_manager = mock_async_client(
        mock_response=provider_response
    )

    with patch(
        "app.main.httpx.AsyncClient",
        return_value=context_manager,
    ):
        response = client.post(
            "/chat",
            headers=API_HEADERS,
            json={"text": "Hello"},
        )

    assert response.status_code == 429
    assert "rate limit" in response.json()["detail"].lower()


def test_chat_handles_provider_timeout(monkeypatch, client):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    _, context_manager = mock_async_client(
        side_effect=httpx.TimeoutException("timed out")
    )

    with patch(
        "app.main.httpx.AsyncClient",
        return_value=context_manager,
    ):
        response = client.post(
            "/chat",
            headers=API_HEADERS,
            json={"text": "Hello"},
        )

    assert response.status_code == 504
    assert "timed out" in response.json()["detail"].lower()





def test_chat_rejects_missing_groq_api_key(monkeypatch, client):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    response = client.post(
        "/chat",
        headers=API_HEADERS,
        json={"text": "Hello"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "LLM provider is not configured."






def test_chat_handles_malformed_provider_response(monkeypatch, client):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    malformed_response = mock_groq_response(
        json_data={"unexpected": "response"}
    )

    _, context_manager = mock_async_client(
        mock_response=malformed_response
    )

    with patch(
        "app.main.httpx.AsyncClient",
        return_value=context_manager,
    ):
        response = client.post(
            "/chat",
            headers=API_HEADERS,
            json={"text": "Hello"},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Unexpected response from the LLM provider."
    )









# adapting the endpoint payload if your existing tests use a different format.






def test_normal_request_body_is_accepted(client):
    response = client.post(
        "/sanitize",
        json={"text": "Hello world"},
        headers=API_HEADERS,
    )

    assert response.status_code == 200


def test_oversized_content_length_is_rejected(client, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "MAX_REQUEST_BODY_BYTES", 10)

    response = client.post(
        "/sanitize",
        content=b'{"text":"Hello world"}',
        headers={
            **API_HEADERS,
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Request body is too large."


def test_invalid_content_length_is_rejected(client):
    response = client.post(
        "/sanitize",
        content=b'{"text":"Hello"}',
        headers={
            **API_HEADERS,
            "Content-Type": "application/json",
            "Content-Length": "invalid",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid Content-Length header."



# Add a streamed-body test




def test_oversized_streamed_body_is_rejected(client, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "MAX_REQUEST_BODY_BYTES", 10)

    oversized_body = b'{"text":"This body is larger than ten bytes"}'

    response = client.post(
        "/sanitize",
        content=oversized_body,
        headers={
            **API_HEADERS,
            "Content-Type": "application/json",
            "Content-Length": str(len(oversized_body)),
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Request body is too large."
