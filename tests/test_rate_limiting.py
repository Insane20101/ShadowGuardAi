
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


def test_rate_limit_returns_429():
    test_app = FastAPI()
    test_limiter = Limiter(
        key_func=get_remote_address,
        storage_uri="memory://",
    )

    test_app.state.limiter = test_limiter
    test_app.add_exception_handler(
        RateLimitExceeded,
        _rate_limit_exceeded_handler,
    )

    @test_app.get("/limited")
    @test_limiter.limit("2/minute")
    def limited_endpoint(request: Request):
        return {"status": "ok"}

    with TestClient(test_app) as client:
        assert client.get("/limited").status_code == 200
        assert client.get("/limited").status_code == 200

        response = client.get("/limited")
        assert response.status_code == 429
