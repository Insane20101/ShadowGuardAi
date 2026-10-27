
import hmac
import math
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.security.sanitizer import Sanitizer


load_dotenv()


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

def load_positive_int(name: str, default: str) -> int:
    try:
        value = int(os.getenv(name, default))
    except (TypeError, ValueError):
        raise RuntimeError(
            f"{name} must be a valid integer."
        ) from None

    if value <= 0:
        raise RuntimeError(
            f"{name} must be greater than zero."
        )

    return value


def load_positive_float(name: str, default: str) -> float:
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        raise RuntimeError(
            f"{name} must be a valid number."
        ) from None

    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(
            f"{name} must be a finite number greater than zero."
        )

    return value


def load_required_secret(name: str) -> str:
    value = os.getenv(name, "")

    if not value or not value.strip():
        raise RuntimeError(
            f"{name} must be configured before starting the application."
        )

    return value


MAX_PROMPT_LENGTH = load_positive_int(
    "MAX_PROMPT_LENGTH",
    "12000",
)

MAX_REQUEST_BODY_BYTES = load_positive_int(
    "MAX_REQUEST_BODY_BYTES",
    "65536",
)

LLM_TIMEOUT_SECONDS = load_positive_float(
    "LLM_TIMEOUT_SECONDS",
    "30",
)

SANITIZE_RATE_LIMIT = os.getenv(
    "SANITIZE_RATE_LIMIT",
    "20/minute",
)

CHAT_RATE_LIMIT = os.getenv(
    "CHAT_RATE_LIMIT",
    "10/minute",
)

SHADOWGUARD_API_KEY = load_required_secret(
    "SHADOWGUARD_API_KEY"
)

if len(SHADOWGUARD_API_KEY) < 32:
    raise RuntimeError(
        "SHADOWGUARD_API_KEY must contain at least 32 characters."
    )


# -------------------------------------------------------------------
# Application lifecycle
# -------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.sanitizer = Sanitizer()
    yield


app = FastAPI(
    title="ShadowGuard AI",
    description="PII sanitization gateway for LLM applications",
    version="0.3.0",
    lifespan=lifespan,
)


# -------------------------------------------------------------------
# Rate limiting
# -------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)


# -------------------------------------------------------------------
# Request and response models
# -------------------------------------------------------------------

class PromptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_prompt_length(self):
        if len(self.text) > MAX_PROMPT_LENGTH:
            raise ValueError(
                "Prompt exceeds the maximum allowed length."
            )

        if not self.text.strip():
            raise ValueError(
                "Prompt cannot contain only whitespace."
            )

        return self


class SanitizeResponse(BaseModel):
    sanitized_text: str
    entity_counts: dict[str, int]


class ChatResponse(BaseModel):
    answer: str
    entity_counts: dict[str, int]
    model: str


# -------------------------------------------------------------------
# Authentication
# -------------------------------------------------------------------

def require_api_key(
    x_api_key: Annotated[
        str | None,
        Header(alias="X-API-Key"),
    ] = None,
) -> None:
    if not x_api_key or not hmac.compare_digest(
        x_api_key,
        SHADOWGUARD_API_KEY,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# -------------------------------------------------------------------
# Bounded request-body middleware
# -------------------------------------------------------------------

@app.middleware("http")
async def enforce_request_body_limit(
    request: Request,
    call_next,
):
    # Reject oversized declared bodies before reading them.
    content_length = request.headers.get("content-length")

    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid Content-Length header."},
            )

        if declared_length < 0:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid Content-Length header."},
            )

        if declared_length > MAX_REQUEST_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body is too large."},
            )

    # Wrap the ASGI receive function. Count bytes as the downstream
    # application reads them, without consuming or discarding the body.
    original_receive = request.receive
    total_bytes = 0

    async def limited_receive():
        nonlocal total_bytes

        message = await original_receive()

        if message["type"] == "http.request":
            total_bytes += len(message.get("body", b""))

            if total_bytes > MAX_REQUEST_BODY_BYTES:
                raise RequestBodyTooLarge

        return message

    request._receive = limited_receive

    try:
        return await call_next(request)
    except RequestBodyTooLarge:
        return JSONResponse(
            status_code=413,
            content={"detail": "Request body is too large."},
        )


class RequestBodyTooLarge(Exception):
    """Raised when a streamed request body exceeds the configured limit."""


# -------------------------------------------------------------------
# Sanitization helper
# -------------------------------------------------------------------

def sanitize_prompt(text: str):
    try:
        return app.state.sanitizer.sanitize(text)
    except HTTPException:
        raise
    except Exception:
        # Do not expose raw prompts, entity mappings, or internal errors.
        raise HTTPException(
            status_code=500,
            detail="Text sanitization failed.",
        ) from None


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ShadowGuard AI",
    }


@app.post(
    "/sanitize",
    response_model=SanitizeResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(SANITIZE_RATE_LIMIT)
def sanitize_text(
    request: Request,
    body: PromptRequest,
):
    result = sanitize_prompt(body.text)

    return SanitizeResponse(
        sanitized_text=result.sanitized_text,
        entity_counts=result.entity_counts,
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(CHAT_RATE_LIMIT)
async def chat(
    request: Request,
    body: PromptRequest,
):
    groq_api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b",
    )

    if not groq_api_key:
        raise HTTPException(
            status_code=503,
            detail="LLM provider is not configured.",
        )

    result = sanitize_prompt(body.text)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": result.sanitized_text,
            }
        ],
        "temperature": 0.2,
        "max_completion_tokens": 1024,
    }

    try:
        async with httpx.AsyncClient(
            timeout=LLM_TIMEOUT_SECONDS,
        ) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            response.raise_for_status()
            data: Any = response.json()

        choices = data["choices"]

        if not isinstance(choices, list) or not choices:
            raise ValueError("Missing completion choices.")

        message = choices[0]["message"]
        answer = message["content"]

        if not isinstance(answer, str):
            raise ValueError("Unexpected model response.")

        return ChatResponse(
            answer=answer,
            entity_counts=result.entity_counts,
            model=model,
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="The LLM provider timed out.",
        ) from None

    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="LLM provider rate limit reached. Try again later.",
            ) from None

        raise HTTPException(
            status_code=502,
            detail="The LLM provider request failed.",
        ) from None

    except (KeyError, IndexError, TypeError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="Unexpected response from the LLM provider.",
        ) from None

    except httpx.RequestError:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the LLM provider.",
        ) from None
