
import hmac
import os

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.security.sanitizer import Sanitizer


from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from fastapi import Request

load_dotenv()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="ShadowGuard AI",
    description="PII sanitization gateway for LLM applications",
    version="0.2.0",
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)

sanitizer = Sanitizer()








def load_positive_int(name: str, default: str) -> int:
    try:
        value = int(os.getenv(name, default))
    except ValueError:
        raise RuntimeError(f"{name} must be a valid integer.") from None

    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero.")

    return value


def load_positive_float(name: str, default: str) -> float:
    try:
        value = float(os.getenv(name, default))
    except ValueError:
        raise RuntimeError(f"{name} must be a valid number.") from None

    if not 0 < value < float("inf"):
        raise RuntimeError(f"{name} must be a finite number greater than zero.")

    return value


MAX_PROMPT_LENGTH = load_positive_int("MAX_PROMPT_LENGTH", "12000")
LLM_TIMEOUT_SECONDS = load_positive_float("LLM_TIMEOUT_SECONDS", "30")








class PromptRequest(BaseModel):
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_prompt_length(self):
        if len(self.text) > MAX_PROMPT_LENGTH:
            raise ValueError(
                "Prompt exceeds the maximum allowed length."
            )
        return self




class SanitizeResponse(BaseModel):
    sanitized_text: str
    entity_counts: dict[str, int]


class ChatResponse(BaseModel):
    answer: str
    entity_counts: dict[str, int]
    model: str


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    expected = os.getenv("SHADOWGUARD_API_KEY")

    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Gateway authentication is not configured.",
        )

    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )



def sanitize_prompt(text: str):
    try:
        return sanitizer.sanitize(text)
    except Exception:
        # Don't return prompt contents or internal errors to clients.
        raise HTTPException(
            status_code=500,
            detail="Text sanitization failed.",
        )
       


@app.get("/health")
def health():
    return {"status": "ok", "service": "ShadowGuard AI"}


@app.post(
    "/sanitize",
    response_model=SanitizeResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("20/minute")
def sanitize_text(request: Request, body: PromptRequest):
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
@limiter.limit("10/minute")
async def chat(request: Request, body: PromptRequest):
    groq_api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

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
            timeout=LLM_TIMEOUT_SECONDS
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
            data = response.json()

        answer = data["choices"][0]["message"]["content"]

        if not isinstance(answer, str):
            raise ValueError("Unexpected model response")

        # Mapping is deliberately not returned or restored automatically.
        return ChatResponse(
            answer=answer,
            entity_counts=result.entity_counts,
            model=model,
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="The LLM provider timed out.",
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="LLM provider rate limit reached. Try again later.",
            )
        raise HTTPException(
            status_code=502,
            detail="The LLM provider request failed.",
        )
    except (KeyError, IndexError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="Unexpected response from the LLM provider.",
        )
    except httpx.RequestError:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the LLM provider.",
        )
