from typing import Protocol

import httpx

from app.config import settings


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Generate an answer from prepared prompts."""


class OpenAICompatibleLLMProvider:
    name = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            },
            timeout=self.timeout_seconds,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"LLM request failed: {response.text}") from exc

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM response did not contain choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM response was empty")
        return content


def create_llm_provider() -> LLMProvider:
    if not settings.llm_api_key:
        raise ValueError("SELF_RAG_LLM_API_KEY is not configured.")
    return OpenAICompatibleLLMProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
