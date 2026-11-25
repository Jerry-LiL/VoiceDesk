import os

import httpx
from openai import OpenAI

from app.prompts import INTENT_INSTRUCTIONS, static_fallback


class MockChat:
    """Reads the system prompt and answers in the shape that template asked for.

    Useful when no OpenAI key and no Ollama server are available. The branch
    is the instruction text, so a fallback template cannot produce a normal
    answer.
    """

    def complete(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"] if messages else ""
        user = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user = " ".join(str(message.get("content", "")).split())
                break
        if INTENT_INSTRUCTIONS["unknown"] in system:
            return "I didn't quite catch that. What would you like help with?"
        if INTENT_INSTRUCTIONS["farewell"] in system:
            return "Goodbye. I'll be here when you want to talk again."
        if INTENT_INSTRUCTIONS["greeting"] in system:
            return "Hi, I'm VoiceDesk. What can I help you with?"
        if INTENT_INSTRUCTIONS["task"] in system:
            return f"Okay. I'll treat that as a request: {user}."
        if INTENT_INSTRUCTIONS["smalltalk"] in system:
            return "I'm doing well. What can I help you with?"
        if INTENT_INSTRUCTIONS["question"] in system:
            return f"You asked, {user}"
        return static_fallback("unknown")


class OpenAIChat:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "",
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise RuntimeError("OpenAI backend selected but no API key is set")
        kwargs: dict = {"api_key": api_key, "timeout": timeout}
        if base_url.strip():
            kwargs["base_url"] = base_url.strip()
        self._client = OpenAI(**kwargs)
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.4,
            max_tokens=180,
        )
        content = response.choices[0].message.content
        return (content or "").strip()


class OllamaChat:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
        )

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.post(
            "/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.4},
            },
        )
        response.raise_for_status()
        body = response.json()
        content = (body.get("message") or {}).get("content") or ""
        return str(content).strip()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def resolve_openai_key(explicit: str) -> str:
    return explicit or os.environ.get("OPENAI_API_KEY", "")
