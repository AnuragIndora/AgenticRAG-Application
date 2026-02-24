import logging
import requests

import settings
from exceptions import OllamaTimeoutError


class OllamaClient:
    """HTTP client for the local Ollama server — handles both embeddings and text generation."""

    def __init__(self) -> None:
        self.settings = settings.get_settings()
        self.base_url = self.settings.ollama_base_url.rstrip("/")
        self.timeout = self.settings.request_timeout_seconds
        self.logger = logging.getLogger(self.__class__.__name__)

    def embed(self, text: str) -> list[float]:
        """Send text to the Ollama embeddings API and return the float vector."""
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.settings.embedding_model, "prompt": text}

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.Timeout as exc:
            raise OllamaTimeoutError("Embedding request timed out") from exc
        except requests.RequestException as exc:
            raise OllamaTimeoutError(f"Embedding request failed: {exc}") from exc

        data = response.json()
        embeddings = data.get("embedding")
        if not embeddings:
            raise OllamaTimeoutError("Ollama embedding API returned empty embedding")
        return embeddings

    def generate(self, messages: list[dict], temperature: float = 0.5) -> str:
        """
        Send a chat message list to Ollama and return the assistant's reply.
        `stream=False` ensures we get the full response in one shot.
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.Timeout as exc:
            raise OllamaTimeoutError("Generate request timed out") from exc
        except requests.RequestException as exc:
            raise OllamaTimeoutError(f"Generate request failed: {exc}") from exc

        data = response.json()
        content = data.get("message", {}).get("content", "")
        if not content:
            raise OllamaTimeoutError("Ollama chat API returned empty response")
        return content.strip()
