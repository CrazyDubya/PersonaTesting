"""
Model API abstraction supporting OpenAI, Anthropic, and OpenRouter providers.

Environment variables:
- OPENAI_API_KEY: API key for OpenAI
- ANTHROPIC_API_KEY: API key for Anthropic
- OPENROUTER_API_KEY: API key for OpenRouter
"""

import os
import time
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from .data_models import ModelConfig


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate a response from the model."""
        pass


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API (GPT-4o, GPT-4o-mini, o3-mini, etc.)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call OpenAI Chat Completions API."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            # Log error and return empty string to avoid crashing
            print(f"OpenAI API error: {e}")
            return ""


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic API (Claude models)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call Anthropic Messages API."""
        # Extract system message and user messages
        system_content = ""
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        try:
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=max_tokens,
                system=system_content,
                messages=user_messages,
                temperature=temperature,
            )

            # Extract text from content blocks
            text_parts = []
            for block in response.content:
                if hasattr(block, 'text'):
                    text_parts.append(block.text)
            return "".join(text_parts)
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return ""


class OpenRouterClient(BaseLLMClient):
    """Client for OpenRouter API (supports many models via unified API)."""

    OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required for OpenRouter. Install with: pip install openai")

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        api_base = config.api_base or self.OPENROUTER_API_BASE

        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call OpenRouter API (OpenAI-compatible)."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers={
                    "HTTP-Referer": "https://github.com/persona-eval",
                    "X-Title": "Persona Evaluation Framework",
                }
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            return ""


class LLMClient:
    """
    Unified LLM client that delegates to provider-specific implementations.
    Supports OpenAI, Anthropic, and OpenRouter.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._client = self._create_client(config)

    def _create_client(self, config: ModelConfig) -> BaseLLMClient:
        """Create the appropriate client based on provider."""
        provider = config.provider.lower()

        if provider == "openai":
            return OpenAIClient(config)
        elif provider == "anthropic":
            return AnthropicClient(config)
        elif provider == "openrouter":
            return OpenRouterClient(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ) -> str:
        """
        Generate a response with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            retry_count: Number of retries on failure.
            retry_delay: Initial delay between retries (exponential backoff).

        Returns:
            Generated text response.
        """
        last_error = None

        for attempt in range(retry_count):
            try:
                return self._client.generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                last_error = e
                if attempt < retry_count - 1:
                    delay = retry_delay * (2 ** attempt)
                    print(f"Retry {attempt + 1}/{retry_count} after {delay}s: {e}")
                    time.sleep(delay)

        print(f"All retries failed: {last_error}")
        return ""


def build_clients(models: List[ModelConfig]) -> Dict[str, LLMClient]:
    """
    Build LLM clients for all configured models.

    Args:
        models: List of model configurations.

    Returns:
        Dictionary mapping model ID to LLMClient instance.
    """
    clients: Dict[str, LLMClient] = {}
    for m in models:
        try:
            clients[m.id] = LLMClient(m)
        except (ImportError, ValueError) as e:
            print(f"Warning: Could not create client for {m.id}: {e}")
    return clients


def test_client(client: LLMClient) -> bool:
    """
    Test if a client is working by sending a simple query.

    Args:
        client: The LLMClient to test.

    Returns:
        True if the client responds successfully.
    """
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'hello' and nothing else."},
    ]
    try:
        response = client.generate(
            messages=test_messages,
            temperature=0.0,
            max_tokens=10,
        )
        return len(response) > 0
    except Exception:
        return False
