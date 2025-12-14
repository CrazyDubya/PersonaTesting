import os
from typing import List, Dict, Any
from openai import OpenAI
from anthropic import Anthropic
from .data_models import ModelConfig


class LLMClient:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.provider = config.provider.lower()

        if self.provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        elif self.provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            self.client = Anthropic(api_key=api_key)
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Call the underlying LLM API and return the assistant's text content.
        """
        if self.provider == "anthropic":
            system_messages = [m["content"] for m in messages if m.get("role") == "system"]
            system_prompt = "\n\n".join(system_messages) if system_messages else None
            chat_messages = [m for m in messages if m.get("role") != "system"]

            response = self.client.messages.create(
                model=self.config.model_name,
                system=system_prompt,
                messages=chat_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content_blocks = response.content
            text_parts = []
            for block in content_blocks:
                if block.type == "text":
                    text_parts.append(block.text)
            return "".join(text_parts)

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


def build_clients(models: List[ModelConfig]) -> Dict[str, LLMClient]:
    clients: Dict[str, LLMClient] = {}
    for m in models:
        clients[m.id] = LLMClient(m)
    return clients
