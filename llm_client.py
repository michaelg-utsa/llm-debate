from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from utils import extract_json


class OpenAIJSONClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate_json(
        self,
        *,
        model: str,
        instructions: str,
        user_input: str,
        max_output_tokens: int = 900,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": user_input,
            "max_output_tokens": max_output_tokens,
        }
        if temperature is not None:
            request["temperature"] = temperature

        response = self.client.responses.create(**request)
        text = response.output_text
        return extract_json(text)
