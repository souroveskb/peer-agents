from __future__ import annotations

import os
from typing import Any, Final, override

from .base import LLMProvider, LLMResponse, Message, ToolDefinition

DEFAULT_MODEL: Final[str] = "gemini-2.0-flash"
DEFAULT_MAX_TOKENS: Final[int] = 8_192


class GeminiProvider(LLMProvider):
    """LLM provider backed by the Google Gemini API (google-generativeai).

    Args:
        api_key: Google API key. Falls back to GOOGLE_API_KEY env var.
        model: Gemini model ID. Defaults to gemini-2.0-flash.
        max_tokens: Default output token budget per call.
    """

    _genai: Any  # google.generativeai module — loaded lazily at construction time
    model_name: str
    max_tokens: int

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "Gemini support requires google-generativeai. "
                "Install it with: pip install 'peer-agents[google]'"
            )
        genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self._genai = genai
        self.model_name = model
        self.max_tokens = max_tokens

    @override
    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        model = self._genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system,
        )

        # Gemini uses role "model" instead of "assistant" and requires
        # alternating user/model turns. Build history from all but the last.
        history: list[dict[str, Any]] = []
        for m in messages[:-1]:
            role: str = "user" if m.role == "user" else "model"
            content: str = m.content if isinstance(m.content, str) else str(m.content)
            history.append({"role": role, "parts": [content]})

        chat = model.start_chat(history=history)
        last = messages[-1].content
        last_str: str = last if isinstance(last, str) else str(last)

        response = await chat.send_message_async(
            last_str,
            generation_config=self._genai.types.GenerationConfig(
                max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
            ),
        )

        return LLMResponse(content=response.text or "", stop_reason="end_turn")
