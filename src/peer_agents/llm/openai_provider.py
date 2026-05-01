from __future__ import annotations

import json
import os
from typing import Any, Final, override

from .base import LLMProvider, LLMResponse, Message, ToolCall, ToolDefinition

DEFAULT_MODEL: Final[str] = "gpt-4o"
DEFAULT_MAX_TOKENS: Final[int] = 4_096


class OpenAIProvider(LLMProvider):
    """LLM provider backed by the OpenAI Chat Completions API.

    Args:
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        model: Model ID. Defaults to gpt-4o.
        max_tokens: Default output token budget per call.
    """

    client: Any  # openai.AsyncOpenAI — loaded lazily at construction time
    model: str
    max_tokens: int

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI support requires the openai package. "
                "Install it with: pip install 'peer-agents[openai]'"
            )
        self.client = openai.AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens

    @override
    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        api_messages: list[dict[str, Any]] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        for m in messages:
            content: str = m.content if isinstance(m.content, str) else json.dumps(m.content)
            api_messages.append({"role": m.role, "content": content})

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": api_messages,
        }

        if tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]

        response = await self.client.chat.completions.create(**params)
        choice = response.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, input=json.loads(tc.function.arguments))
                )

        finish_reason = choice.finish_reason
        stop: str = "end_turn" if finish_reason in ("stop", None) else (finish_reason or "end_turn")

        usage: dict[str, int] = {}
        if response.usage is not None:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            stop_reason=stop,
            usage=usage,
        )
