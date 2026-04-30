from __future__ import annotations

from typing import Any

import anthropic

from .base import LLMProvider, LLMResponse, Message, ToolCall, ToolDefinition

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_MAX_TOKENS = 16_000


class AnthropicProvider(LLMProvider):
    """LLM provider backed by the Anthropic API.

    Prompt caching is enabled automatically on the system prompt via
    cache_control, so repeated calls with the same system share a cached prefix.

    Args:
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        model: Model ID. Defaults to claude-opus-4-7.
        max_tokens: Default output token budget per call.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": api_messages,
        }

        if system:
            # Cache the system prompt — stable content, benefits from prefix caching.
            params["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]

        if tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]

        if kwargs.get("thinking"):
            params["thinking"] = {"type": "adaptive"}

        async with self.client.messages.stream(**params) as stream:
            response = await stream.get_final_message()

        text_content = ""
        tool_calls: list[ToolCall] = []
        raw_content: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                text_content = block.text
                raw_content.append({"type": "text", "text": block.text})
            elif block.type == "thinking":
                # Preserve thinking blocks and their signatures for multi-turn continuity.
                raw_content.append(
                    {
                        "type": "thinking",
                        "thinking": block.thinking,
                        "signature": getattr(block, "signature", ""),
                    }
                )
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))
                raw_content.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                )

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls,
            raw_content=raw_content,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
                "cache_creation_input_tokens": getattr(
                    response.usage, "cache_creation_input_tokens", 0
                ),
            },
            stop_reason=response.stop_reason or "end_turn",
        )
