from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    id: str
    name: str
    input: dict[str, Any]


class Message(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class LLMResponse(BaseModel):
    content: str
    tool_calls: list[ToolCall] = []
    # Full content blocks for accurate conversation reconstruction (needed for tool-use loops).
    raw_content: list[dict[str, Any]] = []
    usage: dict[str, int] = {}
    stop_reason: str = "end_turn"


class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...
