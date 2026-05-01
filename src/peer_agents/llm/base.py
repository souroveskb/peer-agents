from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    input: dict[str, Any]


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    content: str | list[dict[str, Any]]


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Full content blocks for accurate conversation reconstruction (needed for tool-use loops).
    raw_content: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
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
