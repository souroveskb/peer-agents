from __future__ import annotations

import importlib.resources as pkg_resources
from pathlib import Path

from peer_agents.llm.base import LLMProvider, Message


def _resolve_llm(llm: LLMProvider | str) -> LLMProvider:
    """Accept a provider instance or a bare model-name string.

    Model name routing:
      claude-*          → AnthropicProvider
      gpt-* / o1* / o3* / o4* / chatgpt-* → OpenAIProvider
      gemini-*          → GeminiProvider
    """
    if isinstance(llm, LLMProvider):
        return llm
    if not isinstance(llm, str):
        raise TypeError(
            f"llm must be an LLMProvider or a model-name string, got {type(llm).__name__}"
        )
    model = llm
    if model.startswith("claude"):
        from peer_agents.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model)
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        from peer_agents.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(model=model)
    if model.startswith("gemini"):
        from peer_agents.llm.gemini_provider import GeminiProvider
        return GeminiProvider(model=model)
    raise ValueError(
        f"Cannot infer provider from model name '{model}'. "
        "Pass an LLMProvider instance, or use a model name that starts with "
        "'claude', 'gpt-', 'o1', 'o3', 'o4', or 'gemini'."
    )


def _load_prompt(filename: str) -> str:
    """Load a default prompt from the bundled prompts package."""
    return (
        pkg_resources.files("peer_agents.prompts")
        .joinpath(filename)
        .read_text(encoding="utf-8")
    )


def _resolve_prompt(system_prompt: str | Path | None, default_file: str) -> str:
    """Resolve a prompt from a string, file path, or bundled default."""
    if system_prompt is None:
        return _load_prompt(default_file)
    if isinstance(system_prompt, Path):
        return system_prompt.read_text(encoding="utf-8")
    return system_prompt


class GenerationAgent:
    """Base class for agents that maintain a persistent conversation memory.

    Each call to `generate()` appends the user turn and the assistant reply to
    the internal history. On the next call the full history is forwarded to the
    LLM, giving it complete context of how the work has evolved across iterations.

    Args:
        llm: LLM provider instance or model-name string.
        system_prompt: Resolved system prompt string (already loaded by subclass).
        name: Human-readable name for this agent.
        context_files: Optional list of file paths (PDF, DOCX, PPTX, TXT, MD)
            whose text is extracted and appended to the system prompt so the
            agent has document context on every call.
    """

    llm: LLMProvider
    name: str
    _system_prompt: str
    _history: list[Message]
    outputs: list[str]

    def __init__(
        self,
        llm: LLMProvider | str,
        system_prompt: str,
        name: str,
        context_files: list[Path | str] | None = None,
    ) -> None:
        self.llm = _resolve_llm(llm)
        self.name = name
        self._system_prompt = system_prompt
        if context_files:
            from peer_agents.utils.file_context import load_context_files
            context_text = load_context_files(context_files)
            self._system_prompt = (
                f"{system_prompt}\n\n---\n\nContext from provided files:\n\n{context_text}"
            )
        self._history = []
        self.outputs = []

    async def generate(self, user_message: str, history_label: str | None = None) -> str:
        """Append a user turn, generate a reply, and update history."""
        self._history.append(Message(role="user", content=user_message))
        response = await self.llm.complete(
            messages=self._history,
            system=self._system_prompt,
        )
        labeled: str = (
            f"{history_label}\n\n{response.content}" if history_label else response.content
        )
        self._history.append(Message(role="assistant", content=labeled))
        self.outputs.append(response.content)
        return response.content

    def reset(self) -> None:
        """Clear conversation history and output log."""
        self._history = []
        self.outputs = []

    @property
    def history(self) -> list[Message]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)
