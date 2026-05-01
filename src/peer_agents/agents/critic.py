from __future__ import annotations

from pathlib import Path

from peer_agents.llm.base import LLMProvider

from .base import GenerationAgent, _resolve_prompt


class Critic(GenerationAgent):
    """An agent that reviews content and provides constructive feedback.

    Conversation history accumulates so the LLM can see how the content has
    evolved and whether previous issues were addressed.

    Args:
        llm: LLM provider instance, or a model-name string such as
            ``"claude-opus-4-7"``, ``"gpt-4o"``, or ``"gemini-2.0-flash"``.
            The provider is inferred automatically from the model name prefix.
        system_prompt: Prompt string, path to a .txt file, or None to use the
            bundled default (``prompts/critic.txt``).
        name: Display name for this agent.
        context_files: Optional files (PDF, DOCX, PPTX, TXT, MD) whose text is
            injected into the system prompt as context.
    """

    def __init__(
        self,
        llm: LLMProvider | str,
        system_prompt: str | Path | None = None,
        name: str = "Critic",
        context_files: list[Path | str] | None = None,
    ) -> None:
        super().__init__(
            llm=llm,
            system_prompt=_resolve_prompt(system_prompt, "critic.txt"),
            name=name,
            context_files=context_files,
        )

    async def review(self, content: str) -> str:
        """Review the provided content and return feedback."""
        criticism_num: int = len(self.outputs) + 1
        return await self.generate(
            f"Please review the following content:\n\n{content}",
            history_label=f"Criticism #{criticism_num}:",
        )
