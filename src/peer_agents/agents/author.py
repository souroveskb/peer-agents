from __future__ import annotations

from pathlib import Path

from peer_agents.llm.base import LLMProvider

from .base import GenerationAgent, _resolve_prompt


class Author(GenerationAgent):
    """An agent that generates and iteratively refines content.

    Conversation history accumulates across calls so the LLM always has the
    full context of what was written and what feedback was received.

    Args:
        llm: LLM provider instance, or a model-name string such as
            ``"claude-opus-4-7"``, ``"gpt-4o"``, or ``"gemini-2.0-flash"``.
            The provider is inferred automatically from the model name prefix.
        system_prompt: Prompt string, path to a .txt file, or None to use the
            bundled default (``prompts/author.txt``).
        name: Display name for this agent.
        context_files: Optional files (PDF, DOCX, PPTX, TXT, MD) whose text is
            injected into the system prompt as context.
    """

    def __init__(
        self,
        llm: LLMProvider | str,
        system_prompt: str | Path | None = None,
        name: str = "Author",
        context_files: list[Path | str] | None = None,
    ) -> None:
        super().__init__(
            llm=llm,
            system_prompt=_resolve_prompt(system_prompt, "author.txt"),
            name=name,
            context_files=context_files,
        )

    async def write(self, topic: str) -> str:
        """Generate initial content for a topic."""
        return await self.generate(topic, history_label="Initial draft:")

    async def revise(self, criticism: str) -> str:
        """Revise previous content based on critic feedback.

        The revision request is appended to the existing conversation history,
        so the LLM sees the original content alongside the new instructions.
        """
        revision_num = len(self.outputs)
        return await self.generate(
            f"Please revise your content based on this feedback:\n\n{criticism}",
            history_label=f"Revised draft #{revision_num}:",
        )
