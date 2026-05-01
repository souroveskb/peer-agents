from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, ConfigDict, computed_field

from peer_agents.llm.base import LLMProvider, Message

from .author import Author
from .base import _resolve_llm, _resolve_prompt
from .critic import Critic


class CriticResult(BaseModel):
    """Output from a single critic in one iteration."""

    model_config = ConfigDict(frozen=True)

    name: str
    output: str
    has_criticism: bool


class IterationRecord(BaseModel):
    """Record of a single author-critic cycle."""

    model_config = ConfigDict(frozen=True)

    iteration: int
    author_output: str
    critic_results: list[CriticResult]

    @computed_field
    @property
    def has_criticism(self) -> bool:
        """True if any critic found actionable issues."""
        return any(r.has_criticism for r in self.critic_results)

    @computed_field
    @property
    def critic_output(self) -> str:
        """Combined critic feedback. Single-critic: raw output. Multi-critic: labeled sections."""
        if len(self.critic_results) == 1:
            return self.critic_results[0].output
        return "\n\n".join(f"[{r.name}]\n{r.output}" for r in self.critic_results)


class Orchestrator:
    """Controls the author-critic refinement loop.

    Flow per iteration:
    1. Author writes (first iteration) or revises (subsequent iterations).
    2. All critics review the author's output in parallel.
    3. Each critic's output is independently checked for actionable issues.
    4. If no critic found issues → converged, stop.
    5. If issues remain and iterations remain → pass combined feedback to Author, repeat.

    Both Author and Critic(s) maintain their own conversation histories across
    iterations. The Orchestrator keeps a structured ``memory`` of every cycle.

    Args:
        llm: LLM provider instance, or a model-name string such as
            ``"claude-opus-4-7"``, ``"gpt-4o"``, or ``"gemini-2.0-flash"``.
            Used for the YES/NO has-criticism judgement call.
        author: Author agent instance.
        critic: A single Critic, or a list of Critics for multi-perspective review.
        system_prompt: Prompt string, path to a .txt file, or None to use the
            bundled default (``prompts/orchestrator.txt``).
        max_iterations: Maximum number of author-critic cycles before stopping.
        name: Display name for this orchestrator.
    """

    llm: LLMProvider
    author: Author
    critics: list[Critic]
    max_iterations: int
    name: str
    _system_prompt: str
    memory: list[IterationRecord]
    converged: bool

    def __init__(
        self,
        llm: LLMProvider | str,
        author: Author,
        critic: Critic | list[Critic],
        system_prompt: str | Path | None = None,
        max_iterations: int = 5,
        name: str = "Orchestrator",
    ) -> None:
        self.llm = _resolve_llm(llm)
        self.author = author
        self.critics = [critic] if isinstance(critic, Critic) else list(critic)
        self.max_iterations = max_iterations
        self.name = name
        self._system_prompt = _resolve_prompt(system_prompt, "orchestrator.txt")
        self.memory = []
        self.converged = False

    async def run(self, topic: str) -> str:
        """Run the author-critic loop and return the final author output.

        Resets all agents and the orchestrator's memory at the start of each
        call, so successive ``run()`` invocations are independent.
        """
        self.memory = []
        self.converged = False
        self.author.reset()
        for critic in self.critics:
            critic.reset()

        author_output: str = await self.author.write(topic)

        for i in range(1, self.max_iterations + 1):
            # All critics review in parallel.
            critic_results: list[CriticResult] = list(
                await asyncio.gather(*[
                    self._review_and_check(c, author_output) for c in self.critics
                ])
            )

            self.memory.append(
                IterationRecord(
                    iteration=i,
                    author_output=author_output,
                    critic_results=critic_results,
                )
            )

            if not any(r.has_criticism for r in critic_results):
                self.converged = True
                break

            if i < self.max_iterations:
                # Only pass feedback from critics who actually found issues.
                actionable: list[CriticResult] = [r for r in critic_results if r.has_criticism]
                combined: str = (
                    actionable[0].output
                    if len(actionable) == 1
                    else "\n\n".join(f"[{r.name}]\n{r.output}" for r in actionable)
                )
                author_output = await self.author.revise(combined)

        return author_output

    async def _review_and_check(self, critic: Critic, author_output: str) -> CriticResult:
        output: str = await critic.review(author_output)
        has_issues: bool = await self._has_criticism(output)
        return CriticResult(name=critic.name, output=output, has_criticism=has_issues)

    async def _has_criticism(self, critic_output: str) -> bool:
        """Ask the Orchestrator's LLM whether the critic raised actionable issues."""
        response = await self.llm.complete(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Does this criticism contain actionable issues the author should address? "
                        "Reply with only YES or NO.\n\n"
                        f"Criticism:\n{critic_output}"
                    ),
                )
            ],
            system=self._system_prompt,
        )
        return response.content.strip().upper().startswith("YES")

    @property
    def total_iterations(self) -> int:
        """Number of completed author-critic cycles."""
        return len(self.memory)
