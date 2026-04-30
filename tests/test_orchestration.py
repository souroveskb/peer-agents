import pytest

from peer_agents import (
    Author,
    Critic,
    CriticResult,
    IterationRecord,
    LLMProvider,
    LLMResponse,
    Orchestrator,
)


class MockProvider(LLMProvider):
    """Replays a fixed sequence of responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = iter(responses)

    async def complete(self, messages, system=None, tools=None, **kwargs) -> LLMResponse:
        return next(self._responses)


def _ok() -> LLMResponse:
    return LLMResponse(content="NO", stop_reason="end_turn")


def _criticism() -> LLMResponse:
    return LLMResponse(content="YES", stop_reason="end_turn")


# ---------------------------------------------------------------------------
# Orchestrator loop tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stops_immediately_when_no_criticism():
    author = Author(llm=MockProvider([LLMResponse(content="Draft 1", stop_reason="end_turn")]), system_prompt="Write.")
    critic = Critic(llm=MockProvider([LLMResponse(content="No actionable issues.", stop_reason="end_turn")]), system_prompt="Review.")
    orch = Orchestrator(llm=MockProvider([_ok()]), author=author, critic=critic, system_prompt="Orchestrate.", max_iterations=5)

    result = await orch.run("Topic A")

    assert result == "Draft 1"
    assert orch.total_iterations == 1
    assert orch.converged is True
    assert orch.memory[0].has_criticism is False


@pytest.mark.asyncio
async def test_iterates_on_criticism_then_converges():
    author = Author(
        llm=MockProvider([
            LLMResponse(content="Draft 1", stop_reason="end_turn"),
            LLMResponse(content="Draft 2 (revised)", stop_reason="end_turn"),
        ]),
        system_prompt="Write.",
    )
    critic = Critic(
        llm=MockProvider([
            LLMResponse(content="Needs more examples.", stop_reason="end_turn"),
            LLMResponse(content="No actionable issues.", stop_reason="end_turn"),
        ]),
        system_prompt="Review.",
    )
    orch = Orchestrator(
        llm=MockProvider([_criticism(), _ok()]),
        author=author,
        critic=critic,
        system_prompt="Orchestrate.",
        max_iterations=5,
    )

    result = await orch.run("Topic B")

    assert result == "Draft 2 (revised)"
    assert orch.total_iterations == 2
    assert orch.converged is True
    assert orch.memory[0].has_criticism is True
    assert orch.memory[1].has_criticism is False


@pytest.mark.asyncio
async def test_stops_at_max_iterations_without_convergence():
    # Critic always has issues; max_iterations=2 so the loop runs exactly twice.
    # Iterations: write → review1(YES) → revise → review2(YES) → stop.
    author = Author(
        llm=MockProvider([
            LLMResponse(content="Draft 1", stop_reason="end_turn"),
            LLMResponse(content="Draft 2", stop_reason="end_turn"),
        ]),
        system_prompt="Write.",
    )
    critic = Critic(
        llm=MockProvider([
            LLMResponse(content="Still needs work.", stop_reason="end_turn"),
            LLMResponse(content="Still not good enough.", stop_reason="end_turn"),
        ]),
        system_prompt="Review.",
    )
    orch = Orchestrator(
        llm=MockProvider([_criticism(), _criticism()]),
        author=author,
        critic=critic,
        system_prompt="Orchestrate.",
        max_iterations=2,
    )

    result = await orch.run("Topic C")

    assert result == "Draft 2"
    assert orch.total_iterations == 2
    assert orch.converged is False
    assert all(r.has_criticism for r in orch.memory)


# ---------------------------------------------------------------------------
# Memory accumulation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_author_memory_accumulates_across_iterations():
    author = Author(
        llm=MockProvider([
            LLMResponse(content="Draft 1", stop_reason="end_turn"),
            LLMResponse(content="Draft 2", stop_reason="end_turn"),
        ]),
        system_prompt="Write.",
    )
    critic = Critic(
        llm=MockProvider([
            LLMResponse(content="Needs work.", stop_reason="end_turn"),
            LLMResponse(content="No actionable issues.", stop_reason="end_turn"),
        ]),
        system_prompt="Review.",
    )
    orch = Orchestrator(
        llm=MockProvider([_criticism(), _ok()]),
        author=author,
        critic=critic,
        system_prompt="Orchestrate.",
        max_iterations=5,
    )

    await orch.run("Topic")

    # write(1 turn) + revise(1 turn) = 2 user/assistant pairs = 4 messages
    assert len(author.history) == 4
    assert author.outputs == ["Draft 1", "Draft 2"]
    assert author.history[1].content == "Initial draft:\n\nDraft 1"
    assert author.history[3].content == "Revised draft #1:\n\nDraft 2"

    # review(1 turn) × 2 iterations = 4 messages
    assert len(critic.history) == 4
    assert critic.outputs == ["Needs work.", "No actionable issues."]
    assert critic.history[1].content == "Criticism #1:\n\nNeeds work."
    assert critic.history[3].content == "Criticism #2:\n\nNo actionable issues."


@pytest.mark.asyncio
async def test_orchestrator_resets_agents_on_second_run():
    def _make_author():
        return Author(
            llm=MockProvider([
                LLMResponse(content="Run1 draft", stop_reason="end_turn"),
                LLMResponse(content="Run2 draft", stop_reason="end_turn"),
            ]),
            system_prompt="Write.",
        )

    author = _make_author()
    critic = Critic(
        llm=MockProvider([
            LLMResponse(content="No actionable issues.", stop_reason="end_turn"),
            LLMResponse(content="No actionable issues.", stop_reason="end_turn"),
        ]),
        system_prompt="Review.",
    )
    orch = Orchestrator(
        llm=MockProvider([_ok(), _ok()]),
        author=author,
        critic=critic,
        system_prompt="Orchestrate.",
        max_iterations=3,
    )

    await orch.run("Topic 1")
    assert len(author.outputs) == 1
    assert len(orch.memory) == 1

    await orch.run("Topic 2")
    # After reset, only the second run's output remains.
    assert len(author.outputs) == 1
    assert author.outputs[0] == "Run2 draft"
    assert len(orch.memory) == 1


# ---------------------------------------------------------------------------
# IterationRecord structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_iteration_record_fields():
    author = Author(llm=MockProvider([LLMResponse(content="Content", stop_reason="end_turn")]), system_prompt="Write.")
    critic = Critic(llm=MockProvider([LLMResponse(content="No actionable issues.", stop_reason="end_turn")]), system_prompt="Review.")
    orch = Orchestrator(llm=MockProvider([_ok()]), author=author, critic=critic, system_prompt="Orchestrate.", max_iterations=3)

    await orch.run("Test topic")

    record = orch.memory[0]
    assert isinstance(record, IterationRecord)
    assert record.iteration == 1
    assert record.author_output == "Content"
    assert "No actionable issues" in record.critic_output
    assert record.has_criticism is False
    assert len(record.critic_results) == 1
    assert isinstance(record.critic_results[0], CriticResult)


# ---------------------------------------------------------------------------
# Multiple critics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_critics_all_satisfied_converges():
    author = Author(llm=MockProvider([LLMResponse(content="Draft", stop_reason="end_turn")]), system_prompt="Write.")
    critic_a = Critic(llm=MockProvider([LLMResponse(content="No actionable issues.", stop_reason="end_turn")]), system_prompt="Review A.", name="CriticA")
    critic_b = Critic(llm=MockProvider([LLMResponse(content="No actionable issues.", stop_reason="end_turn")]), system_prompt="Review B.", name="CriticB")
    orch = Orchestrator(
        llm=MockProvider([_ok(), _ok()]),
        author=author,
        critic=[critic_a, critic_b],
        system_prompt="Orchestrate.",
        max_iterations=3,
    )

    result = await orch.run("Topic")

    assert result == "Draft"
    assert orch.converged is True
    assert orch.total_iterations == 1
    assert len(orch.memory[0].critic_results) == 2
    assert orch.memory[0].has_criticism is False


@pytest.mark.asyncio
async def test_multiple_critics_one_objects_causes_revision():
    author = Author(
        llm=MockProvider([
            LLMResponse(content="Draft 1", stop_reason="end_turn"),
            LLMResponse(content="Draft 2", stop_reason="end_turn"),
        ]),
        system_prompt="Write.",
    )
    # CriticA happy on first pass; CriticB objects then is happy.
    critic_a = Critic(
        llm=MockProvider([
            LLMResponse(content="No actionable issues.", stop_reason="end_turn"),
            LLMResponse(content="No actionable issues.", stop_reason="end_turn"),
        ]),
        system_prompt="Review A.",
        name="CriticA",
    )
    critic_b = Critic(
        llm=MockProvider([
            LLMResponse(content="Needs more examples.", stop_reason="end_turn"),
            LLMResponse(content="No actionable issues.", stop_reason="end_turn"),
        ]),
        system_prompt="Review B.",
        name="CriticB",
    )
    # Iteration 1: CriticA=NO, CriticB=YES → revise.
    # Iteration 2: CriticA=NO, CriticB=NO → converged.
    orch = Orchestrator(
        llm=MockProvider([_ok(), _criticism(), _ok(), _ok()]),
        author=author,
        critic=[critic_a, critic_b],
        system_prompt="Orchestrate.",
        max_iterations=5,
    )

    result = await orch.run("Topic")

    assert result == "Draft 2"
    assert orch.converged is True
    assert orch.total_iterations == 2
    assert orch.memory[0].has_criticism is True   # CriticB objected
    assert orch.memory[1].has_criticism is False  # both satisfied
