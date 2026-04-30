# peer-agents — Agent Context

## What this package is

`peer-agents` is an async Python library implementing an author-critic refinement pipeline. An `Author` agent generates content, one or more `Critic` agents review it, and an `Orchestrator` runs the loop — revising until all critics are satisfied or a max iteration limit is hit. Every agent is provider-agnostic; swap between Anthropic, OpenAI, and Gemini by changing one constructor argument.

## Install

```bash
pip install "peer-agents[anthropic] @ git+https://github.com/your-username/peer-agents.git"
pip install "peer-agents[openai] @ git+https://github.com/your-username/peer-agents.git"
pip install "peer-agents[google] @ git+https://github.com/your-username/peer-agents.git"
pip install "peer-agents[files] @ git+https://github.com/your-username/peer-agents.git"  # PDF/DOCX/PPTX context
pip install "peer-agents[all] @ git+https://github.com/your-username/peer-agents.git"
```

## Import paths

```python
# Pipeline agents
from peer_agents import Author, Critic, Orchestrator
from peer_agents import IterationRecord, CriticResult  # data types
from peer_agents import LLMProvider, LLMResponse, Message  # base types

# Providers (import from submodule — not top-level)
from peer_agents.llm.anthropic_provider import AnthropicProvider
from peer_agents.llm.openai_provider import OpenAIProvider
from peer_agents.llm.gemini_provider import GeminiProvider
```

## Providers

Every `llm` parameter accepts either a **model name string** or a **provider instance**.

```python
# String shorthand — provider auto-detected from prefix
Author(llm="claude-opus-4-7")    # → AnthropicProvider
Critic(llm="gpt-4o")             # → OpenAIProvider
Orchestrator(llm="gemini-2.0-flash", ...)  # → GeminiProvider

# Provider instance — full control
from peer_agents.llm.anthropic_provider import AnthropicProvider
Author(llm=AnthropicProvider(model="claude-opus-4-7", max_tokens=8000))
```

Model name routing:

| Prefix | Provider class | Env var | Default model |
|---|---|---|---|
| `claude-*` | `AnthropicProvider` | `ANTHROPIC_API_KEY` | `claude-opus-4-7` |
| `gpt-*` / `o1*` / `o3*` / `o4*` / `chatgpt-*` | `OpenAIProvider` | `OPENAI_API_KEY` | `gpt-4o` |
| `gemini-*` | `GeminiProvider` | `GOOGLE_API_KEY` | `gemini-2.0-flash` |

Provider constructors:
```python
AnthropicProvider(api_key=None, model="claude-opus-4-7", max_tokens=16_000)
OpenAIProvider(api_key=None, model="gpt-4o", max_tokens=4_096)
GeminiProvider(api_key=None, model="gemini-2.0-flash", max_tokens=8_192)
```

Providers can be mixed freely across agents in the same pipeline.

## Author

Generates the initial draft and revises it based on feedback. Maintains full conversation history across calls (this IS its memory).

```python
Author(
    llm: LLMProvider | str,  # model name string or provider instance
    system_prompt: str | Path | None = None,       # None = bundled default
    name: str = "Author",
    context_files: list[Path | str] | None = None, # PDF, DOCX, PPTX, TXT, MD
)
```

Methods:
```python
await author.write(topic: str) -> str        # first draft
await author.revise(criticism: str) -> str   # revision based on feedback
author.reset()                               # clear history (called automatically by Orchestrator)
author.history  -> list[Message]             # full conversation history
author.outputs  -> list[str]                 # raw text outputs (no labels)
```

History labels stored in the assistant turn:
- `write()` → `"Initial draft:\n\n{content}"`
- First `revise()` → `"Revised draft #1:\n\n{content}"`
- Second `revise()` → `"Revised draft #2:\n\n{content}"`

`outputs` always contains the raw text without labels.

## Critic

Reviews content and returns feedback. Accumulates history so it can track whether earlier issues were fixed.

```python
Critic(
    llm: LLMProvider | str,
    system_prompt: str | Path | None = None,
    name: str = "Critic",
    context_files: list[Path | str] | None = None,
)
```

Methods:
```python
await critic.review(content: str) -> str
critic.reset()
critic.history  -> list[Message]
critic.outputs  -> list[str]
```

History labels: `"Criticism #1:\n\n{content}"`, `"Criticism #2:\n\n{content}"`, …

Bundled prompt instructs the critic to say `"No actionable issues."` when satisfied. The Orchestrator detects this via a YES/NO LLM call.

## Orchestrator

Controls the write → review → revise loop.

```python
Orchestrator(
    llm: LLMProvider | str,
    author: Author,
    critic: Critic | list[Critic],         # single or multiple critics
    system_prompt: str | Path | None = None,
    max_iterations: int = 5,
    name: str = "Orchestrator",
)
```

```python
result: str = await orchestrator.run(topic: str)
# run() resets author, all critics, and memory before starting

orchestrator.converged        -> bool             # True if all critics satisfied
orchestrator.total_iterations -> int              # number of completed cycles
orchestrator.memory           -> list[IterationRecord]
```

Loop logic:
1. `author.write(topic)` — first iteration only
2. All critics review in **parallel** (`asyncio.gather`)
3. Orchestrator's LLM judges each critic output: YES (has issues) / NO (satisfied)
4. If any critic → YES and iterations remain: combine actionable feedback → `author.revise(combined)`
5. If all critics → NO: `converged = True`, stop
6. If `iteration == max_iterations`: stop regardless

## IterationRecord

Stored in `orchestrator.memory` after each cycle.

```python
record.iteration       -> int
record.author_output   -> str
record.critic_results  -> list[CriticResult]   # one per critic
record.has_criticism   -> bool                 # True if any critic found issues (computed)
record.critic_output   -> str                  # combined text; labeled per critic if multiple (computed)
```

## CriticResult

One per critic per iteration, inside `IterationRecord.critic_results`.

```python
result.name           -> str
result.output         -> str
result.has_criticism  -> bool
```

## context_files

Both `Author` and `Critic` accept `context_files`. Text is extracted at construction time and appended to the system prompt. Supported formats: `.pdf`, `.docx`, `.pptx`, `.txt`, `.md`. Requires `pip install "peer-agents[files]"`.

```python
from pathlib import Path

author = Author(llm=llm, context_files=[Path("brief.pdf"), Path("notes.docx")])
critic = Critic(llm=llm, context_files=[Path("rubric.pdf")])
```

## Custom system_prompt

All three agents accept `system_prompt` as:
- `None` — use bundled default from `peer_agents/prompts/{author,critic,orchestrator}.txt`
- `str` — inline prompt string
- `Path` — path to a `.txt` file on disk

## Custom LLMProvider

Implement `LLMProvider` to use any backend:

```python
from peer_agents import LLMProvider, LLMResponse, Message

class MyProvider(LLMProvider):
    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools=None,
        **kwargs,
    ) -> LLMResponse:
        text = my_api_call(messages, system)
        return LLMResponse(content=text, stop_reason="end_turn")
```

`LLMResponse` fields: `content: str`, `stop_reason: str`, `tool_calls: list = []`, `raw_content: list = []`, `usage: dict = {}`.

`Message` fields: `role: str` (`"user"` or `"assistant"`), `content: str | list`.

## Complete working example

```python
import asyncio
from pathlib import Path
from peer_agents import Author, Critic, Orchestrator
from peer_agents.llm.anthropic_provider import AnthropicProvider

async def main():
    author = Author(
        llm="claude-opus-4-7",
        system_prompt="You are a concise technical writer.",
        context_files=[Path("research.pdf")],   # optional
    )

    critic_a = Critic(
        llm="gpt-4o",
        name="Accuracy",
        system_prompt="Check factual accuracy only.",
    )
    critic_b = Critic(
        llm="claude-opus-4-7",
        name="Clarity",
        system_prompt="Check clarity and readability only.",
    )

    orch = Orchestrator(
        llm="claude-opus-4-7",
        author=author,
        critic=[critic_a, critic_b],  # or a single Critic
        max_iterations=4,
    )

    result = await orch.run("Explain how transformers work")

    print(result)
    print(f"converged={orch.converged}, iterations={orch.total_iterations}")

    for record in orch.memory:
        print(f"\nIteration {record.iteration}")
        for r in record.critic_results:
            print(f"  {r.name}: has_criticism={r.has_criticism}")

asyncio.run(main())
```

## Common patterns

**Single critic, default prompts:**
```python
orch = Orchestrator(llm="claude-opus-4-7", author=Author(llm="claude-opus-4-7"), critic=Critic(llm="claude-opus-4-7"))
result = await orch.run("My topic")
```

**Mixed models:**
```python
author = Author(llm="claude-opus-4-7")
critic = Critic(llm="gpt-4o")
orch = Orchestrator(llm="gemini-2.0-flash", author=author, critic=critic)
```

**Check if loop converged:**
```python
result = await orch.run("topic")
if not orch.converged:
    print(f"Hit max_iterations ({orch.max_iterations}) without full agreement")
```

**Inspect per-critic results:**
```python
for record in orch.memory:
    for r in record.critic_results:
        if r.has_criticism:
            print(f"[{record.iteration}] {r.name} objected: {r.output[:100]}")
```

**Use Author/Critic standalone (no Orchestrator):**
```python
author = Author(llm=llm)
critic = Critic(llm=llm)

draft = await author.write("A blog post about async Python")
feedback = await critic.review(draft)
revision = await author.revise(feedback)
```

## Project layout (src/)

```
peer_agents/
├── __init__.py                  # exports Author, Critic, Orchestrator, IterationRecord, CriticResult, LLMProvider, LLMResponse, Message
├── llm/
│   ├── base.py                  # LLMProvider (ABC), LLMResponse, Message, ToolCall, ToolDefinition
│   ├── anthropic_provider.py    # AnthropicProvider
│   ├── openai_provider.py       # OpenAIProvider
│   └── gemini_provider.py       # GeminiProvider
├── agents/
│   ├── base.py                  # GenerationAgent (history, generate, reset, context_files)
│   ├── author.py                # Author
│   ├── critic.py                # Critic
│   └── orchestrator.py          # Orchestrator, IterationRecord, CriticResult
├── utils/
│   └── file_context.py          # load_context_files() — PDF/DOCX/PPTX/TXT extraction
└── prompts/
    ├── author.txt
    ├── critic.txt
    └── orchestrator.txt         # instructs LLM to reply YES or NO only
```
