# peer-agents

> An async, provider-agnostic multi-agent framework for Python — an author-critic refinement pipeline that iteratively improves content through structured LLM-powered review.

<details>
<summary>📋 AI coding agent context (Claude Code, Codex, Cursor…) — click to expand &amp; copy</summary>

```text
peer-agents — author-critic refinement pipeline for Python (async, provider-agnostic)

INSTALL
  pip install "peer-agents @ git+https://github.com/your-username/peer-agents.git"

IMPORTS
  from peer_agents import Author, Critic, Orchestrator, IterationRecord, CriticResult
  from peer_agents import LLMProvider, LLMResponse, Message
  from peer_agents.llm.anthropic_provider import AnthropicProvider  # claude-opus-4-7, ANTHROPIC_API_KEY
  from peer_agents.llm.openai_provider    import OpenAIProvider     # gpt-4o,          OPENAI_API_KEY
  from peer_agents.llm.gemini_provider    import GeminiProvider     # gemini-2.0-flash, GOOGLE_API_KEY

CLASSES  (llm = LLMProvider instance OR model-name string)
  Author(llm, system_prompt=None, name="Author", context_files=None)
    .write(topic)      -> str   [history label: "Initial draft:"]
    .revise(criticism) -> str   [history label: "Revised draft #N:"]
    .reset() | .history -> list[Message] | .outputs -> list[str]

  Critic(llm, system_prompt=None, name="Critic", context_files=None)
    .review(content)   -> str   [history label: "Criticism #N:"]
    .reset() | .history -> list[Message] | .outputs -> list[str]

  Orchestrator(llm, author, critic, system_prompt=None, max_iterations=5, name="Orchestrator")
    critic = single Critic OR list[Critic]  (parallel review via asyncio.gather)
    .run(topic) -> str          resets all agents each call
    .converged -> bool | .total_iterations -> int | .memory -> list[IterationRecord]

  IterationRecord
    .iteration, .author_output, .critic_results: list[CriticResult]
    .has_criticism -> bool (computed: any critic found issues)
    .critic_output -> str  (computed: combined text, labeled per critic if multiple)

  CriticResult  .name, .output, .has_criticism

LOOP LOGIC
  1. author.write(topic)
  2. all critics review in parallel
  3. orchestrator LLM judges each: YES (issues) / NO (satisfied)
  4. if any YES and iterations remain: combine actionable feedback -> author.revise(combined)
  5. if all NO: converged=True, stop
  6. if iteration==max_iterations: stop (converged=False)

CONTEXT FILES  (requires [files] extra)
  Author/Critic accept context_files=[Path("x.pdf"), Path("y.docx")]
  Extracted text is appended to the system prompt at construction time.
  Supported: .pdf (pypdf), .docx (python-docx), .pptx (python-pptx), .txt, .md

CUSTOM PROVIDER
  class MyProvider(LLMProvider):
      async def complete(self, messages, system=None, tools=None, **kwargs) -> LLMResponse:
          return LLMResponse(content=my_api(messages, system), stop_reason="end_turn")

MINIMAL EXAMPLE
  orch = Orchestrator(llm="claude-opus-4-7", author=Author(llm="claude-opus-4-7"), critic=Critic(llm="claude-opus-4-7"), max_iterations=3)
  result = await orch.run("Explain transformers")

MULTI-CRITIC EXAMPLE
  orch = Orchestrator(llm="claude-opus-4-7", author=Author(llm="claude-opus-4-7"),
      critic=[Critic(llm="gpt-4o", name="Facts"), Critic(llm="claude-opus-4-7", name="Style")], max_iterations=4)

FULL API CONTEXT: see context.md in the repository root
```

</details>

---

## Table of Contents

- [Installation](#installation)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [LLM Providers](#llm-providers)
- [Author-Critic Pipeline](#author-critic-pipeline)
  - [Author](#author)
  - [Critic](#critic)
  - [Multiple Critics](#multiple-critics)
  - [Feeding Files as Context](#feeding-files-as-context)
  - [Orchestrator](#orchestrator)
  - [Full Pipeline Example](#full-pipeline-example)
- [Bring Your Own LLM Provider](#bring-your-own-llm-provider)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)

---

## Installation

Install directly from GitHub. All LLM backends (Anthropic, OpenAI, Gemini) and file-context support (PDF / DOCX / PPTX) are included by default.

**With `pip`**

```bash
pip install "peer-agents @ git+https://github.com/your-username/peer-agents.git"
```

**With `uv`**

```bash
uv add "peer-agents @ git+https://github.com/your-username/peer-agents.git"
```

**Pin to a specific version** (recommended for reproducible installs)

```bash
# by tag
pip install "peer-agents @ git+https://github.com/your-username/peer-agents.git@v0.1.0"

# by commit
pip install "peer-agents @ git+https://github.com/your-username/peer-agents.git@abc1234"
```

**From source** (local development)

```bash
git clone https://github.com/your-username/peer-agents
cd peer-agents
uv sync
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          peer-agents                             │
│                                                                  │
│   ┌────────────────────────────────────────────────────────┐    │
│   │                  Author-Critic Pipeline                │    │
│   │                                                        │    │
│   │   ┌─────────────────────────────────────────────────┐  │    │
│   │   │                  Orchestrator                   │  │    │
│   │   │             (controls the loop)                 │  │    │
│   │   └───────────────────┬─────────────────────────────┘  │    │
│   │                       │                                │    │
│   │          ┌────────────┴────────────┐                   │    │
│   │          ▼                         ▼                   │    │
│   │     ┌────────┐          ┌──────────────────┐           │    │
│   │     │ Author │          │   Critic(s)      │           │    │
│   │     │        │◄────────►│  (run in         │           │    │
│   │     └────────┘          │   parallel)      │           │    │
│   │                         └──────────────────┘           │    │
│   └────────────────────────────────────────────────────────┘    │
│                                                                  │
│   ┌────────────────────────────────────────────────────────┐    │
│   │                   LLM Provider Layer                   │    │
│   │  LLMProvider (ABC) ──► AnthropicProvider  (Claude)     │    │
│   │                    ──► OpenAIProvider     (GPT-4o, ...)│    │
│   │                    ──► GeminiProvider     (Gemini, ...)│    │
│   │                    ──► (your own implementation)       │    │
│   └────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

```python
import asyncio
from peer_agents import Author, Critic, Orchestrator

async def main():
    # Pass a model name string — provider is inferred automatically
    author = Author(llm="claude-opus-4-7")
    critic = Critic(llm="claude-opus-4-7")
    orchestrator = Orchestrator(llm="claude-opus-4-7", author=author, critic=critic, max_iterations=3)

    result = await orchestrator.run("The history of the internet")
    print(result)

asyncio.run(main())
```

---

## LLM Providers

Every agent accepts either a **model name string** or a **provider instance**. When you pass a string, the provider is inferred automatically from the model name prefix.

```python
# String shorthand — provider inferred from model name
author = Author(llm="claude-opus-4-7")   # → AnthropicProvider
critic = Critic(llm="gpt-4o")            # → OpenAIProvider
orch   = Orchestrator(llm="gemini-2.0-flash", author=author, critic=critic)
```

```python
# Provider instance — full control over api_key, max_tokens, etc.
from peer_agents.llm.anthropic_provider import AnthropicProvider
from peer_agents.llm.openai_provider import OpenAIProvider
from peer_agents.llm.gemini_provider import GeminiProvider

author = Author(llm=AnthropicProvider(model="claude-opus-4-7", max_tokens=8000))
critic = Critic(llm=OpenAIProvider(model="gpt-4o-mini"))
orch   = Orchestrator(llm=GeminiProvider(), author=author, critic=critic)
```

Model name routing:

| Prefix | Provider | Env var | Default model |
|---|---|---|---|
| `claude-*` | `AnthropicProvider` | `ANTHROPIC_API_KEY` | `claude-opus-4-7` |
| `gpt-*`, `o1*`, `o3*`, `o4*` | `OpenAIProvider` | `OPENAI_API_KEY` | `gpt-4o` |
| `gemini-*` | `GeminiProvider` | `GOOGLE_API_KEY` | `gemini-2.0-flash` |

**Anthropic-specific features** — streaming, prompt caching, and adaptive thinking are enabled automatically in `AnthropicProvider`. These are Anthropic-only; other providers ignore them.

---

## Author-Critic Pipeline

The pipeline runs a self-improving loop: the **Author** writes, the **Critic(s)** review, and the **Orchestrator** decides whether to revise or stop.

```
topic
  │
  ▼
┌──────────┐   write / revise      ┌───────────────────────┐
│  Author  │ ─────────────────────►│  Critic A  │ Critic B │  (parallel)
│          │ ◄─────────────────────│            │          │
└──────────┘   combined feedback   └───────────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │   Orchestrator   │
                                    │ Any criticism?   │
                                    │  YES ──► revise  │
                                    │  NO  ──► done    │
                                    └──────────────────┘
```

### Author

Generates initial content and revises it based on feedback. Its full conversation history is preserved across iterations so the model always knows what it wrote and what feedback it received.

```python
from peer_agents import Author

author = Author(llm="claude-opus-4-7")

# First draft
draft = await author.write("Explain transformer architecture")

# Revise based on feedback
revised = await author.revise("Add more detail about the attention mechanism.")

# Inspect what was produced
print(author.outputs)       # ["First draft...", "Revised draft..."]
print(len(author.history))  # 4 messages (2 user/assistant pairs)
```

History entries are automatically labeled:

| Call | Label stored in history |
|---|---|
| `write()` | `Initial draft:` |
| First `revise()` | `Revised draft #1:` |
| Second `revise()` | `Revised draft #2:` |

**Custom system prompt**

```python
# Inline string
author = Author(llm="claude-opus-4-7", system_prompt="You are a technical writer. Be concise.")

# From a file
from pathlib import Path
author = Author(llm="gpt-4o", system_prompt=Path("my_prompts/author.txt"))

# Default (uses the bundled prompts/author.txt)
author = Author(llm="gemini-2.0-flash")
```

---

### Critic

Reviews content and returns structured feedback. Accumulates conversation history across reviews so it can track whether earlier issues were addressed.

```python
from peer_agents import Critic

critic = Critic(llm="claude-opus-4-7")

feedback = await critic.review("Transformers use attention to...")
print(feedback)

# History entries are labeled:  Criticism #1, Criticism #2, ...
print(critic.history[1].content)  # "Criticism #1:\n\n<feedback>"
```

The bundled critic prompt instructs the model to say `"No actionable issues."` when satisfied — the Orchestrator uses this to detect convergence.

---

### Multiple Critics

Pass a list of `Critic` instances to the `Orchestrator`. All critics review the author's output **in parallel**, each bringing a different perspective. The Orchestrator revises only when at least one critic finds actionable issues, and combines only the actionable feedback before passing it to the Author.

```python
fact_checker = Critic(
    llm=llm,
    name="FactChecker",
    system_prompt="Check factual accuracy only. Ignore style.",
)
style_editor = Critic(
    llm=llm,
    name="StyleEditor",
    system_prompt="Check tone, clarity, and conciseness. Ignore facts.",
)

orchestrator = Orchestrator(
    llm=llm,
    author=author,
    critic=[fact_checker, style_editor],   # list of critics
    max_iterations=4,
)
```

Each `IterationRecord` in `orchestrator.memory` contains a `critic_results` list with one `CriticResult` per critic:

```python
from peer_agents import CriticResult

for record in orchestrator.memory:
    print(f"Iteration {record.iteration} — revised: {record.has_criticism}")
    for r in record.critic_results:
        print(f"  [{r.name}] has_criticism={r.has_criticism}")
        print(f"    {r.output[:80]}...")
```

---

### Feeding Files as Context

Both `Author` and `Critic` accept a `context_files` argument — a list of paths to PDF, DOCX, PPTX, TXT, or MD files. Text is extracted at construction time and appended to the system prompt, so the LLM has the document content available on every call.

```python
from pathlib import Path
from peer_agents import Author, Critic

# Author informed by a research brief
author = Author(
    llm=llm,
    context_files=[Path("research_brief.pdf"), Path("data_summary.docx")],
)

# Critic checks against a style guide
style_critic = Critic(
    llm=llm,
    name="StyleGuide",
    system_prompt="Evaluate compliance with the provided style guide.",
    context_files=[Path("style_guide.pdf")],
)

# Critic checks against a rubric
rubric_critic = Critic(
    llm=llm,
    name="Rubric",
    system_prompt="Score the content against the provided rubric.",
    context_files=[Path("rubric.pptx")],
)
```

Supported formats:

| Extension | Requires |
|---|---|
| `.pdf` | `pypdf` |
| `.docx` | `python-docx` |
| `.pptx` | `python-pptx` |
| `.txt` / `.md` | nothing (stdlib) |

All three libraries are included in the default install.

---

### Orchestrator

Controls the loop. Uses its own LLM call to judge each critic's output (YES / NO) and decides whether to revise or stop.

```python
from peer_agents import Orchestrator

orchestrator = Orchestrator(
    llm="claude-opus-4-7",      # or a provider instance
    author=author,
    critic=critic,              # single Critic or list[Critic]
    max_iterations=5,
    system_prompt=None,         # defaults to bundled prompts/orchestrator.txt
)

result = await orchestrator.run("The future of renewable energy")

print(f"Converged:  {orchestrator.converged}")
print(f"Iterations: {orchestrator.total_iterations}")
```

**Inspecting iteration history**

```python
for record in orchestrator.memory:
    print(f"Iteration {record.iteration}")
    print(f"  Author output:  {record.author_output[:60]}...")
    print(f"  Has criticism:  {record.has_criticism}")
    for r in record.critic_results:
        print(f"  [{r.name}] {r.output[:60]}...")
```

`IterationRecord` fields:

| Field | Type | Description |
|---|---|---|
| `iteration` | `int` | Cycle number (1-indexed) |
| `author_output` | `str` | Content produced by the Author |
| `critic_results` | `list[CriticResult]` | One result per critic |
| `has_criticism` | `bool` | `True` if any critic found issues (computed) |
| `critic_output` | `str` | Combined critic text (computed) |

`CriticResult` fields:

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Critic's name |
| `output` | `str` | Raw feedback text |
| `has_criticism` | `bool` | Whether this critic found actionable issues |

**Stopping conditions**

```
Loop stops when:
  ├── all critics satisfied (has_criticism is False)  →  converged = True
  └── iteration == max_iterations                     →  converged = False
```

Calling `run()` a second time resets all agents and clears the memory, so successive calls are fully independent.

---

### Full Pipeline Example

```python
import asyncio
from pathlib import Path
from peer_agents import Author, Critic, Orchestrator

async def main():
    # Mix model strings and provider instances freely
    author = Author(
        llm="claude-opus-4-7",
        system_prompt="You are a science writer. Be accurate and accessible.",
        context_files=[Path("research_notes.pdf")],
    )
    fact_critic = Critic(
        llm="gpt-4o",
        name="FactChecker",
        system_prompt="Check for factual accuracy. Be specific about any errors.",
    )
    style_critic = Critic(
        llm="claude-opus-4-7",
        name="StyleEditor",
        system_prompt="Check for clarity and accessibility for a general audience.",
    )
    orchestrator = Orchestrator(
        llm="claude-opus-4-7",
        author=author,
        critic=[fact_critic, style_critic],
        max_iterations=4,
    )

    result = await orchestrator.run("How does CRISPR gene editing work?")

    print("=== Final Output ===")
    print(result)
    print()
    print(f"Converged:       {orchestrator.converged}")
    print(f"Iterations used: {orchestrator.total_iterations}")

    print()
    print("=== Iteration log ===")
    for r in orchestrator.memory:
        for cr in r.critic_results:
            status = "objected" if cr.has_criticism else "satisfied"
            print(f"  [{r.iteration}] {cr.name}: {status}")

asyncio.run(main())
```

---

## Bring Your Own LLM Provider

Implement the `LLMProvider` ABC to plug in any backend — a local model, a custom API, or a mock for testing:

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
        text = call_my_llm_api(messages, system)
        return LLMResponse(content=text, stop_reason="end_turn")
```

Pass an instance to any agent:

```python
author = Author(llm=MyProvider())
agent  = Agent(llm=MyProvider())
```

---

## API Reference

### Providers

```python
AnthropicProvider(
    api_key: str | None = None,   # defaults to ANTHROPIC_API_KEY env var
    model: str = "claude-opus-4-7",
    max_tokens: int = 16_000,
)

OpenAIProvider(
    api_key: str | None = None,   # defaults to OPENAI_API_KEY env var
    model: str = "gpt-4o",
    max_tokens: int = 4_096,
)

GeminiProvider(
    api_key: str | None = None,   # defaults to GOOGLE_API_KEY env var
    model: str = "gemini-2.0-flash",
    max_tokens: int = 8_192,
)
```

### `Author`

```python
Author(
    llm: LLMProvider | str,                        # provider instance or model name
    system_prompt: str | Path | None = None,       # None uses bundled default
    name: str = "Author",
    context_files: list[Path | str] | None = None, # PDF, DOCX, PPTX, TXT, MD
)

await author.write(topic: str) -> str
await author.revise(criticism: str) -> str
author.reset()                      # clear history and outputs
author.history  -> list[Message]    # full conversation history
author.outputs  -> list[str]        # raw outputs (without labels)
```

### `Critic`

```python
Critic(
    llm: LLMProvider | str,
    system_prompt: str | Path | None = None,
    name: str = "Critic",
    context_files: list[Path | str] | None = None,
)

await critic.review(content: str) -> str
critic.reset()
critic.history  -> list[Message]
critic.outputs  -> list[str]
```

### `Orchestrator`

```python
Orchestrator(
    llm: LLMProvider | str,
    author: Author,
    critic: Critic | list[Critic],   # single or multiple critics
    system_prompt: str | Path | None = None,
    max_iterations: int = 5,
    name: str = "Orchestrator",
)

await orchestrator.run(topic: str) -> str   # resets all agents on each call
orchestrator.converged        -> bool
orchestrator.total_iterations -> int
orchestrator.memory           -> list[IterationRecord]
```

### `IterationRecord`

```python
record.iteration       -> int
record.author_output   -> str
record.critic_results  -> list[CriticResult]
record.has_criticism   -> bool   # True if any critic found issues
record.critic_output   -> str    # combined text (labeled per critic if multiple)
```

### `CriticResult`

```python
result.name           -> str
result.output         -> str
result.has_criticism  -> bool
```

---

## Running Tests

```bash
uv run pytest              # all tests
uv run pytest -v           # verbose output
```

Tests use a `MockProvider` that replays a fixed sequence of responses — no API key required.
