from .agents import Author, Critic, CriticResult, GenerationAgent, IterationRecord, Orchestrator
from .llm.base import LLMProvider, LLMResponse, Message, ToolCall, ToolDefinition

__all__ = [
    # Author-critic pipeline
    "GenerationAgent",
    "Author",
    "Critic",
    "Orchestrator",
    "IterationRecord",
    "CriticResult",
    # LLM layer
    "LLMProvider",
    "LLMResponse",
    "Message",
    "ToolCall",
    "ToolDefinition",
]


def main() -> None:
    print("peer-agents: async multi-agent framework")
