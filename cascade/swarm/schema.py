"""Data types for swarm dispatch."""

from dataclasses import dataclass, field


# Model routing hints: what each provider is best at
PROVIDER_STRENGTHS: dict[str, list[str]] = {
    "gemini": ["large context", "research", "analysis", "documentation", "summarization"],
    "claude": ["planning", "reasoning", "architecture", "code review", "refactoring"],
    "openai": ["code generation", "implementation", "debugging", "testing"],
    "openrouter": ["code generation", "fast execution", "translation"],
}


@dataclass(frozen=True)
class SubTask:
    """A single subtask to be dispatched to a provider."""

    id: str
    description: str
    provider: str              # recommended provider
    prompt: str                # full prompt for the provider
    depends_on: tuple[str, ...] = ()  # subtask ids this depends on
    priority: int = 0          # execution priority (lower = first)


@dataclass
class SubTaskResult:
    """Result from executing a subtask."""

    task_id: str
    provider: str
    response: str
    tokens: int = 0
    success: bool = True
    error: str = ""


@dataclass(frozen=True)
class SwarmPlan:
    """A decomposed plan of subtasks for parallel execution."""

    objective: str
    subtasks: tuple[SubTask, ...]
    orchestrator_provider: str
    synthesis_prompt: str = ""  # prompt for combining results


@dataclass
class SwarmResult:
    """Final result of a swarm execution."""

    objective: str
    subtask_results: list[SubTaskResult] = field(default_factory=list)
    synthesis: str = ""
    total_tokens: int = 0
    providers_used: list[str] = field(default_factory=list)


@dataclass
class CompetitionEntry:
    """Result from running the same task against one provider."""

    provider: str
    response: str
    tokens: int = 0
    duration_seconds: float = 0.0
    success: bool = True
    error: str = ""
    worktree_path: str = ""
    changed_files: list[str] = field(default_factory=list)
    diff_stat: str = ""
    diff_excerpt: str = ""
    retained: bool = False


@dataclass(frozen=True)
class CompetitionJudgment:
    """Judge output for a competition run."""

    winner_provider: str
    rationale: str
    summary: str = ""


@dataclass
class CompetitionResult:
    """Final result of a parallel competition across providers."""

    objective: str
    entries: list[CompetitionEntry] = field(default_factory=list)
    judgment: CompetitionJudgment | None = None
    winner_provider: str = ""
    winner_response: str = ""
    total_tokens: int = 0
    judge_provider: str = ""
