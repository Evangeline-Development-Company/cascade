"""Swarm and competition system for multi-model orchestration.

Inspired by Slate's Thread Weaving: an orchestrator model plans and
decomposes tasks, then dispatches subtasks to the best available
providers in parallel. Competitive execution runs the same task against
multiple providers and judges the winner.
"""

from .schema import (
    CompetitionEntry,
    CompetitionJudgment,
    CompetitionResult,
    SubTask,
    SwarmPlan,
    SwarmResult,
)
from .competition import CompetitionOrchestrator
from .orchestrator import SwarmOrchestrator

__all__ = [
    "CompetitionEntry",
    "CompetitionJudgment",
    "CompetitionOrchestrator",
    "CompetitionResult",
    "SubTask",
    "SwarmPlan",
    "SwarmResult",
    "SwarmOrchestrator",
]
