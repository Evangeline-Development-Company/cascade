"""Swarm orchestrator for multi-model task dispatch.

Uses one provider to plan task decomposition, then dispatches
subtasks to the best available providers in parallel.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, TYPE_CHECKING

from .schema import (
    PROVIDER_STRENGTHS,
    SubTask,
    SubTaskResult,
    SwarmPlan,
    SwarmResult,
)

if TYPE_CHECKING:
    from ..cli import CascadeCore


# System prompt for the orchestrator
_ORCHESTRATOR_SYSTEM = """\
You are a task orchestrator. You decompose complex tasks into subtasks \
that can be executed in parallel by different AI models.

Available providers and their strengths:
{provider_info}

Respond with a JSON object containing:
{{
  "subtasks": [
    {{
      "id": "task_1",
      "description": "Brief description of what this subtask does",
      "provider": "provider_name",
      "prompt": "The full prompt to send to this provider",
      "depends_on": [],
      "priority": 0
    }}
  ],
  "synthesis_prompt": "Instructions for combining the results"
}}

Rules:
- Each subtask should be self-contained and bounded
- Assign providers based on their strengths
- Use depends_on only when genuinely needed (prefer parallel)
- Keep subtask count between 2 and 6
- Priority 0 = highest (runs first)
- The synthesis_prompt tells the final model how to combine results
"""

_SYNTHESIS_SYSTEM = """\
You synthesize results from multiple AI providers into a coherent response.
Each result was produced by a different model working on a subtask.
Integrate the results, resolve any conflicts, and present a unified answer.
"""


ProgressCallback = Optional[Callable[[str, str], None]]


class SwarmOrchestrator:
    """Orchestrate multi-model task execution.

    Usage:
        swarm = SwarmOrchestrator(cascade_core)
        result = swarm.execute("Build a REST API for user management")
    """

    def __init__(
        self,
        app: "CascadeCore",
        orchestrator_provider: Optional[str] = None,
        max_workers: int = 4,
    ) -> None:
        self._app = app
        self._orchestrator = orchestrator_provider or self._pick_orchestrator()
        self._max_workers = max_workers

    def _pick_orchestrator(self) -> str:
        """Pick the best available provider for orchestration."""
        preference = ["claude", "gemini", "openai", "openrouter"]
        for name in preference:
            if name in self._app.providers:
                return name
        # Fall back to whatever is available
        available = list(self._app.providers.keys())
        if not available:
            raise RuntimeError("No providers available for swarm orchestration")
        return available[0]

    def _available_provider_info(self) -> str:
        """Build provider info string for the orchestrator prompt."""
        lines = []
        for name, prov in self._app.providers.items():
            strengths = PROVIDER_STRENGTHS.get(name, ["general"])
            model = prov.config.model
            lines.append(f"- {name} ({model}): {', '.join(strengths)}")
        return "\n".join(lines)

    def plan(self, objective: str, on_progress: ProgressCallback = None) -> SwarmPlan:
        """Use the orchestrator to decompose a task into subtasks."""
        if on_progress:
            on_progress("planning", f"Orchestrator ({self._orchestrator}) decomposing task...")

        provider_info = self._available_provider_info()
        system = _ORCHESTRATOR_SYSTEM.format(provider_info=provider_info)

        prov = self._app.providers[self._orchestrator]
        response = prov.ask_single(
            f"Decompose this task into parallel subtasks:\n\n{objective}",
            system=system,
        )

        return self._parse_plan(objective, response)

    def _parse_plan(self, objective: str, response: str) -> SwarmPlan:
        """Parse the orchestrator's JSON response into a SwarmPlan."""
        # Extract JSON from response (might be wrapped in markdown)
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            # Fallback: single-task plan using the orchestrator
            return SwarmPlan(
                objective=objective,
                subtasks=(SubTask(
                    id="task_1",
                    description="Complete the full task",
                    provider=self._orchestrator,
                    prompt=objective,
                ),),
                orchestrator_provider=self._orchestrator,
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return SwarmPlan(
                objective=objective,
                subtasks=(SubTask(
                    id="task_1",
                    description="Complete the full task",
                    provider=self._orchestrator,
                    prompt=objective,
                ),),
                orchestrator_provider=self._orchestrator,
            )

        raw_tasks = data.get("subtasks", [])
        subtasks = []
        available = set(self._app.providers.keys())

        for task in raw_tasks:
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("id", f"task_{len(subtasks) + 1}"))
            provider = str(task.get("provider", self._orchestrator))

            # Validate provider exists, fall back if not
            if provider not in available:
                provider = self._orchestrator

            subtasks.append(SubTask(
                id=task_id,
                description=str(task.get("description", "")),
                provider=provider,
                prompt=str(task.get("prompt", "")),
                depends_on=tuple(str(d) for d in task.get("depends_on", [])),
                priority=int(task.get("priority", 0)),
            ))

        if not subtasks:
            subtasks = [SubTask(
                id="task_1",
                description="Complete the full task",
                provider=self._orchestrator,
                prompt=objective,
            )]

        return SwarmPlan(
            objective=objective,
            subtasks=tuple(subtasks),
            orchestrator_provider=self._orchestrator,
            synthesis_prompt=str(data.get("synthesis_prompt", "")),
        )

    def _execute_subtask(
        self,
        task: SubTask,
        prior_results: dict[str, SubTaskResult],
        on_progress: ProgressCallback = None,
    ) -> SubTaskResult:
        """Execute a single subtask against its assigned provider."""
        if on_progress:
            on_progress("executing", f"[{task.provider}] {task.description}")

        prov = self._app.providers.get(task.provider)
        if prov is None:
            return SubTaskResult(
                task_id=task.id,
                provider=task.provider,
                response="",
                success=False,
                error=f"Provider '{task.provider}' not available",
            )

        # Inject dependency results into prompt
        prompt = task.prompt
        if task.depends_on:
            dep_context = []
            for dep_id in task.depends_on:
                dep_result = prior_results.get(dep_id)
                if dep_result and dep_result.success:
                    dep_context.append(
                        f"[Result from {dep_id}]\n{dep_result.response[:2000]}"
                    )
            if dep_context:
                prompt = "\n\n".join(dep_context) + f"\n\nTask:\n{prompt}"

        try:
            response = prov.ask_single(prompt)
            usage = prov.last_usage or (0, 0)
            return SubTaskResult(
                task_id=task.id,
                provider=task.provider,
                response=response,
                tokens=usage[0] + usage[1],
                success=True,
            )
        except Exception as e:
            return SubTaskResult(
                task_id=task.id,
                provider=task.provider,
                response="",
                success=False,
                error=str(e),
            )

    def _synthesize(
        self,
        plan: SwarmPlan,
        results: list[SubTaskResult],
        on_progress: ProgressCallback = None,
    ) -> str:
        """Combine subtask results into a unified response."""
        if on_progress:
            on_progress("synthesizing", "Combining results...")

        # Build synthesis input
        result_blocks = []
        for r in results:
            if r.success:
                result_blocks.append(
                    f"[{r.task_id} via {r.provider}]\n{r.response}"
                )
            else:
                result_blocks.append(
                    f"[{r.task_id} via {r.provider}] FAILED: {r.error}"
                )

        synthesis_prompt = (
            f"Original objective: {plan.objective}\n\n"
            f"Subtask results:\n\n"
            + "\n\n---\n\n".join(result_blocks)
        )

        if plan.synthesis_prompt:
            synthesis_prompt += f"\n\nSynthesis instructions: {plan.synthesis_prompt}"

        prov = self._app.providers[self._orchestrator]
        return prov.ask_single(synthesis_prompt, system=_SYNTHESIS_SYSTEM)

    def execute(
        self,
        objective: str,
        on_progress: ProgressCallback = None,
    ) -> SwarmResult:
        """Full swarm execution: plan -> dispatch -> synthesize.

        Args:
            objective: The high-level task to accomplish.
            on_progress: Optional callback (stage, detail) for progress updates.

        Returns:
            SwarmResult with all subtask results and final synthesis.
        """
        # Phase 1: Plan
        plan = self.plan(objective, on_progress)

        # Phase 2: Execute subtasks
        completed: dict[str, SubTaskResult] = {}
        all_results: list[SubTaskResult] = []

        # Group by priority level for wave-based execution
        tasks_by_priority: dict[int, list[SubTask]] = {}
        for task in plan.subtasks:
            tasks_by_priority.setdefault(task.priority, []).append(task)

        for priority in sorted(tasks_by_priority.keys()):
            wave = tasks_by_priority[priority]

            # Check dependencies are satisfied
            ready = [t for t in wave if all(d in completed for d in t.depends_on)]
            blocked = [t for t in wave if t not in ready]

            # Execute ready tasks in parallel
            # Pass snapshot of completed (not live dict) to avoid races
            snapshot = dict(completed)
            wave_results: list[SubTaskResult] = []
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                futures = {
                    pool.submit(
                        self._execute_subtask, task, snapshot, on_progress
                    ): task
                    for task in ready
                }
                for future in as_completed(futures):
                    wave_results.append(future.result())

            # Merge wave results after all threads are done
            for result in wave_results:
                completed[result.task_id] = result
                all_results.append(result)

            # Execute any blocked tasks sequentially (deps now satisfied)
            for task in blocked:
                result = self._execute_subtask(task, completed, on_progress)
                completed[result.task_id] = result
                all_results.append(result)

        # Phase 3: Synthesize (skip if only one subtask)
        if len(all_results) == 1 and all_results[0].success:
            synthesis = all_results[0].response
        else:
            synthesis = self._synthesize(plan, all_results, on_progress)

        # Calculate totals
        total_tokens = sum(r.tokens for r in all_results)
        providers_used = list({r.provider for r in all_results})

        return SwarmResult(
            objective=objective,
            subtask_results=all_results,
            synthesis=synthesis,
            total_tokens=total_tokens,
            providers_used=providers_used,
        )
