"""Tests for the swarm dispatch system."""

import json
from unittest.mock import MagicMock

from cascade.swarm.schema import (
    SubTask,
    SubTaskResult,
    SwarmPlan,
    SwarmResult,
    PROVIDER_STRENGTHS,
)
from cascade.swarm.orchestrator import SwarmOrchestrator


def _mock_app(providers=None):
    """Create a mock CascadeCore with provider stubs."""
    app = MagicMock()
    if providers is None:
        providers = {"claude": _mock_provider("claude"), "gemini": _mock_provider("gemini")}
    app.providers = providers
    return app


def _mock_provider(name, response="Mock response", model="test-model"):
    """Create a mock provider that returns a fixed response."""
    prov = MagicMock()
    prov.name = name
    prov.config = MagicMock()
    prov.config.model = model
    prov.ask_single.return_value = response
    prov.last_usage = (100, 50)
    return prov


class TestSubTask:
    def test_frozen(self):
        task = SubTask(id="t1", description="Test", provider="claude", prompt="Do it")
        assert task.id == "t1"
        assert task.depends_on == ()
        assert task.priority == 0


class TestSwarmPlan:
    def test_basic_plan(self):
        plan = SwarmPlan(
            objective="Build an API",
            subtasks=(
                SubTask(id="t1", description="Plan", provider="claude", prompt="Plan it"),
                SubTask(id="t2", description="Build", provider="openai", prompt="Build it"),
            ),
            orchestrator_provider="claude",
        )
        assert len(plan.subtasks) == 2
        assert plan.orchestrator_provider == "claude"


class TestSwarmOrchestrator:
    def test_picks_best_orchestrator(self):
        app = _mock_app({"gemini": _mock_provider("gemini"), "openai": _mock_provider("openai")})
        swarm = SwarmOrchestrator(app)
        # Claude not available, should pick gemini (second in preference)
        assert swarm._orchestrator == "gemini"

    def test_explicit_orchestrator(self):
        app = _mock_app()
        swarm = SwarmOrchestrator(app, orchestrator_provider="gemini")
        assert swarm._orchestrator == "gemini"

    def test_no_providers_raises(self):
        app = _mock_app(providers={})
        try:
            SwarmOrchestrator(app)
            assert False, "Should raise"
        except RuntimeError:
            pass

    def test_plan_parses_json_response(self):
        plan_response = json.dumps({
            "subtasks": [
                {
                    "id": "t1",
                    "description": "Research",
                    "provider": "gemini",
                    "prompt": "Research the topic",
                    "depends_on": [],
                    "priority": 0,
                },
                {
                    "id": "t2",
                    "description": "Write code",
                    "provider": "claude",
                    "prompt": "Write the implementation",
                    "depends_on": ["t1"],
                    "priority": 1,
                },
            ],
            "synthesis_prompt": "Combine research with code",
        })
        app = _mock_app()
        app.providers["claude"].ask_single.return_value = plan_response
        swarm = SwarmOrchestrator(app)

        plan = swarm.plan("Build a feature")
        assert len(plan.subtasks) == 2
        assert plan.subtasks[0].provider == "gemini"
        assert plan.subtasks[1].depends_on == ("t1",)
        assert plan.synthesis_prompt == "Combine research with code"

    def test_plan_fallback_on_bad_json(self):
        app = _mock_app()
        app.providers["claude"].ask_single.return_value = "Not valid JSON at all"
        swarm = SwarmOrchestrator(app)

        plan = swarm.plan("Do something")
        # Should fall back to single-task plan
        assert len(plan.subtasks) == 1
        assert plan.subtasks[0].prompt == "Do something"

    def test_plan_validates_providers(self):
        plan_response = json.dumps({
            "subtasks": [
                {
                    "id": "t1",
                    "description": "Test",
                    "provider": "nonexistent_provider",
                    "prompt": "Test prompt",
                },
            ],
        })
        app = _mock_app()
        app.providers["claude"].ask_single.return_value = plan_response
        swarm = SwarmOrchestrator(app)

        plan = swarm.plan("Test")
        # Should fall back to orchestrator provider
        assert plan.subtasks[0].provider == "claude"

    def test_execute_subtask(self):
        app = _mock_app()
        swarm = SwarmOrchestrator(app)

        task = SubTask(id="t1", description="Test", provider="claude", prompt="Do it")
        result = swarm._execute_subtask(task, {})
        assert result.success is True
        assert result.response == "Mock response"
        assert result.tokens == 150  # 100 + 50

    def test_execute_subtask_unavailable_provider(self):
        app = _mock_app()
        swarm = SwarmOrchestrator(app)

        task = SubTask(id="t1", description="Test", provider="missing", prompt="Do it")
        result = swarm._execute_subtask(task, {})
        assert result.success is False
        assert "not available" in result.error

    def test_execute_subtask_with_dependencies(self):
        app = _mock_app()
        swarm = SwarmOrchestrator(app)

        prior = {
            "t1": SubTaskResult(
                task_id="t1", provider="gemini",
                response="Research findings here",
                success=True,
            ),
        }
        task = SubTask(
            id="t2", description="Build", provider="claude",
            prompt="Build based on research", depends_on=("t1",),
        )
        result = swarm._execute_subtask(task, prior)
        assert result.success is True
        # Verify the dependency result was injected
        call_args = app.providers["claude"].ask_single.call_args
        assert "Research findings here" in call_args[0][0]

    def test_full_execute(self):
        # Set up orchestrator to return a simple 2-task plan
        plan_response = json.dumps({
            "subtasks": [
                {"id": "t1", "description": "Research", "provider": "gemini", "prompt": "Research"},
                {"id": "t2", "description": "Write", "provider": "claude", "prompt": "Write"},
            ],
            "synthesis_prompt": "Combine results",
        })

        app = _mock_app()
        # First call = plan, subsequent calls = execution/synthesis
        app.providers["claude"].ask_single.side_effect = [
            plan_response,      # plan
            "Write result",     # execute t2
            "Final synthesis",  # synthesize
        ]
        app.providers["gemini"].ask_single.return_value = "Research result"

        swarm = SwarmOrchestrator(app)
        result = swarm.execute("Build a feature")

        assert isinstance(result, SwarmResult)
        assert len(result.subtask_results) == 2
        assert result.synthesis == "Final synthesis"
        assert "claude" in result.providers_used
        assert "gemini" in result.providers_used

    def test_progress_callback(self):
        plan_response = json.dumps({
            "subtasks": [
                {"id": "t1", "description": "Task", "provider": "claude", "prompt": "Do it"},
            ],
        })

        app = _mock_app()
        app.providers["claude"].ask_single.side_effect = [
            plan_response,
            "Done",
        ]

        progress_calls = []

        def on_progress(stage, detail):
            progress_calls.append((stage, detail))

        swarm = SwarmOrchestrator(app)
        swarm.execute("Test task", on_progress=on_progress)

        stages = [p[0] for p in progress_calls]
        assert "planning" in stages
        assert "executing" in stages

    def test_single_task_skips_synthesis(self):
        plan_response = json.dumps({
            "subtasks": [
                {"id": "t1", "description": "Only task", "provider": "claude", "prompt": "Do it"},
            ],
        })

        app = _mock_app()
        app.providers["claude"].ask_single.side_effect = [
            plan_response,
            "Direct result",
        ]

        swarm = SwarmOrchestrator(app)
        result = swarm.execute("Simple task")

        # Should use the direct result, not synthesize
        assert result.synthesis == "Direct result"
        # ask_single should be called exactly twice (plan + execute), no synthesis
        assert app.providers["claude"].ask_single.call_count == 2


class TestProviderStrengths:
    def test_all_providers_have_strengths(self):
        for provider in ("gemini", "claude", "openai", "openrouter"):
            assert provider in PROVIDER_STRENGTHS
            assert len(PROVIDER_STRENGTHS[provider]) > 0
