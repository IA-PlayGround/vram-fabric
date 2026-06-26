from __future__ import annotations

import pytest

from vram_fabric.core.types import Task, Workflow
from vram_fabric.engines.agent_runtime import AgentRuntime


class TestAgentRuntime:
    def setup_method(self) -> None:
        self.runtime = AgentRuntime(max_instances=8, hidden_dim=128)

    def test_register_agent(self) -> None:
        handle = self.runtime.register("assistant", "You are helpful.")
        assert handle.name == "assistant"
        assert handle.system_prompt == "You are helpful."

    def test_register_duplicate_raises(self) -> None:
        self.runtime.register("agent1", "prompt")
        with pytest.raises(ValueError):
            self.runtime.register("agent1", "another prompt")

    def test_spawn_agent(self) -> None:
        self.runtime.register("assistant", "You are helpful.")
        instance = self.runtime.spawn("assistant")
        assert instance is not None
        assert instance.active is True

    def test_spawn_unregistered_raises(self) -> None:
        with pytest.raises(ValueError):
            self.runtime.spawn("nonexistent")

    def test_run_task_without_llm(self) -> None:
        self.runtime.register("echo", "You echo the user.")
        task = Task(agent_name="echo", prompt="Hello!", task_id="t1")
        result = self.runtime.run_task(task)
        assert "No LLM backend" in result

    def test_run_task_with_llm(self) -> None:
        self.runtime.register("test", "You are a test agent.")

        def mock_llm(prompt: str) -> str:
            return f"MOCK: {prompt[:50]}"

        self.runtime.set_llm_backend(mock_llm)

        task = Task(agent_name="test", prompt="What is 2+2?", task_id="t1")
        result = self.runtime.run_task(task)
        assert "MOCK:" in result
        assert "test agent" in result

    def test_workflow_sequential(self) -> None:
        self.runtime.register("step1", "Agent 1")
        self.runtime.register("step2", "Agent 2")

        t1 = Task(agent_name="step1", prompt="Step 1", task_id="t1")
        t2 = Task(agent_name="step2", prompt="Step 2", task_id="t2")
        t2.dependencies = ["t1"]

        workflow = Workflow(tasks=[t1, t2])
        result = self.runtime.run_workflow(workflow)

        assert "t1" in result.results
        assert "t2" in result.results
        assert result.total_time_ms >= 0

    def test_workflow_parallel(self) -> None:
        self.runtime.register("a", "Agent A")
        self.runtime.register("b", "Agent B")

        t1 = Task(agent_name="a", prompt="Task A", task_id="t1")
        t2 = Task(agent_name="b", prompt="Task B", task_id="t2")

        workflow = Workflow(tasks=[t1, t2])
        result = self.runtime.run_workflow(workflow)

        assert len(result.results) == 2

    def test_collect_instance(self) -> None:
        self.runtime.register("agent", "prompt")
        instance = self.runtime.spawn("agent")
        inst_ids = list(self.runtime._instances.keys())

        assert len(inst_ids) > 0
        self.runtime.collect(inst_ids[0])
        assert not instance.active

    def test_shutdown(self) -> None:
        self.runtime.register("agent", "prompt")
        self.runtime.shutdown()
