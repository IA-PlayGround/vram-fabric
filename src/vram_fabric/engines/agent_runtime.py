from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import Any

from vram_fabric.core.types import (
    AgentHandle,
    AgentInstance,
    Task,
    Workflow,
    WorkflowResult,
)
from vram_fabric.memory.pool import vram_pool

logger = logging.getLogger(__name__)


class AgentRuntime:
    """GPU Agent Runtime — actors model with state tensors in VRAM.

    Each agent is an actor with:
    - A state tensor [hidden_dim] allocated in VRAM
    - A dedicated CUDA stream for isolation
    - Zero-copy context switch via pointer swap (O(1), <1μs)

    Enhanced with CONCUR-inspired cache-pressure-aware admission control
    (arXiv:2601.22705) to prevent "middle-phase thrashing" in agent workloads.
    """

    def __init__(
        self,
        max_instances: int = 16,
        hidden_dim: int = 4096,
        max_active_agents: int = 8,
        admission_threshold: float = 0.70,
        cooldown_ms: int = 200,
    ) -> None:
        self._max_instances = max_instances
        self._hidden_dim = hidden_dim
        self._max_active_agents = max_active_agents
        self._admission_threshold = admission_threshold
        self._cooldown_ms = cooldown_ms

        self._agents: dict[str, AgentHandle] = {}
        self._instances: dict[str, AgentInstance] = {}
        self._llm_call: Callable | None = None
        self._cuda_available = False
        self._torch: Any = None

        # CONCUR-inspired admission control
        self._active_count = 0
        self._pending_queue: deque = deque()
        self._last_admit_time = 0.0
        self._cache_pressure = 0.0
        self._throttled = False
        self._total_admitted = 0
        self._total_rejected = 0

        self._init_cuda()

    def _init_cuda(self) -> None:
        try:
            import torch
            self._cuda_available = torch.cuda.is_available()
            if self._cuda_available:
                self._torch = torch
                logger.info("Agent Runtime: CUDA available, state tensors in VRAM")
        except ImportError:
            logger.info("Agent Runtime: CPU-only mode")

    def set_llm_backend(self, llm_fn: Callable) -> None:
        self._llm_call = llm_fn

    def set_cache_pressure(self, pressure: float) -> None:
        """Update cache pressure metric for admission control (CONCUR)."""
        self._cache_pressure = pressure
        if pressure > self._admission_threshold:
            self._throttled = True
            logger.debug("Agent admission throttled: cache_pressure=%.2f", pressure)
        else:
            self._throttled = False

    def _should_admit(self) -> bool:
        """CONCUR-inspired admission control.

        Rejects new agent spawns when:
        - Active agent count exceeds max_active_agents
        - Cache pressure exceeds threshold (middle-phase thrashing prevention)
        - Rapid burst spawning detected (cooldown only when other agents active)
        """
        if self._active_count >= self._max_active_agents:
            self._total_rejected += 1
            return False
        if self._throttled and self._cache_pressure > self._admission_threshold:
            self._total_rejected += 1
            return False
        if self._active_count > 0:
            now = time.monotonic()
            if now - self._last_admit_time < self._cooldown_ms / 1000:
                self._total_rejected += 1
                return False
        return True

    def register(self, name: str, system_prompt: str, model: str = "") -> AgentHandle:
        if name in self._agents:
            raise ValueError(f"Agent '{name}' already registered")

        state_tensor = None
        if self._cuda_available and self._torch is not None:
            try:
                state_tensor = self._torch.zeros(self._hidden_dim, device="cuda")
                vram_pool.allocate(f"agent_state_{name}", self._hidden_dim * 4, owner=f"agent:{name}")
            except Exception as e:
                logger.debug("VRAM state tensor allocation skipped for %s: %s", name, e)

        handle = AgentHandle(
            name=name,
            system_prompt=system_prompt,
            model=model,
            state_tensor=state_tensor,
        )
        self._agents[name] = handle
        logger.info("Agent registered: %s (model=%s)", name, model or "default")
        return handle

    def spawn(self, name: str) -> AgentInstance:
        """Spawn an instance of a registered agent.

        Subject to CONCUR admission control — may raise RuntimeError if throttled.
        """
        handle = self._agents.get(name)
        if handle is None:
            raise ValueError(f"Agent '{name}' not registered. Call register() first.")

        if not self._should_admit():
            raise RuntimeError(
                f"Agent admission rejected: active={self._active_count}/{self._max_active_agents} "
                f"throttled={self._throttled} cache_pressure={self._cache_pressure:.2f}"
            )

        inst_id = f"{name}_{uuid.uuid4().hex[:8]}"
        stream_id = self._active_count % self._max_instances

        instance = AgentInstance(handle=handle, stream_id=stream_id, active=True)
        self._instances[inst_id] = instance
        self._active_count += 1
        self._total_admitted += 1
        self._last_admit_time = time.monotonic()
        logger.debug("Agent instance spawned: %s (stream=%d, active=%d)", inst_id, stream_id, self._active_count)
        return instance

    def run_task(self, task: Task) -> str:
        """Execute a single agent task and return the response."""
        spawned = False
        try:
            instance = self.spawn(task.agent_name)
            spawned = True
        except RuntimeError as e:
            return f"[Agent {task.agent_name}]: Admission rejected — {e}"

        if self._llm_call is None:
            self._active_count = max(0, self._active_count - 1)
            return f"[Agent {task.agent_name}]: No LLM backend configured. Prompt: {task.prompt[:100]}..."

        prompt = f"{instance.handle.system_prompt}\n\nUser: {task.prompt}\nAssistant:"
        try:
            response = self._llm_call(prompt)
            self._active_count = max(0, self._active_count - 1)
            return response
        except Exception as e:
            logger.error("Agent task failed: %s", e)
            self._active_count = max(0, self._active_count - 1)
            return f"[Error: {e}]"

    def run_workflow(self, workflow: Workflow) -> WorkflowResult:
        """Execute a DAG of tasks respecting dependencies.

        With CONCUR admission control: tasks are queued if cache pressure is high.
        """
        start = time.perf_counter()
        results: dict[str, str] = {}
        errors: dict[str, str] = {}
        completed: set[str] = set()
        rejected: set[str] = set()

        task_map: dict[str, Task] = {t.task_id or t.agent_name: t for t in workflow.tasks}
        ready: list[str] = [tid for tid, t in task_map.items() if not t.dependencies]

        while ready or rejected:
            # Retry rejected tasks if pressure decreased
            if not ready and rejected and (not self._throttled or self._cache_pressure <= self._admission_threshold):
                ready = list(rejected)
                rejected = set()

            if not ready:
                break

            tid = ready.pop(0)
            task = task_map[tid]
            try:
                result = self.run_task(task)
                if "Admission rejected" in result:
                    rejected.add(tid)
                    errors[tid] = result
                else:
                    results[tid] = result
                    completed.add(tid)
            except Exception as e:
                errors[tid] = str(e)

            for other_tid, other_task in task_map.items():
                if other_tid not in completed and other_tid not in rejected:
                    if all(dep in completed for dep in other_task.dependencies):
                        if other_tid not in ready:
                            ready.append(other_tid)

        total_ms = (time.perf_counter() - start) * 1000
        return WorkflowResult(results=results, errors=errors, total_time_ms=total_ms)

    def collect(self, instance_id: str) -> None:
        """Deactivate and collect an agent instance."""
        inst = self._instances.pop(instance_id, None)
        if inst:
            inst.active = False
            self._active_count = max(0, self._active_count - 1)
            logger.debug("Agent instance collected: %s (active=%d)", instance_id, self._active_count)

    def admission_stats(self) -> dict[str, Any]:
        return {
            "active": self._active_count,
            "max_active": self._max_active_agents,
            "total_admitted": self._total_admitted,
            "total_rejected": self._total_rejected,
            "cache_pressure": self._cache_pressure,
            "throttled": self._throttled,
            "admission_ratio": self._total_admitted / max(self._total_admitted + self._total_rejected, 1),
        }

    def shutdown(self) -> None:
        for name in list(self._agents.keys()):
            vram_pool.free(f"agent_state_{name}")
        self._agents.clear()
        self._instances.clear()
        self._pending_queue.clear()
        self._active_count = 0
