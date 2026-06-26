"""Integration tests for VRAMFabric and regression tests for bugs fixed."""
from __future__ import annotations

import uuid

import numpy as np
import pytest

from vram_fabric.core.fabric import VRAMFabric
from vram_fabric.core.types import Task, Workflow
from vram_fabric.engines.vector_engine import VectorEngine, VectorIndexConfig


class TestVRAMFabricIntegration:
    def setup_method(self) -> None:
        self.fabric = VRAMFabric(
            vector_dim=64,
            cache_size_mb=16,
            scheduler_policy="auto",
        )
        self.fabric.scheduler.stop()

    def teardown_method(self) -> None:
        self.fabric.shutdown()

    def test_index_and_search(self) -> None:
        docs = [f"document number {i}" for i in range(20)]
        count = self.fabric.index_documents(docs)
        assert count == 20

        results = self.fabric.search("document number 5", k=3)
        assert len(results) == 3
        for r in results:
            assert r.score >= -1.0

    def test_query_cache_miss_then_hit(self) -> None:
        responses: list[str] = []
        responses.append(self.fabric.query("What is AI?"))
        responses.append(self.fabric.query("What is AI?"))

        stats = self.fabric.cache_stats()
        assert stats.hits >= 1
        assert stats.misses >= 1
        assert responses[0] == responses[1]

    def test_query_cache_stats_hit_ratio(self) -> None:
        self.fabric.query("hello world")
        self.fabric.query("hello world")
        self.fabric.query("hello world")

        stats = self.fabric.cache_stats()
        assert stats.hit_ratio > 0.0
        assert stats.hits >= 2
        assert stats.avg_lookup_time_ms >= 0.0

    def test_register_and_run_agent(self) -> None:
        self.fabric.register_agent("assistant", "You are helpful.")

        def mock_llm(prompt: str) -> str:
            return f"ANSWER: {prompt[:30]}"

        self.fabric.set_llm_backend(mock_llm)
        task = self.fabric.task("assistant", "What is 2+2?")
        workflow = self.fabric.create_workflow([task])
        result = self.fabric.run(workflow)

        assert len(result.results) == 1
        answer = list(result.results.values())[0]
        assert "ANSWER:" in answer

    def test_multi_agent_workflow_no_duplicate_task_ids(self) -> None:
        """Regression: fabric.task() used to generate duplicate IDs for the same agent."""
        self.fabric.register_agent("bot", "You are a bot.")
        t1 = self.fabric.task("bot", "Step 1")
        t2 = self.fabric.task("bot", "Step 2")
        assert t1.task_id != t2.task_id

    def test_workflow_with_explicit_task_id(self) -> None:
        self.fabric.register_agent("agent", "Agent prompt.")
        t = self.fabric.task("agent", "Do something", task_id="custom_id")
        assert t.task_id == "custom_id"

    def test_memory_stats_structure(self) -> None:
        mem = self.fabric.memory_stats()
        assert hasattr(mem, "vram_total_mb")
        assert hasattr(mem, "vram_used_mb")
        assert hasattr(mem, "spillover_count")

    def test_scheduler_stats_structure(self) -> None:
        stats = self.fabric.scheduler_stats()
        assert "llm" in stats.shares
        assert "vector" in stats.shares
        assert isinstance(stats.throttled, bool)

    def test_cache_prefetch_stats(self) -> None:
        self.fabric.query("first query")
        self.fabric.query("second query")
        stats = self.fabric.cache_prefetch_stats()
        assert "lsh_enabled" in stats
        assert "prefetch_hits" in stats

    def test_telemetry_snapshot(self) -> None:
        self.fabric.search("test query", k=5)
        self.fabric.query("test telemetry")
        snap = self.fabric.telemetry_snapshot()
        assert "uptime_s" in snap
        assert "counters" in snap
        assert "latencies" in snap

    def test_search_batch_consistent_with_single(self) -> None:
        vecs = np.random.randn(50, 64).astype(np.float32)
        self.fabric.vector_engine.add(vecs)

        queries = ["query a", "query b", "query c"]
        batch = self.fabric.search_batch(queries, k=5)
        assert len(batch) == 3
        for row in batch:
            assert len(row) == 5

    def test_model_dump_on_dataclasses(self) -> None:
        """Regression: model_dump() was missing from dataclasses, breaking /stats."""
        cache_stats = self.fabric.cache_stats()
        sched_stats = self.fabric.scheduler_stats()
        mem_stats = self.fabric.memory_stats()

        assert isinstance(cache_stats.model_dump(), dict)
        assert isinstance(sched_stats.model_dump(), dict)
        assert isinstance(mem_stats.model_dump(), dict)

        assert "hits" in cache_stats.model_dump()
        assert "shares" in sched_stats.model_dump()
        assert "vram_total_mb" in mem_stats.model_dump()

    def test_set_llm_backend(self) -> None:
        called: list[str] = []

        def my_llm(prompt: str) -> str:
            called.append(prompt)
            return "custom response"

        self.fabric.set_llm_backend(my_llm)
        resp = self.fabric.query("unique question xyz")
        assert resp == "custom response"
        assert len(called) == 1


class TestVectorEngineShutdownBug:
    """Regression tests for shutdown() not resetting internal state."""

    def test_shutdown_resets_count(self) -> None:
        cfg = VectorIndexConfig(dim=32, index_type="flat", metric="cosine")
        engine = VectorEngine(config=cfg)

        vecs = np.random.randn(50, 32).astype(np.float32)
        engine.add(vecs)
        assert engine.count == 50

        engine.shutdown()
        assert engine.count == 0

    def test_shutdown_resets_ids(self) -> None:
        cfg = VectorIndexConfig(dim=32, index_type="flat", metric="cosine")
        engine = VectorEngine(config=cfg)

        engine.add(np.random.randn(10, 32).astype(np.float32))
        engine.shutdown()
        assert engine._ids == []

    def test_can_add_after_shutdown(self) -> None:
        cfg = VectorIndexConfig(dim=32, index_type="flat", metric="cosine")
        engine = VectorEngine(config=cfg)

        engine.add(np.random.randn(10, 32).astype(np.float32))
        engine.shutdown()
        assert engine.count == 0

        new_vecs = np.random.randn(5, 32).astype(np.float32)
        count = engine.add(new_vecs)
        assert count == 5


class TestVectorEngineLazyRebuild:
    """Tests for lazy index rebuild — index is built only before search."""

    def test_dirty_flag_set_on_add(self) -> None:
        cfg = VectorIndexConfig(dim=32, index_type="flat", metric="cosine")
        engine = VectorEngine(config=cfg)
        assert engine._dirty is False

        engine.add(np.random.randn(10, 32).astype(np.float32))
        assert engine._dirty is True

    def test_dirty_flag_cleared_after_search(self) -> None:
        cfg = VectorIndexConfig(dim=32, index_type="flat", metric="cosine")
        engine = VectorEngine(config=cfg)
        engine.add(np.random.randn(10, 32).astype(np.float32))
        assert engine._dirty is True

        engine.search(np.random.randn(32).astype(np.float32), k=3)
        assert engine._dirty is False

    def test_multiple_adds_single_rebuild(self) -> None:
        cfg = VectorIndexConfig(dim=32, index_type="flat", metric="cosine")
        engine = VectorEngine(config=cfg)

        # Three adds without search — only one rebuild should happen at search time
        for _ in range(3):
            engine.add(np.random.randn(10, 32).astype(np.float32))

        assert engine.count == 30
        assert engine._dirty is True

        results = engine.search(np.random.randn(32).astype(np.float32), k=5)
        assert len(results) == 5
        assert engine._dirty is False


class TestVectorEngineBatchSearch:
    """Tests for vectorized batch search."""

    def test_batch_matches_single(self) -> None:
        cfg = VectorIndexConfig(dim=64, index_type="flat", metric="cosine")
        engine = VectorEngine(config=cfg)

        vecs = np.random.randn(100, 64).astype(np.float32)
        engine.add(vecs)

        queries = np.random.randn(5, 64).astype(np.float32)
        batch_results = engine.search_batch(queries, k=3)
        assert len(batch_results) == 5

        for i, q in enumerate(queries):
            single = engine.search(q, k=3)
            assert len(batch_results[i]) == len(single)
            assert [r.id for r in batch_results[i]] == [r.id for r in single]

    def test_batch_empty_index_returns_empty_lists(self) -> None:
        cfg = VectorIndexConfig(dim=32)
        engine = VectorEngine(config=cfg)
        queries = np.random.randn(3, 32).astype(np.float32)
        results = engine.search_batch(queries, k=5)
        assert results == [[], [], []]

    def test_batch_l2_metric(self) -> None:
        cfg = VectorIndexConfig(dim=32, metric="l2")
        engine = VectorEngine(config=cfg)
        engine.add(np.random.randn(20, 32).astype(np.float32))

        queries = np.random.randn(4, 32).astype(np.float32)
        results = engine.search_batch(queries, k=3)
        assert len(results) == 4
        for row in results:
            assert len(row) == 3


class TestSchedulerPIController:
    """Tests for PI controller with integral term."""

    def test_integral_resets_on_policy_change(self) -> None:
        from vram_fabric.core.types import SchedulerPolicy
        from vram_fabric.scheduler.dynamic import DynamicScheduler

        sched = DynamicScheduler(SchedulerPolicy(tick_interval_ms=50))
        sched._integral["llm"] = 99.9

        sched.set_policy(shares={"llm": 0.60, "vector": 0.20, "agent": 0.10, "cache": 0.10})
        assert sched._integral.get("llm", 0.0) == 0.0

    def test_integral_accumulates_over_ticks(self) -> None:
        from vram_fabric.core.types import SchedulerPolicy
        from vram_fabric.scheduler.dynamic import DynamicScheduler

        sched = DynamicScheduler(SchedulerPolicy(tick_interval_ms=50))
        # Simulate occupancy always at 0 (max error = target)
        sched.register_engine("llm", lambda: 0.0)
        sched.register_engine("vector", lambda: 0.0)
        sched.register_engine("agent", lambda: 0.0)
        sched.register_engine("cache", lambda: 0.0)

        sched._tick()
        sched._tick()

        # After 2 ticks with occupancy=0 and target=0.70, integral should be positive
        assert sched._integral["llm"] > 0.0
