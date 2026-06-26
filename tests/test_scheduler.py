from __future__ import annotations

import time

import pytest

from vram_fabric.core.types import SchedulerPolicy
from vram_fabric.scheduler.dynamic import DynamicScheduler


class TestDynamicScheduler:
    def setup_method(self) -> None:
        self.policy = SchedulerPolicy(
            shares={"llm": 0.70, "vector": 0.15, "agent": 0.10, "cache": 0.05},
            target_occupancy=0.70,
            tick_interval_ms=50,
        )
        self.scheduler = DynamicScheduler(policy=self.policy)

    def test_initial_shares(self) -> None:
        for engine, expected in self.policy.shares.items():
            assert abs(self.scheduler.get_share(engine) - expected) < 0.01

    def test_set_policy_updates_shares(self) -> None:
        new_shares = {"llm": 0.50, "vector": 0.25, "agent": 0.15, "cache": 0.10}
        self.scheduler.set_policy(shares=new_shares)

        for engine, expected in new_shares.items():
            assert abs(self.scheduler.get_share(engine) - expected) < 0.01

    def test_shares_normalized(self) -> None:
        self.scheduler.set_policy(shares={"llm": 1.0, "vector": 1.0, "agent": 1.0, "cache": 1.0})
        total = sum(self.scheduler.stats().shares.values())
        assert abs(total - 1.0) < 0.01

    def test_start_stop(self) -> None:
        self.scheduler.start()
        assert self.scheduler._running is True

        time.sleep(0.15)
        stats = self.scheduler.stats()
        assert stats.adjustments >= 1

        self.scheduler.stop()
        assert self.scheduler._running is False

    def test_register_engine_callback(self) -> None:
        called = []

        def mock_occupancy() -> float:
            called.append(True)
            return 0.75

        self.scheduler.register_engine("test_engine", mock_occupancy)

    def test_stats(self) -> None:
        stats = self.scheduler.stats()
        assert "llm" in stats.shares
        assert "vector" in stats.shares
        assert stats.adjustments >= 0
        assert isinstance(stats.throttled, bool)

    def test_shutdown(self) -> None:
        self.scheduler.start()
        self.scheduler.shutdown()
        assert self.scheduler._running is False
