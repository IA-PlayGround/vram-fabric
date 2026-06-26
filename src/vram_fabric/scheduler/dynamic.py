from __future__ import annotations

import logging
import threading
import time
from typing import Any

from vram_fabric.core.types import SchedulerPolicy, SchedulerStats
from vram_fabric.memory.pool import vram_pool

logger = logging.getLogger(__name__)

_MAX_SHARE = 0.80
_MIN_SHARE = 0.02


_Kp = 0.05
_Ki = 0.01


class DynamicScheduler:
    """Proportional-Integral (PI) controller for GPU resource distribution.

    Monitors SM occupancy and VRAM pressure to dynamically adjust
    engine shares: LLM, Vector, Agent, Cache.
    Feedback loop runs every tick_interval_ms.
    """

    def __init__(self, policy: SchedulerPolicy | None = None) -> None:
        self._policy = policy or SchedulerPolicy()
        self._shares: dict[str, float] = dict(self._policy.shares)
        self._occupancy: dict[str, float] = {e: 0.0 for e in self._shares}
        self._integral: dict[str, float] = {e: 0.0 for e in self._shares}
        self._vram_allocated: dict[str, int] = {e: 0 for e in self._shares}
        self._adjustments = 0
        self._throttled = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._engine_callbacks: dict[str, Any] = {}
        self._cuda_available = False

        self._init_cuda()

    def _init_cuda(self) -> None:
        try:
            import torch
            self._cuda_available = torch.cuda.is_available()
        except ImportError:
            pass

    def register_engine(self, name: str, occupancy_callback: Any) -> None:
        self._engine_callbacks[name] = occupancy_callback

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        tick_ms = self._policy.tick_interval_ms
        self._thread = threading.Thread(target=self._loop, args=(tick_ms / 1000,), daemon=True)
        self._thread.start()
        logger.info("Dynamic Scheduler started: tick=%dms", tick_ms)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self, interval_s: float) -> None:
        while self._running:
            time.sleep(interval_s)
            self._tick()

    def _tick(self) -> None:
        with self._lock:
            for engine in self._shares:
                occ = self._sample_occupancy(engine)
                self._occupancy[engine] = occ

                error = self._policy.target_occupancy - occ
                self._integral[engine] += error
                # PI controller: proportional + integral term
                adjustment = _Kp * error + _Ki * self._integral[engine]
                new_share = self._shares[engine] + adjustment
                self._shares[engine] = max(_MIN_SHARE, min(_MAX_SHARE, new_share))

            self._normalize_shares()
            self._adjustments += 1
            self._check_vram_pressure()

            if self._adjustments % 10 == 0:
                logger.debug(
                    "Scheduler tick #%d: shares=%s",
                    self._adjustments,
                    {k: f"{v:.2f}" for k, v in self._shares.items()},
                )

    def _sample_occupancy(self, engine: str) -> float:
        cb = self._engine_callbacks.get(engine)
        if cb and callable(cb):
            try:
                return float(cb())
            except Exception:
                pass
        return 0.0

    def _normalize_shares(self) -> None:
        total = sum(self._shares.values())
        if total > 0:
            for engine in self._shares:
                self._shares[engine] /= total

    def _check_vram_pressure(self) -> None:
        stats = vram_pool.stats()
        free_ratio = stats.free_bytes / max(stats.total_bytes, 1)
        if free_ratio < 0.10:
            self._throttled = True
            # Keep critical blocks (LLM weights and active agents); evict only cache/vector shards
            keep_keys = {k for k in vram_pool._blocks if k.startswith("agent_state_") or k == "llm_weights"}
            freed = vram_pool.evict_lru(keep_keys=keep_keys)
            logger.warning(
                "VRAM pressure: free=%.1f%%, evicted=%dMB, throttled=True",
                free_ratio * 100, freed // 1024**2,
            )
        else:
            self._throttled = False

    def get_share(self, engine: str) -> float:
        return self._shares.get(engine, 0.0)

    def set_policy(self, shares: dict[str, float] | None = None, target_occupancy: float | None = None) -> None:
        with self._lock:
            if shares:
                self._shares = dict(shares)
                self._normalize_shares()
                self._integral = {e: 0.0 for e in self._shares}
            if target_occupancy is not None:
                self._policy.target_occupancy = target_occupancy
            logger.info("Scheduler policy updated: shares=%s", self._shares)

    def stats(self) -> SchedulerStats:
        return SchedulerStats(
            shares=dict(self._shares),
            occupancy=dict(self._occupancy),
            vram_allocated=dict(self._vram_allocated),
            adjustments=self._adjustments,
            throttled=self._throttled,
        )

    def shutdown(self) -> None:
        self.stop()


scheduler = DynamicScheduler()
