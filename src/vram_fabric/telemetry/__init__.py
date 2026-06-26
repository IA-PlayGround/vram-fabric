"""VRAM Fabric Telemetry — lightweight in-process metrics collector.

Exposes counters and gauges for:
- Cache hit/miss rate
- Vector search latency (p50, p95, p99)
- Scheduler adjustments
- VRAM utilization

Usage:
    from vram_fabric.telemetry import Telemetry
    t = Telemetry()
    t.record_latency("vector_search", elapsed_ms)
    t.snapshot()  # → dict with all current metrics
"""
from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Any


class _LatencyTracker:
    """Ring-buffer latency tracker; computes p50/p95/p99 over last N samples."""

    def __init__(self, maxlen: int = 1000) -> None:
        self._buf: deque[float] = deque(maxlen=maxlen)

    def record(self, value_ms: float) -> None:
        self._buf.append(value_ms)

    def percentile(self, p: float) -> float:
        if not self._buf:
            return 0.0
        sorted_vals = sorted(self._buf)
        idx = max(0, int(len(sorted_vals) * p / 100) - 1)
        return sorted_vals[idx]

    def mean(self) -> float:
        if not self._buf:
            return 0.0
        return sum(self._buf) / len(self._buf)

    def count(self) -> int:
        return len(self._buf)


class Telemetry:
    """In-process metrics collector for VRAM Fabric components.

    Thread-safe. All times are in milliseconds.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._latencies: dict[str, _LatencyTracker] = {}
        self._start_time = time.monotonic()

    def increment(self, name: str, delta: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + delta

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def record_latency(self, name: str, elapsed_ms: float) -> None:
        with self._lock:
            if name not in self._latencies:
                self._latencies[name] = _LatencyTracker()
            self._latencies[name].record(elapsed_ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = {
                "uptime_s": round(time.monotonic() - self._start_time, 1),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "latencies": {
                    name: {
                        "count": tracker.count(),
                        "mean_ms": round(tracker.mean(), 3),
                        "p50_ms": round(tracker.percentile(50), 3),
                        "p95_ms": round(tracker.percentile(95), 3),
                        "p99_ms": round(tracker.percentile(99), 3),
                    }
                    for name, tracker in self._latencies.items()
                },
            }
        return result

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._latencies.clear()
            self._start_time = time.monotonic()


# Global singleton — optional; components can also instantiate their own.
telemetry = Telemetry()
