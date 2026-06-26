from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class _DictMixin:
    """Adds a model_dump() helper to dataclasses for API serialization."""

    def model_dump(self) -> dict:
        return dataclasses.asdict(self)


class DeviceBackend(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"


class EngineType(str, Enum):
    LLM = "llm"
    VECTOR = "vector"
    AGENT = "agent"
    CACHE = "cache"


class SpilloverTier(str, Enum):
    VRAM = "vram"
    RAM = "ram"
    SSD = "ssd"


@dataclass
class SchedulerPolicy:
    shares: dict[str, float] = field(default_factory=lambda: {
        "llm": 0.70,
        "vector": 0.15,
        "agent": 0.10,
        "cache": 0.05,
    })
    target_occupancy: float = 0.70
    tick_interval_ms: int = 100


@dataclass
class MemoryStats(_DictMixin):
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    vram_free_mb: int = 0
    ram_spillover_mb: int = 0
    ssd_spillover_mb: int = 0
    spillover_count: int = 0


@dataclass
class VectorIndexConfig:
    dim: int = 768
    index_type: str = "flat"   # flat | ivf
    metric: str = "cosine"     # cosine | l2 | ip
    nlist: int = 100           # IVF clusters
    use_fp16: bool = True


@dataclass
class CacheStats(_DictMixin):
    max_entries: int = 0
    current_entries: int = 0
    hits: int = 0
    misses: int = 0
    hit_ratio: float = 0.0
    vram_used_mb: int = 0
    evictions: int = 0
    avg_lookup_time_ms: float = 0.0

    @property
    def size(self) -> int:
        return self.current_entries


@dataclass
class SchedulerStats(_DictMixin):
    shares: dict[str, float] = field(default_factory=dict)
    occupancy: dict[str, float] = field(default_factory=dict)
    vram_allocated: dict[str, int] = field(default_factory=dict)
    adjustments: int = 0
    throttled: bool = False


@dataclass
class SearchResult:
    id: int
    score: float
    vector: Any = None   # torch.Tensor | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentHandle:
    name: str
    system_prompt: str
    model: str = ""
    state_tensor: Any = None   # torch.Tensor | None


@dataclass
class AgentInstance:
    handle: AgentHandle
    stream_id: int = 0
    active: bool = True


@dataclass
class Task:
    agent_name: str
    prompt: str
    task_id: str = ""
    dependencies: list[str] = field(default_factory=list)


@dataclass
class Workflow:
    tasks: list[Task] = field(default_factory=list)
    dag: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    results: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    total_time_ms: float = 0.0
