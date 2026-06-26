from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml


@dataclass
class FabricConfig:
    # Device
    device: str = "cuda"
    gpu_id: int = 0
    vram_limit_mb: int = 0            # 0 = auto-detect
    ram_spillover_limit_mb: int = 8192

    # LLM
    llm_model: str = "llama3:8b"
    llm_backend: str = "llama.cpp"   # llama.cpp | vllm | transformers

    # Vector Engine
    vector_dim: int = 768
    vector_index_type: str = "flat"
    vector_metric: str = "cosine"
    vector_nlist: int = 100
    vector_fp16: bool = True

    # Cache Engine
    cache_enabled: bool = True
    cache_size_mb: int = 2048
    cache_max_entries: int = 100_000
    cache_similarity_threshold: float = 0.95
    cache_lru_cycle_max: int = 1_000_000

    # Agent Runtime
    agent_max_instances: int = 16
    agent_state_dim: int = 4096
    agent_hidden_dim: int = 4096

    # Scheduler
    scheduler_policy: str = "auto"    # auto | balanced | custom
    scheduler_tick_ms: int = 100
    scheduler_target_occupancy: float = 0.70

    # Shares (scheduler)
    share_llm: float = 0.70
    share_vector: float = 0.15
    share_agent: float = 0.10
    share_cache: float = 0.05

    # Telemetry
    telemetry_enabled: bool = False
    telemetry_port: int = 9090

    @classmethod
    def from_yaml(cls, path: str) -> FabricConfig:
        if os.path.exists(path):
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
        return cls()

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "gpu_id": self.gpu_id,
            "vram_limit_mb": self.vram_limit_mb,
            "llm_model": self.llm_model,
            "vector_dim": self.vector_dim,
            "cache_size_mb": self.cache_size_mb,
            "scheduler_policy": self.scheduler_policy,
        }
