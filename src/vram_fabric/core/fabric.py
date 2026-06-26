from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from vram_fabric.config import FabricConfig
from vram_fabric.core.types import (
    AgentHandle,
    AgentInstance,
    CacheStats,
    MemoryStats,
    SchedulerPolicy,
    SchedulerStats,
    SearchResult,
    Task,
    VectorIndexConfig,
    Workflow,
    WorkflowResult,
)
from vram_fabric.engines.agent_runtime import AgentRuntime
from vram_fabric.engines.cache_engine import SemanticCacheEngine
from vram_fabric.engines.vector_engine import VectorEngine
from vram_fabric.memory.pool import VRAMMemoryPool, vram_pool
from vram_fabric.scheduler.dynamic import DynamicScheduler, scheduler as global_scheduler
from vram_fabric.telemetry import Telemetry

logger = logging.getLogger(__name__)


class VRAMFabric:
    """VRAM Fabric — Deep Tech Runtime for Local AI.

    Main orchestrator that ties together:
    - Vector Engine (FAISS GPU ANN search in VRAM)
    - Semantic Cache (LRU cache with CUDA batched dot-product)
    - Agent Runtime (actors with state tensors in VRAM)
    - Dynamic Scheduler (PI controller for GPU resource distribution)
    - Memory Pool (VRAM allocation with RAM/SSD spillover)

    Usage:
        fabric = VRAMFabric()
        fabric.index_documents(docs)
        response = fabric.query("What is AI?")
        fabric.shutdown()
    """

    def __init__(
        self,
        llm_model: str = "llama3:8b",
        vector_dim: int = 768,
        cache_size_mb: int = 2048,
        scheduler_policy: str = "auto",
        config: FabricConfig | None = None,
    ) -> None:
        self._config = config or FabricConfig(
            llm_model=llm_model,
            vector_dim=vector_dim,
            cache_size_mb=cache_size_mb,
            scheduler_policy=scheduler_policy,
        )

        self._setup_logging()

        # Memory pool
        vram_pool.setup_cuda()
        self._pool = vram_pool

        # Engines
        vec_cfg = VectorIndexConfig(
            dim=self._config.vector_dim,
            index_type=self._config.vector_index_type,
            metric=self._config.vector_metric,
            nlist=self._config.vector_nlist,
            use_fp16=self._config.vector_fp16,
        )
        self.vector_engine = VectorEngine(config=vec_cfg)

        cache_max = max(1000, (self._config.cache_size_mb * 1024**2) // (self._config.vector_dim * 2 + 256))
        self.cache_engine = SemanticCacheEngine(
            max_entries=min(cache_max, self._config.cache_max_entries),
            dim=self._config.vector_dim,
            similarity_threshold=self._config.cache_similarity_threshold,
        )

        self.agent_runtime = AgentRuntime(
            max_instances=self._config.agent_max_instances,
            hidden_dim=self._config.agent_hidden_dim,
        )

        # Scheduler
        policy = SchedulerPolicy(
            shares={
                "llm": self._config.share_llm,
                "vector": self._config.share_vector,
                "agent": self._config.share_agent,
                "cache": self._config.share_cache,
            },
            target_occupancy=self._config.scheduler_target_occupancy,
            tick_interval_ms=self._config.scheduler_tick_ms,
        )
        self.scheduler = DynamicScheduler(policy=policy)
        self.scheduler.start()

        # LLM backend (lazy init)
        self._llm_fn: Any = None
        self._llm_model = self._config.llm_model

        # Telemetry
        self.telemetry = Telemetry()

        logger.info("VRAM Fabric initialized: model=%s dim=%d cache=%dMB",
                     self._llm_model, vector_dim, cache_size_mb)

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

    # ── Vector Engine ──────────────────────────────────────────

    def index_documents(self, documents: list[str], model: str = "bge-large") -> int:
        embeddings = self._embed_documents(documents, model)
        return self.vector_engine.add(embeddings)

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        embedding = self._embed_query(query)
        t0 = time.perf_counter()
        results = self.vector_engine.search(embedding, k=k)
        self.telemetry.record_latency("vector_search", (time.perf_counter() - t0) * 1000)
        self.telemetry.increment("vector_search_total")
        return results

    def search_batch(self, queries: list[str], k: int = 10) -> list[list[SearchResult]]:
        embeddings = [self._embed_query(q) for q in queries]
        t0 = time.perf_counter()
        results = self.vector_engine.search_batch(embeddings, k=k)
        self.telemetry.record_latency("vector_search_batch", (time.perf_counter() - t0) * 1000)
        self.telemetry.increment("vector_search_total", len(queries))
        return results

    # ── Semantic Cache ─────────────────────────────────────────

    def query(self, prompt: str) -> str:
        """Unified query: cache hit → return, cache miss → LLM → cache → return."""
        embedding = self._embed_query(prompt)

        t0 = time.perf_counter()
        cached, sim = self.cache_engine.lookup(embedding)
        self.telemetry.record_latency("cache_lookup", (time.perf_counter() - t0) * 1000)

        if cached is not None:
            logger.debug("Cache HIT: sim=%.3f", sim)
            self.telemetry.increment("cache_hits")
            return cached

        logger.debug("Cache MISS: invoking LLM")
        self.telemetry.increment("cache_misses")
        response = self._invoke_llm(prompt)
        self.cache_engine.insert(prompt, embedding, response)
        return response

    def telemetry_snapshot(self) -> dict:
        """Return current telemetry snapshot with all counters, gauges, and latency percentiles."""
        snap = self.telemetry.snapshot()
        snap["cache"] = {
            "hit_ratio": self.cache_engine.stats().hit_ratio,
            "entries": self.cache_engine.size,
        }
        snap["scheduler"] = {
            "adjustments": self.scheduler.stats().adjustments,
            "throttled": self.scheduler.stats().throttled,
        }
        return snap

    def cache_stats(self) -> CacheStats:
        return self.cache_engine.stats()

    def cache_prefetch_stats(self) -> dict:
        return self.cache_engine.prefetch_stats()

    def cache_fuzzy_contains(self, query: str) -> bool:
        emb = self._embed_query(query)
        return self.cache_engine.fuzzy_contains(emb)

    def cache_speculative_prefetch(self) -> list[str]:
        return self.cache_engine.speculative_prefetch()

    def index_sentence_chunks(self, document: str, sentences: list[str], base_response: str = "") -> int:
        embeddings = [self._embed_query(s) for s in sentences]
        return self.cache_engine.insert_sentence_chunks(document, sentences, embeddings, base_response)

    def search_sentence_chunks(self, query: str) -> list[tuple[str, float]]:
        emb = self._embed_query(query)
        return self.cache_engine.lookup_sentence_chunks(emb)

    # ── Agent Runtime ──────────────────────────────────────────

    def register_agent(self, name: str, system_prompt: str, model: str | None = None) -> AgentHandle:
        if self._llm_fn:
            self.agent_runtime.set_llm_backend(self._llm_fn)
        return self.agent_runtime.register(name, system_prompt, model or self._llm_model)

    def spawn_agent(self, name: str) -> AgentInstance:
        return self.agent_runtime.spawn(name)

    def task(self, agent_name: str, prompt: str, task_id: str | None = None) -> Task:
        tid = task_id or f"{agent_name}_{uuid.uuid4().hex[:8]}"
        return Task(agent_name=agent_name, prompt=prompt, task_id=tid)

    def create_workflow(self, tasks: list[Task]) -> Workflow:
        dag: dict[str, list[str]] = {}
        for t in tasks:
            dag[t.task_id or t.agent_name] = list(t.dependencies)
        return Workflow(tasks=tasks, dag=dag)

    def run(self, workflow: Workflow) -> WorkflowResult:
        return self.agent_runtime.run_workflow(workflow)

    # ── Scheduler ─────────────────────────────────────────────

    def set_scheduler_policy(self, shares: dict[str, float] | None = None) -> None:
        self.scheduler.set_policy(shares=shares)

    def scheduler_stats(self) -> SchedulerStats:
        return self.scheduler.stats()

    def agent_admission_stats(self) -> dict:
        return self.agent_runtime.admission_stats()

    def update_cache_pressure(self) -> None:
        """Update agent runtime with current cache pressure for admission control."""
        stats = self.cache_engine.stats()
        total = stats.hits + stats.misses
        pressure = 1.0 - (stats.hits / max(total, 1)) if total > 0 else 0.0
        self.agent_runtime.set_cache_pressure(pressure)

    # ── Memory ────────────────────────────────────────────────

    def memory_stats(self) -> MemoryStats:
        s = self._pool.stats()
        return MemoryStats(
            vram_total_mb=s.total_bytes // 1024**2,
            vram_used_mb=s.used_bytes // 1024**2,
            vram_free_mb=s.free_bytes // 1024**2,
            ram_spillover_mb=s.spillover_ram_bytes // 1024**2,
            ssd_spillover_mb=s.spillover_ssd_bytes // 1024**2,
            spillover_count=s.blocks,
        )

    # ── Internal ───────────────────────────────────────────────

    def _embed_query(self, text: str) -> Any:
        """Simple hash-based embedding (placeholder for real embedder)."""
        import hashlib
        import numpy as np

        h = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(h * (self._config.vector_dim // 32 + 1), dtype=np.uint8)[:self._config.vector_dim].astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec

    def _embed_documents(self, documents: list[str], model: str = "bge-large") -> Any:
        return [self._embed_query(doc) for doc in documents]

    def _invoke_llm(self, prompt: str) -> str:
        if self._llm_fn:
            try:
                return self._llm_fn(prompt)
            except Exception:
                pass
        return f"[LLM:{self._llm_model} placeholder] Response to: {prompt[:100]}"

    def set_llm_backend(self, fn: Any) -> None:
        """Set a custom LLM inference function: fn(prompt: str) -> str."""
        self._llm_fn = fn
        self.agent_runtime.set_llm_backend(fn)
        logger.info("Custom LLM backend registered")

    # ── Lifecycle ──────────────────────────────────────────────

    def shutdown(self) -> None:
        self.scheduler.shutdown()
        self.vector_engine.shutdown()
        self.cache_engine.shutdown()
        self.agent_runtime.shutdown()
        self._pool.shutdown()
        logger.info("VRAM Fabric shut down")
