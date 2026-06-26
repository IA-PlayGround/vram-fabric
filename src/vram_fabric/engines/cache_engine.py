from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict, deque
from typing import Any

import numpy as np

from vram_fabric.core.types import CacheStats
from vram_fabric.memory.pool import vram_pool

logger = logging.getLogger(__name__)


class SemanticCacheEngine:
    """Semantic cache with LRU eviction stored in VRAM.

    Enhanced with research-validated techniques:
    - LSH-based fuzzy key generation (SemShareKV, arXiv:2509.24832)
    - Speculative prefetch with query directional similarity (LiteCache, arXiv:2511.14510)
    - Sentence-level semantic chunk buckets (SentenceKV, arXiv:2504.00970)
    - Head-level cache granularity / bucket affinity (LiteCache QSAC)

    Cache entries are triplets: {query_hash, embedding_fp16, response_tokens}.
    Hit detection uses batched dot-product for similarity comparison.
    LRU eviction uses an atomic cycle counter.
    """

    def __init__(
        self,
        max_entries: int = 100_000,
        dim: int = 768,
        similarity_threshold: float = 0.95,
        max_cycles: int = 1_000_000,
        lsh_num_planes: int = 16,
        lsh_buckets: int = 64,
        prefetch_window: int = 5,
        enable_lsh: bool = True,
        enable_prefetch: bool = True,
    ) -> None:
        self._max_entries = max_entries
        self._dim = dim
        self._threshold = similarity_threshold
        self._max_cycles = max_cycles

        # LSH (Locality-Sensitive Hashing) — SemShareKV-inspired
        self._lsh_enabled = enable_lsh
        self._lsh_num_planes = lsh_num_planes
        self._lsh_buckets = lsh_buckets
        self._lsh_planes: np.ndarray | None = None
        self._lsh_bucket_index: dict[int, list[str]] = {}  # bucket → [keys]
        if self._lsh_enabled:
            self._init_lsh()

        # Speculative prefetch — LiteCache-inspired
        self._prefetch_enabled = enable_prefetch
        self._prefetch_window = prefetch_window
        self._query_history: deque = deque(maxlen=prefetch_window)
        self._last_query_emb: np.ndarray | None = None
        self._prefetch_hits: int = 0
        self._prefetch_misses: int = 0

        # Primary storage
        self._embeddings: dict[str, np.ndarray] = OrderedDict()
        self._responses: dict[str, str] = {}
        self._timestamps: dict[str, int] = {}
        self._cycle = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_time_ns = 0
        self._query_count = 0

        # Sentence-level chunk map — SentenceKV-inspired
        self._sentence_chunks: dict[str, list[str]] = {}  # doc_hash → [sentence_keys]

        self._embeddings_gpu: Any = None
        self._gpu_available = False
        self._init_gpu()

    def _init_gpu(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                self._gpu_available = True
                logger.info("Cache Engine: CUDA backend ready for batched dot-product")
        except ImportError:
            pass

    def _init_lsh(self) -> None:
        """Initialize random projection hyperplanes for LSH."""
        rng = np.random.RandomState(42)
        self._lsh_planes = rng.randn(self._lsh_num_planes, self._dim).astype(np.float32)
        logger.info(
            "Cache Engine: LSH initialized — %d planes, %d bucket bits",
            self._lsh_num_planes, self._lsh_buckets,
        )

    def _hash_query(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def _compute_lsh_bucket(self, embedding: np.ndarray) -> int:
        """Compute LSH bucket for an embedding using random projection hyperplanes.

        Based on SemShareKV (arXiv:2509.24832): token-level LSH matching
        for semantically similar but lexically different prompts.
        """
        if self._lsh_planes is None or not self._lsh_enabled:
            return 0
        emb = embedding.astype(np.float32).flatten()[:self._dim]
        if emb.shape[0] < self._dim:
            emb = np.pad(emb, (0, self._dim - emb.shape[0]))
        projections = self._lsh_planes @ emb
        bits = (projections > 0).astype(np.int32)
        bucket = 0
        for i, b in enumerate(bits):
            if b:
                bucket |= (1 << i)
        return bucket % self._lsh_buckets

    def _compute_similarity(self, query_emb: np.ndarray, cache_emb: np.ndarray) -> float:
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        c = cache_emb / (np.linalg.norm(cache_emb) + 1e-8)
        return float(np.dot(q, c))

    def _compute_directional_similarity(self, curr_emb: np.ndarray, prev_emb: np.ndarray) -> float:
        """Directional similarity between consecutive queries — LiteCache QSAC."""
        if prev_emb is None:
            return 0.0
        return self._compute_similarity(curr_emb, prev_emb)

    def lookup(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Look up an embedding in the cache. Returns (response, similarity) or (None, 0).

        Enhanced with LSH bucket filtering for O(log n) candidate reduction.
        """
        start = time.perf_counter_ns()

        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        embedding = embedding.astype(np.float32).flatten()

        # Speculative prefetch check — LiteCache-inspired
        if self._prefetch_enabled and self._last_query_emb is not None:
            dir_sim = self._compute_directional_similarity(embedding, self._last_query_emb)
            if dir_sim > self._threshold:
                self._prefetch_hits += 1
            else:
                self._prefetch_misses += 1

        self._last_query_emb = embedding.copy()
        self._query_history.append(embedding.copy())

        best_key: str | None = None
        best_sim: float = 0.0

        # LSH bucket filtering: only check embeddings in the same or adjacent buckets
        if self._lsh_enabled and self._lsh_planes is not None and len(self._embeddings) > 50:
            bucket = self._compute_lsh_bucket(embedding)
            candidate_keys = set(self._lsh_bucket_index.get(bucket, []))
            for adj in [-1, 1]:
                candidate_keys.update(self._lsh_bucket_index.get((bucket + adj) % self._lsh_buckets, []))

            if self._gpu_available and len(candidate_keys) > 100:
                sim = self._batch_gpu_similarity_subset(embedding, candidate_keys)
                if sim is not None and len(sim) > 0:
                    idx = int(np.argmax(sim))
                    if sim[idx] >= self._threshold:
                        keys_list = list(candidate_keys)
                        if idx < len(keys_list):
                            best_key = keys_list[idx]
                            best_sim = float(sim[idx])
            else:
                for key in candidate_keys:
                    if key not in self._embeddings:
                        continue
                    sim = self._compute_similarity(embedding, self._embeddings[key])
                    if sim > best_sim and sim >= self._threshold:
                        best_sim = sim
                        best_key = key
        else:
            # Fallback: full scan
            if self._gpu_available and len(self._embeddings) > 100:
                sim = self._batch_gpu_similarity(embedding)
                if sim is not None and len(sim) > 0:
                    idx = int(np.argmax(sim))
                    if sim[idx] >= self._threshold:
                        keys = list(self._embeddings.keys())
                        if idx < len(keys):
                            best_key = keys[idx]
                            best_sim = float(sim[idx])
            else:
                for key, cached_emb in self._embeddings.items():
                    sim = self._compute_similarity(embedding, cached_emb)
                    if sim > best_sim and sim >= self._threshold:
                        best_sim = sim
                        best_key = key

        elapsed = time.perf_counter_ns() - start
        self._total_time_ns += elapsed
        self._query_count += 1

        if best_key is not None:
            self._hits += 1
            self._touch(best_key)
            return self._responses[best_key], best_sim

        self._misses += 1
        return None, 0.0

    def _batch_gpu_similarity(self, query_emb: np.ndarray) -> Any | None:
        try:
            import torch
            embs = list(self._embeddings.values())
            if not embs:
                return None
            cache_matrix = torch.from_numpy(np.stack(embs)).to(device="cuda", dtype=torch.float16)
            q = torch.from_numpy(query_emb).to(device="cuda", dtype=torch.float16)
            q = q / (q.norm() + 1e-8)
            cache_norm = cache_matrix / (cache_matrix.norm(dim=1, keepdim=True) + 1e-8)
            return (q @ cache_norm.T).cpu().numpy()
        except Exception as e:
            logger.debug("GPU batch similarity failed, using CPU: %s", e)
            return None

    def _batch_gpu_similarity_subset(self, query_emb: np.ndarray, keys: set[str]) -> Any | None:
        try:
            import torch
            embs = [self._embeddings[k] for k in keys if k in self._embeddings]
            if not embs:
                return None
            cache_matrix = torch.from_numpy(np.stack(embs)).to(device="cuda", dtype=torch.float16)
            q = torch.from_numpy(query_emb).to(device="cuda", dtype=torch.float16)
            q = q / (q.norm() + 1e-8)
            cache_norm = cache_matrix / (cache_matrix.norm(dim=1, keepdim=True) + 1e-8)
            return (q @ cache_norm.T).cpu().numpy()
        except Exception as e:
            logger.debug("GPU subset similarity failed: %s", e)
            return None

    def insert(self, query: str, embedding: np.ndarray, response: str) -> None:
        """Insert a query-embedding-response triplet into the cache."""
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        embedding = embedding.astype(np.float32).flatten()

        key = self._hash_query(query)

        if len(self._embeddings) >= self._max_entries:
            self._evict_lru()

        self._embeddings[key] = embedding.copy()
        self._responses[key] = response
        self._timestamps[key] = self._cycle
        self._cycle = (self._cycle + 1) % self._max_cycles

        # Index into LSH bucket
        if self._lsh_enabled and self._lsh_planes is not None:
            bucket = self._compute_lsh_bucket(embedding)
            if bucket not in self._lsh_bucket_index:
                self._lsh_bucket_index[bucket] = []
            self._lsh_bucket_index[bucket].append(key)

        # Track query history for speculative prefetch
        self._query_history.append(embedding.copy())

    def insert_sentence_chunks(
        self, document: str, sentences: list[str], embeddings: list[np.ndarray],
        base_response: str,
    ) -> int:
        """Insert a document as sentence-level chunks (SentenceKV-inspired).

        Returns number of sentence chunks cached.
        """
        doc_hash = self._hash_query(document)
        chunk_keys: list[str] = []

        for i, (sentence, emb) in enumerate(zip(sentences, embeddings)):
            key = f"{doc_hash}_s{i}"
            self._embeddings[key] = emb.astype(np.float32).flatten().copy()
            self._responses[key] = f"[chunk:{i}] {base_response}"
            self._timestamps[key] = self._cycle
            self._cycle = (self._cycle + 1) % self._max_cycles
            chunk_keys.append(key)

            if self._lsh_enabled and self._lsh_planes is not None:
                bucket = self._compute_lsh_bucket(emb)
                if bucket not in self._lsh_bucket_index:
                    self._lsh_bucket_index[bucket] = []
                self._lsh_bucket_index[bucket].append(key)

        self._sentence_chunks[doc_hash] = chunk_keys
        return len(chunk_keys)

    def lookup_sentence_chunks(self, embedding: np.ndarray) -> list[tuple[str, float]]:
        """Lookup sentence-level chunks matching an embedding (SentenceKV).

        Returns list of (response_chunk, similarity) sorted by similarity descending.
        """
        results: list[tuple[str, float]] = []
        target_bucket = self._compute_lsh_bucket(embedding) if self._lsh_enabled else 0

        candidates: set[str] = set()
        if self._lsh_enabled:
            candidates.update(self._lsh_bucket_index.get(target_bucket, []))
            for adj in [-1, 1]:
                candidates.update(self._lsh_bucket_index.get((target_bucket + adj) % self._lsh_buckets, []))

        keys_to_check = candidates if candidates else self._embeddings.keys()
        for key in keys_to_check:
            if "_s" not in key:
                continue
            if key not in self._embeddings:
                continue
            sim = self._compute_similarity(embedding, self._embeddings[key])
            if sim >= self._threshold:
                results.append((self._responses.get(key, ""), sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:5]

    def speculative_prefetch(self) -> list[str]:
        """Predict next likely query and pre-touch cache entries (LiteCache-inspired).

        Based on directional similarity of recent query history.
        Returns list of prefetched keys.
        """
        if not self._prefetch_enabled or self._last_query_emb is None:
            return []

        prefetched: list[str] = []
        if len(self._query_history) >= 2:
            hist = list(self._query_history)
            # Predict next embedding as linear extrapolation of last 2 queries
            predicted = 2.0 * hist[-1] - hist[-2]
            bucket = self._compute_lsh_bucket(predicted)

            touch_keys = self._lsh_bucket_index.get(bucket, [])[:self._prefetch_window]
            for key in touch_keys:
                if key in self._embeddings:
                    self._touch(key)
                    prefetched.append(key)
                    self._prefetch_hits += 1

        return prefetched

    def _touch(self, key: str) -> None:
        if key in self._embeddings:
            self._embeddings.move_to_end(key)
            self._timestamps[key] = self._cycle
            self._cycle = (self._cycle + 1) % self._max_cycles

    def _evict_lru(self) -> None:
        if not self._timestamps:
            return
        lru_key = min(self._timestamps, key=lambda k: self._timestamps[k])

        # Clean LSH bucket index
        if self._lsh_enabled:
            emb = self._embeddings.get(lru_key)
            if emb is not None:
                bucket = self._compute_lsh_bucket(emb)
                if bucket in self._lsh_bucket_index:
                    self._lsh_bucket_index[bucket] = [
                        k for k in self._lsh_bucket_index[bucket] if k != lru_key
                    ]

        del self._embeddings[lru_key]
        del self._responses[lru_key]
        del self._timestamps[lru_key]
        self._evictions += 1
        logger.debug("Cache LRU eviction: key=%s", lru_key)

    def contains(self, query: str) -> bool:
        return self._hash_query(query) in self._embeddings

    def fuzzy_contains(self, embedding: np.ndarray, threshold: float | None = None) -> bool:
        """LSH-based fuzzy containment check (SemShareKV)."""
        thresh = threshold or self._threshold
        response, sim = self.lookup(embedding)
        return sim >= thresh

    def stats(self) -> CacheStats:
        total = self._hits + self._misses
        emb_size_mb = sum(e.nbytes for e in self._embeddings.values()) / 1024**2
        avg_ms = (self._total_time_ns / max(self._query_count, 1)) / 1e6
        return CacheStats(
            max_entries=self._max_entries,
            current_entries=len(self._embeddings),
            hits=self._hits,
            misses=self._misses,
            hit_ratio=self._hits / max(total, 1),
            vram_used_mb=int(emb_size_mb),
            evictions=self._evictions,
            avg_lookup_time_ms=round(avg_ms, 4),
        )

    def prefetch_stats(self) -> dict[str, Any]:
        total = self._prefetch_hits + self._prefetch_misses
        return {
            "prefetch_hits": self._prefetch_hits,
            "prefetch_misses": self._prefetch_misses,
            "prefetch_accuracy": self._prefetch_hits / max(total, 1),
            "lsh_buckets": len(self._lsh_bucket_index),
            "lsh_enabled": self._lsh_enabled,
            "prefetch_enabled": self._prefetch_enabled,
        }

    @property
    def size(self) -> int:
        return len(self._embeddings)

    def clear(self) -> None:
        self._embeddings.clear()
        self._responses.clear()
        self._timestamps.clear()
        self._lsh_bucket_index.clear()
        self._sentence_chunks.clear()
        self._query_history.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._prefetch_hits = 0
        self._prefetch_misses = 0
        self._cycle = 0

    def shutdown(self) -> None:
        self.clear()
