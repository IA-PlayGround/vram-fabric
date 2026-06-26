from __future__ import annotations

import numpy as np
import pytest

from vram_fabric.engines.cache_engine import SemanticCacheEngine


class TestSemanticCache:
    def setup_method(self) -> None:
        self.cache = SemanticCacheEngine(max_entries=100, dim=64, similarity_threshold=0.95)

    def test_insert_and_lookup_hit(self) -> None:
        emb = np.random.randn(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        self.cache.insert("test query", emb, "test response")
        response, sim = self.cache.lookup(emb)

        assert response == "test response"
        assert sim >= 0.99

    def test_lookup_miss(self) -> None:
        emb = np.random.randn(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        response, sim = self.cache.lookup(emb)
        assert response is None
        assert sim == 0.0

    def test_lookup_below_threshold_misses(self) -> None:
        cache = SemanticCacheEngine(max_entries=100, dim=64, similarity_threshold=0.99)

        emb1 = np.ones(64, dtype=np.float32) / 8.0
        emb2 = -np.ones(64, dtype=np.float32) / 8.0

        cache.insert("q1", emb1, "response1")
        response, sim = cache.lookup(emb2)

        assert response is None
        assert sim < 0.99

    def test_contains(self) -> None:
        emb = np.random.randn(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        self.cache.insert("hello world", emb, "reply")
        assert self.cache.contains("hello world")
        assert not self.cache.contains("unknown query")

    def test_lru_eviction(self) -> None:
        cache = SemanticCacheEngine(max_entries=5, dim=16, similarity_threshold=0.0)

        for i in range(7):
            emb = np.random.randn(16).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            cache.insert(f"q{i}", emb, f"r{i}")

        assert cache.size <= 5
        assert cache.stats().evictions >= 2

    def test_stats_accuracy(self) -> None:
        emb = np.random.randn(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        self.cache.insert("q1", emb, "r1")
        self.cache.lookup(emb)
        self.cache.lookup(np.random.randn(64).astype(np.float32))

        stats = self.cache.stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_ratio == 0.5

    def test_clear(self) -> None:
        emb = np.random.randn(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        self.cache.insert("q1", emb, "r1")
        self.cache.clear()
        assert self.cache.size == 0
        assert self.cache.stats().hits == 0

    def test_list_embedding_input(self) -> None:
        emb_list = [float(i) for i in range(64)]
        emb_norm = np.array(emb_list, dtype=np.float32)
        emb_norm = emb_norm / np.linalg.norm(emb_norm)

        self.cache.insert("list query", emb_list, "list response")
        response, _ = self.cache.lookup(emb_norm)
        assert response == "list response"

    def test_multiple_inserts_same_query(self) -> None:
        emb = np.random.randn(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        self.cache.insert("same query", emb, "first")
        self.cache.insert("same query", emb, "second")
        response, _ = self.cache.lookup(emb)
        assert response == "second"

    def test_shutdown(self) -> None:
        emb = np.random.randn(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        self.cache.insert("q", emb, "r")
        self.cache.shutdown()
        assert self.cache.size == 0
