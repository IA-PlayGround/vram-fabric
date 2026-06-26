from __future__ import annotations

import numpy as np
import pytest

from vram_fabric.core.types import VectorIndexConfig
from vram_fabric.engines.vector_engine import VectorEngine


class TestVectorEngine:
    def setup_method(self) -> None:
        cfg = VectorIndexConfig(dim=64, index_type="flat", metric="cosine")
        self.engine = VectorEngine(config=cfg)

    def test_add_and_count(self) -> None:
        vectors = np.random.randn(100, 64).astype(np.float32)
        count = self.engine.add(vectors)
        assert count == 100
        assert self.engine.count == 100

    def test_add_list(self) -> None:
        vectors = [[float(i) for i in range(64)] for _ in range(10)]
        count = self.engine.add(vectors)
        assert count == 10

    def test_search_returns_top_k(self) -> None:
        vectors = np.random.randn(200, 64).astype(np.float32)
        self.engine.add(vectors)

        query = np.random.randn(64).astype(np.float32)
        results = self.engine.search(query, k=5)
        assert len(results) == 5
        for r in results:
            assert 0 <= r.id < 200
            assert -1.0 <= r.score <= 1.0

    def test_search_empty_returns_empty(self) -> None:
        results = self.engine.search(np.random.randn(64).astype(np.float32), k=5)
        assert results == []

    def test_search_batch(self) -> None:
        vectors = np.random.randn(100, 64).astype(np.float32)
        self.engine.add(vectors)

        queries = np.random.randn(10, 64).astype(np.float32)
        results = self.engine.search_batch(queries, k=3)
        assert len(results) == 10
        for batch_result in results:
            assert len(batch_result) == 3

    def test_remove(self) -> None:
        vectors = np.random.randn(50, 64).astype(np.float32)
        self.engine.add(vectors)
        assert self.engine.count == 50

        new_count = self.engine.remove([0, 1, 2])
        assert new_count == 47

    def test_add_single_vector_reshaped(self) -> None:
        vec = np.random.randn(64).astype(np.float32)
        count = self.engine.add(vec)
        assert count == 1

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            self.engine.add(np.random.randn(10, 128).astype(np.float32))

    def test_stats(self) -> None:
        vectors = np.random.randn(50, 64).astype(np.float32)
        self.engine.add(vectors)
        stats = self.engine.stats()
        assert stats["count"] == 50
        assert stats["dim"] == 64
        assert stats["metric"] == "cosine"

    def test_shutdown(self) -> None:
        self.engine.shutdown()
        assert self.engine.count == 0
