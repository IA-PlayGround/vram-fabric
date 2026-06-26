from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from vram_fabric.core.types import SearchResult, VectorIndexConfig
from vram_fabric.memory.pool import MemoryBlock, vram_pool

logger = logging.getLogger(__name__)


class VectorEngine:
    """GPU-accelerated vector indexing and ANN search.
    
    Uses FAISS GPU when available, falls back to CPU numpy.
    Supports Flat (brute-force cuBLAS batched matmul) and IVF indices.
    """

    def __init__(self, config: VectorIndexConfig | None = None) -> None:
        self._config = config or VectorIndexConfig()
        self._index: Any = None
        self._vectors: Any = None         # numpy ndarray | torch Tensor
        self._ids: list[int] = []
        self._count = 0
        self._dim = self._config.dim
        self._memory_block: MemoryBlock | None = None
        self._faiss_available = False
        self._torch_available = False
        self._device = "cpu"
        self._dirty = False

        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                self._torch_available = True
                self._device = "cuda"
                logger.info("Vector Engine: PyTorch CUDA backend available")
            else:
                logger.info("Vector Engine: PyTorch CPU backend")
        except ImportError:
            logger.info("Vector Engine: NumPy-only backend")

        try:
            import faiss
            if hasattr(faiss, "StandardGpuResources"):
                self._faiss_available = True
                self._gpu_resources = faiss.StandardGpuResources()
                logger.info("Vector Engine: FAISS GPU available — IVF/Flat indices on CUDA")
        except ImportError:
            logger.info("Vector Engine: FAISS not available — using numpy torch matmul")

    def add(self, vectors: Any, ids: list[int] | None = None) -> int:
        """Add vectors to the index. Accepts numpy array or list of lists."""
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float32)
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self._dim:
            raise ValueError(f"Vector dim mismatch: expected {self._dim}, got {vectors.shape[1]}")

        n = vectors.shape[0]
        if ids is None:
            ids = list(range(self._count, self._count + n))

        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])

        self._ids.extend(ids)
        self._count += n
        self._dirty = True

        return self._count

    def _allocate_vram(self) -> None:
        try:
            import torch
            size = self._vectors.nbytes
            tensor = torch.from_numpy(self._vectors).to(
                dtype=torch.float16 if self._config.use_fp16 else torch.float32,
                device="cuda",
            )
            self._vectors_gpu = tensor
            self._memory_block = vram_pool.allocate(
                "vector_index", size, owner="vector_engine"
            )
            logger.info("Vector index cached in VRAM: %.1fMB", size / 1024**2)
        except Exception as e:
            logger.debug("VRAM allocation for vector index skipped: %s", e)

    def _build_index(self) -> None:
        if self._faiss_available and self._count >= 100:
            try:
                import faiss
                if self._config.index_type == "ivf":
                    quantizer = faiss.IndexFlatIP(self._dim)
                    self._index = faiss.IndexIVFFlat(quantizer, self._dim, self._config.nlist)
                    if not self._index.is_trained:
                        self._index.train(np.ascontiguousarray(self._vectors.astype(np.float32)))
                else:
                    if self._config.metric == "cosine":
                        self._index = faiss.IndexFlatIP(self._dim)
                    elif self._config.metric == "l2":
                        self._index = faiss.IndexFlatL2(self._dim)
                    else:
                        self._index = faiss.IndexFlatIP(self._dim)

                if hasattr(self, "_gpu_resources") and self._faiss_available:
                    self._index = faiss.index_cpu_to_gpu(
                        self._gpu_resources, 0, self._index
                    )

                data = np.ascontiguousarray(self._vectors.astype(np.float32))
                self._index.add(data)
                logger.info("FAISS GPU index built: %d vectors", self._count)
            except Exception as e:
                logger.warning("FAISS index build failed, using fallback: %s", e)
                self._index = None

    def _ensure_index(self) -> None:
        """Rebuild index only when vectors changed since last build."""
        if self._dirty and self._count > 0:
            self._index = None
            self._build_index()
            if self._torch_available:
                self._allocate_vram()
            self._dirty = False

    def search(self, query: Any, k: int = 10) -> list[SearchResult]:
        """Search the index for top-k nearest neighbors."""
        if isinstance(query, list):
            query = np.array(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if self._count == 0:
            return []

        self._ensure_index()
        k = min(k, self._count)
        scores, indices = self._do_search(query, k)
        results: list[SearchResult] = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx < 0 or idx >= len(self._ids):
                continue
            results.append(SearchResult(
                id=self._ids[idx],
                score=float(scores[0][i]),
            ))
        return results

    def _do_search(self, query: np.ndarray, k: int) -> tuple[Any, Any]:
        q_contig = np.ascontiguousarray(query.astype(np.float32))

        if self._index is not None:
            try:
                return self._index.search(q_contig, k)
            except Exception:
                pass

        # Fallback: cosine similarity via dot product
        vecs = self._vectors.astype(np.float32)
        if self._config.metric == "cosine":
            q_norm = q_contig / (np.linalg.norm(q_contig, axis=1, keepdims=True) + 1e-8)
            v_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
            scores = np.dot(q_norm, v_norm.T)
        elif self._config.metric == "l2":
            diff = q_contig[:, np.newaxis, :] - vecs[np.newaxis, :, :]
            scores = -np.linalg.norm(diff, axis=2)
        else:
            scores = np.dot(q_contig, vecs.T)

        top_k = np.argpartition(-scores[0], k)[:k]
        top_k = top_k[np.argsort(-scores[0][top_k])]
        return scores[0][top_k].reshape(1, -1), top_k.reshape(1, -1)

    def search_batch(self, queries: Any, k: int = 10) -> list[list[SearchResult]]:
        """Vectorized batch search — one matmul for all queries."""
        if isinstance(queries, list):
            queries = np.array(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        if self._count == 0:
            return [[] for _ in range(len(queries))]

        self._ensure_index()
        k_clamped = min(k, self._count)

        if self._index is not None:
            try:
                q_contig = np.ascontiguousarray(queries.astype(np.float32))
                scores_mat, indices_mat = self._index.search(q_contig, k_clamped)
                results: list[list[SearchResult]] = []
                for row_scores, row_indices in zip(scores_mat, indices_mat):
                    row: list[SearchResult] = []
                    for score, idx in zip(row_scores, row_indices):
                        if 0 <= idx < len(self._ids):
                            row.append(SearchResult(id=self._ids[int(idx)], score=float(score)))
                    results.append(row)
                return results
            except Exception:
                pass

        # Vectorized NumPy fallback: one matmul for all queries
        vecs = self._vectors.astype(np.float32)
        q = queries.astype(np.float32)
        if self._config.metric == "cosine":
            q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
            v_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
            scores_all = q_norm @ v_norm.T
        elif self._config.metric == "l2":
            diff = q[:, np.newaxis, :] - vecs[np.newaxis, :, :]
            scores_all = -np.linalg.norm(diff, axis=2)
        else:
            scores_all = q @ vecs.T

        out: list[list[SearchResult]] = []
        for row_scores in scores_all:
            top_k = np.argpartition(-row_scores, k_clamped)[:k_clamped]
            top_k = top_k[np.argsort(-row_scores[top_k])]
            row = [
                SearchResult(id=self._ids[int(i)], score=float(row_scores[i]))
                for i in top_k if 0 <= i < len(self._ids)
            ]
            out.append(row)
        return out

    def remove(self, ids: list[int]) -> int:
        mask = np.isin(np.array(self._ids), ids, invert=True)
        self._vectors = self._vectors[mask]
        self._ids = [self._ids[i] for i in range(len(self._ids)) if mask[i]]
        self._count = len(self._ids)
        self._index = None
        self._build_index()
        return self._count

    @property
    def count(self) -> int:
        return self._count

    def stats(self) -> dict:
        return {
            "count": self._count,
            "dim": self._dim,
            "index_type": self._config.index_type,
            "metric": self._config.metric,
            "fp16": self._config.use_fp16,
            "device": self._device,
            "faiss_available": self._faiss_available,
            "vram_mb": self._memory_block.size_bytes / 1024**2 if self._memory_block else 0,
        }

    def shutdown(self) -> None:
        if self._memory_block:
            vram_pool.free("vector_index")
            self._memory_block = None
        self._index = None
        self._vectors = None
        self._ids = []
        self._count = 0
        self._dirty = False
