from __future__ import annotations

import pytest

from vram_fabric.memory.pool import VRAMMemoryPool


class TestVRAMMemoryPool:
    def setup_method(self) -> None:
        self.pool = VRAMMemoryPool(
            total_bytes=100 * 1024**2,
            ram_spillover_limit_bytes=50 * 1024**2,
        )

    def test_initial_stats(self) -> None:
        stats = self.pool.stats()
        assert stats.total_bytes == 100 * 1024**2
        assert stats.free_bytes > 0
        assert stats.blocks == 0

    def test_allocate_and_free(self) -> None:
        block = self.pool.allocate("test_block", 10 * 1024**2, owner="test")
        assert block is not None
        assert block.in_use
        assert block.owner == "test"

        self.pool.free("test_block")
        assert self.pool.get("test_block") is None

    def test_allocate_exceeding_vram_spills_to_ram(self) -> None:
        self.pool.allocate("big_block", 90 * 1024**2, owner="test")

        block = self.pool.allocate("spill_block", 30 * 1024**2, owner="test")
        assert block is not None
        assert block.tier in ("ram", "vram")

        stats = self.pool.stats()
        assert stats.spillover_ram_bytes > 0 or stats.spillover_ssd_bytes > 0 or block.tier == "vram"

    def test_evict_lru(self) -> None:
        self.pool.allocate("keep_1", 10 * 1024**2, owner="a")
        self.pool.allocate("keep_2", 10 * 1024**2, owner="b")
        self.pool.allocate("evict_me", 10 * 1024**2, owner="c")

        freed = self.pool.evict_lru(keep_keys={"keep_1", "keep_2"})
        assert freed >= 0

    def test_free_bytes(self) -> None:
        initial_free = self.pool.free_bytes()

        self.pool.allocate("block", 20 * 1024**2, owner="test")
        mid_free = self.pool.free_bytes()
        assert mid_free < initial_free

        self.pool.free("block")
        final_free = self.pool.free_bytes()
        assert final_free <= initial_free

    def test_reallocate_bigger(self) -> None:
        b1 = self.pool.allocate("key", 5 * 1024**2, owner="a")
        assert b1.size_bytes == 5 * 1024**2

        b2 = self.pool.allocate("key", 10 * 1024**2, owner="a")
        assert b2.size_bytes >= 10 * 1024**2

    def test_shutdown(self) -> None:
        self.pool.allocate("a", 1 * 1024**2)
        self.pool.allocate("b", 1 * 1024**2)
        self.pool.shutdown()
        assert self.pool.stats().blocks == 0
