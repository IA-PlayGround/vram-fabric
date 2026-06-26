from __future__ import annotations

import ctypes
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MemoryBlock:
    ptr: int = 0
    size_bytes: int = 0
    in_use: bool = False
    owner: str = ""
    tier: str = "vram"
    cdata: Any = None

    @property
    def free(self) -> bool:
        return not self.in_use


@dataclass
class MemoryPoolStats:
    total_bytes: int = 0
    free_bytes: int = 0
    used_bytes: int = 0
    blocks: int = 0
    spillover_ram_bytes: int = 0
    spillover_ssd_bytes: int = 0


class VRAMMemoryPool:
    """Unified VRAM memory pool with spillover to RAM and SSD."""

    def __init__(
        self,
        total_bytes: int = 0,
        ram_spillover_limit_bytes: int = 8 * 1024**3,
        ssd_spillover_path: str = "/tmp/vram_fabric_spillover",
    ) -> None:
        self._total_bytes = total_bytes
        self._ram_limit = ram_spillover_limit_bytes
        self._ssd_path = ssd_spillover_path
        self._blocks: dict[str, MemoryBlock] = {}
        self._ram_blocks: dict[str, bytes] = {}
        self._ram_used = 0
        self._ssd_used = 0
        self._tensor_allocator: Callable | None = None
        self._cuda_available = False

    def setup_cuda(self, device_id: int = 0) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(device_id)
                props = torch.cuda.get_device_properties(device_id)
                if self._total_bytes == 0:
                    self._total_bytes = int(props.total_memory * 0.90)
                self._cuda_available = True
                self._tensor_allocator = lambda size, device='cuda': torch.empty(size, dtype=torch.float16, device=device)
                logger.info(
                    "VRAM pool initialized: total=%.1fGB, device=%s",
                    self._total_bytes / (1024**3), props.name,
                )
        except ImportError:
            logger.warning("PyTorch not available. Running in CPU-only mode.")
            self._total_bytes = self._total_bytes or 1024**3
            self._cuda_available = False

    def allocate(self, key: str, size_bytes: int, owner: str = "") -> MemoryBlock:
        if key in self._blocks and self._blocks[key].size_bytes >= size_bytes:
            block = self._blocks[key]
            block.in_use = True
            block.owner = owner
            return block

        free = self.free_bytes()
        if size_bytes <= free:
            return self._allocate_vram(key, size_bytes, owner)
        return self._allocate_spillover(key, size_bytes, owner)

    def _allocate_vram(self, key: str, size_bytes: int, owner: str) -> MemoryBlock:
        cdata = None
        ptr = 0
        if self._cuda_available and self._tensor_allocator:
            try:
                import torch
                n_floats = size_bytes // 2
                tensor = self._tensor_allocator(n_floats)
                ptr = tensor.data_ptr()
                cdata = tensor
            except Exception as e:
                logger.warning("CUDA allocation failed: %s", e)
                self._cuda_available = False

        block = MemoryBlock(ptr=ptr, size_bytes=size_bytes, in_use=True, owner=owner, tier="vram", cdata=cdata)
        self._blocks[key] = block
        return block

    def _allocate_spillover(self, key: str, size_bytes: int, owner: str) -> MemoryBlock:
        if size_bytes + self._ram_used <= self._ram_limit:
            data = bytes(size_bytes)
            self._ram_blocks[key] = data
            self._ram_used += size_bytes
            logger.info("Spillover RAM: key=%s size=%dMB", key, size_bytes // 1024**2)
            block = MemoryBlock(ptr=0, size_bytes=size_bytes, in_use=True, owner=owner, tier="ram", cdata=data)
            self._blocks[key] = block
            return block

        import os
        os.makedirs(self._ssd_path, exist_ok=True)
        spill_path = os.path.join(self._ssd_path, f"{key}.spill")
        with open(spill_path, "wb") as f:
            f.write(b"\x00" * min(size_bytes, 1024**3))
        self._ssd_used += size_bytes
        logger.warning("Spillover SSD: key=%s size=%dMB", key, size_bytes // 1024**2)
        block = MemoryBlock(ptr=0, size_bytes=size_bytes, in_use=True, owner=owner, tier="ssd")
        self._blocks[key] = block
        return block

    def free(self, key: str) -> None:
        block = self._blocks.pop(key, None)
        if block is None:
            return
        if block.tier == "vram" and block.cdata is not None:
            del block.cdata
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
        elif block.tier == "ram":
            self._ram_blocks.pop(key, None)
            self._ram_used = max(0, self._ram_used - block.size_bytes)
        elif block.tier == "ssd":
            import os
            spill_path = os.path.join(self._ssd_path, f"{key}.spill")
            if os.path.exists(spill_path):
                os.remove(spill_path)
            self._ssd_used = max(0, self._ssd_used - block.size_bytes)

    def get(self, key: str) -> MemoryBlock | None:
        return self._blocks.get(key)

    def used_bytes(self) -> int:
        return sum(b.size_bytes for b in self._blocks.values() if b.tier == "vram" and b.in_use)

    def free_bytes(self) -> int:
        return max(0, self._total_bytes - self.used_bytes())

    def stats(self) -> MemoryPoolStats:
        return MemoryPoolStats(
            total_bytes=self._total_bytes,
            free_bytes=self.free_bytes(),
            used_bytes=self.used_bytes(),
            blocks=len(self._blocks),
            spillover_ram_bytes=self._ram_used,
            spillover_ssd_bytes=self._ssd_used,
        )

    def evict_lru(self, keep_keys: set[str]) -> int:
        freed = 0
        for key in list(self._blocks.keys()):
            if key not in keep_keys:
                block = self._blocks[key]
                freed += block.size_bytes
                self.free(key)
                logger.debug("Evicted VRAM block: %s (%dMB)", key, block.size_bytes // 1024**2)
                if self.free_bytes() > 512 * 1024**2:
                    break
        return freed

    def shutdown(self) -> None:
        for key in list(self._blocks.keys()):
            self.free(key)


vram_pool = VRAMMemoryPool()
