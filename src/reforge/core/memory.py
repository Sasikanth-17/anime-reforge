"""VRAM lifecycle management and deterministic garbage collection."""

import gc
import logging
from typing import Generator
from contextlib import contextmanager
import torch

logger = logging.getLogger("reforge.core.memory")


class VRAMManager:
    """Guarantees hardware memory boundaries by controlling device placement."""

    @staticmethod
    def get_optimal_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @classmethod
    def purge(cls) -> None:
        """Forces Python garbage collection and clears CUDA runtime memory allocator caches."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.debug(
                "CUDA cache purged. Allocated: %.2f MB | Reserved: %.2f MB",
                torch.cuda.memory_allocated() / (1024 ** 2),
                torch.cuda.memory_reserved() / (1024 ** 2),
            )

    @classmethod
    @contextmanager
    def scoped_allocation(cls, name: str = "operation") -> Generator[None, None, None]:
        """Context manager to ensure cache teardown after an intensive pass."""
        cls.purge()
        try:
            yield
        finally:
            logger.debug("Tearing down scope: %s", name)
            cls.purge()