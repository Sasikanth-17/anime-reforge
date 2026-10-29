"""Abstract base interface for all neural and procedural feature extractors."""

from abc import ABC, abstractmethod
from typing import Optional, Any
import numpy as np
import torch

from reforge.core.memory import VRAMManager


class BaseExtractor(ABC):
    """Lifecycle-managed extractor interface ensuring zero memory leaks."""

    def __init__(self, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float16):
        self.device = device or VRAMManager.get_optimal_device()
        self.dtype = dtype
        self.model: Optional[Any] = None

    @abstractmethod
    def load(self) -> None:
        """Loads weights to device."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Purges weights from device and cleans VRAM."""
        pass

    @abstractmethod
    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Executes feature extraction on a single RGB frame (H, W, 3).
        Must return a numpy ndarray ready for disk serialization or downstream caching.
        """
        pass

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()