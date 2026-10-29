"""High-fidelity lineart and structural edge extractor."""

import logging
from typing import Optional
import cv2
import numpy as np
import torch
from PIL import Image
from controlnet_aux import LineartAnimeDetector

from reforge.extractors.base import BaseExtractor
from reforge.core.memory import VRAMManager

logger = logging.getLogger("reforge.extractors.lineart")


class LineartExtractor(BaseExtractor):
    """Extracts crisp anime ink lines and structural contour maps."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(device=device, dtype=dtype)
        self.detector: Optional[LineartAnimeDetector] = None

    def load(self) -> None:
        if self.detector is not None:
            return

        logger.info("Loading LineArt Anime detector on %s", self.device)
        self.detector = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators").to(self.device)

    def unload(self) -> None:
        logger.info("Unloading LineArt detector and reclaiming VRAM.")
        self.detector = None
        VRAMManager.purge()

    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Extracts anime lineart map.
        Returns:
            np.ndarray: uint8 array shaped (H, W) where 255 = ink edge, 0 = background.
        """
        if self.detector is None:
            raise RuntimeError("Lineart detector not loaded. Call .load() or use context manager.")

        h, w = frame_rgb.shape[:2]
        pil_img = Image.fromarray(frame_rgb)
        
        # LineartAnimeDetector expects PIL and returns PIL
        lineart_pil = self.detector(pil_img, detect_resolution=max(h, w))
        lineart_np = np.array(lineart_pil)

        # If 3 channels, convert to single-channel grayscale
        if lineart_np.ndim == 3:
            lineart_np = cv2.cvtColor(lineart_np, cv2.COLOR_RGB2GRAY)

        # Ensure exact spatial match to source
        if lineart_np.shape != (h, w):
            lineart_np = cv2.resize(lineart_np, (w, h), interpolation=cv2.INTER_AREA)

        return lineart_np.astype(np.uint8)