"""Kinematic skeleton pose estimator using OpenPose/DWPose."""

import logging
from typing import Optional
import cv2
import numpy as np
import torch
from PIL import Image
from controlnet_aux import OpenposeDetector

from reforge.extractors.base import BaseExtractor
from reforge.core.memory import VRAMManager

logger = logging.getLogger("reforge.extractors.pose")


class PoseExtractor(BaseExtractor):
    """Extracts skeletal joint graphs rendered onto an RGB canvas."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(device=device, dtype=dtype)
        self.detector: Optional[OpenposeDetector] = None

    def load(self) -> None:
        if self.detector is not None:
            return

        logger.info("Loading OpenPose detector on %s", self.device)
        self.detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(self.device)

    def unload(self) -> None:
        logger.info("Unloading OpenPose detector and reclaiming VRAM.")
        self.detector = None
        VRAMManager.purge()

    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Extracts 2D skeletal pose canvas.
        Returns:
            np.ndarray: uint8 array shaped (H, W, 3) representing the color-coded joint bones.
        """
        if self.detector is None:
            raise RuntimeError("Pose detector not loaded. Call .load() or use context manager.")

        h, w = frame_rgb.shape[:2]
        pil_img = Image.fromarray(frame_rgb)

        pose_pil = self.detector(pil_img, detect_resolution=max(h, w))
        pose_np = np.array(pose_pil)

        if pose_np.shape[:2] != (h, w):
            pose_np = cv2.resize(pose_np, (w, h), interpolation=cv2.INTER_NEAREST)

        return pose_np.astype(np.uint8)