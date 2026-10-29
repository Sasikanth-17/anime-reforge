"""High-performance dense optical flow computation using DIS Optical Flow."""

import logging
from typing import Tuple
import cv2
import numpy as np

logger = logging.getLogger("reforge.temporal.flow")


class OpticalFlowEstimator:
    """Computes forward and backward dense flow fields between adjacent frames."""

    def __init__(self, preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM):
        """
        Presets available:
            PRESET_ULTRAFAST: Lowest latency (~5ms)
            PRESET_FAST: Balanced (~10ms)
            PRESET_MEDIUM: High accuracy, sharp edges (~20ms)
        """
        self.dis = cv2.DISOpticalFlow.create(preset)

    @staticmethod
    def _to_grayscale(frame_rgb: np.ndarray) -> np.ndarray:
        if frame_rgb.ndim == 3:
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        return frame_rgb

    def compute_flow(
        self, prev_frame_rgb: np.ndarray, curr_frame_rgb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes bidirectional optical flow:
        - forward_flow: maps pixels from prev_frame -> curr_frame
        - backward_flow: maps pixels from curr_frame -> prev_frame

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                forward_flow: float32 array shaped (H, W, 2)
                backward_flow: float32 array shaped (H, W, 2)
        """
        prev_gray = self._to_grayscale(prev_frame_rgb)
        curr_gray = self._to_grayscale(curr_frame_rgb)

        # Forward flow: I1 -> I2
        forward_flow = np.empty((*prev_gray.shape, 2), dtype=np.float32)
        forward_flow = self.dis.calc(prev_gray, curr_gray, forward_flow)

        # Backward flow: I2 -> I1
        backward_flow = np.empty((*curr_gray.shape, 2), dtype=np.float32)
        backward_flow = self.dis.calc(curr_gray, prev_gray, backward_flow)

        return forward_flow, backward_flow