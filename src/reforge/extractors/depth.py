"""Monocular metric/relative depth extractor using Depth-Anything-V2-Small."""

import logging
from typing import Optional
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from reforge.extractors.base import BaseExtractor
from reforge.core.memory import VRAMManager

logger = logging.getLogger("reforge.extractors.depth")


class DepthExtractor(BaseExtractor):
    """Extracts normalized continuous depth maps [0.0, 1.0] from RGB frames."""

    DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(device=device, dtype=dtype)
        self.model_id = model_id
        self.image_processor: Optional[AutoImageProcessor] = None

    def load(self) -> None:
        if self.model is not None:
            return

        logger.info("Loading Depth model: %s on %s (dtype=%s)", self.model_id, self.device, self.dtype)
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def unload(self) -> None:
        logger.info("Unloading Depth model and reclaiming VRAM.")
        self.model = None
        self.image_processor = None
        VRAMManager.purge()

    @torch.inference_mode()
    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Infers depth from RGB array (H, W, 3) uint8.
        Returns:
            np.ndarray: float32 array shaped (H, W) normalized to [0.0, 1.0].
        """
        if self.model is None or self.image_processor is None:
            raise RuntimeError("Model is not loaded. Call .load() or use context manager.")

        h, w = frame_rgb.shape[:2]
        pil_img = Image.fromarray(frame_rgb)
        
        inputs = self.image_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.device.type == "cuda" and self.dtype == torch.float16:
            inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}

        outputs = self.model(**inputs)
        predicted_depth = outputs.predicted_depth

        # Interpolate directly to original resolution
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_np = prediction.cpu().float().numpy()

        # Min-max normalization to [0.0, 1.0] range
        d_min = depth_np.min()
        d_max = depth_np.max()
        if d_max - d_min > 1e-6:
            depth_normalized = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_normalized = np.zeros_like(depth_np)

        return depth_normalized.astype(np.float32)