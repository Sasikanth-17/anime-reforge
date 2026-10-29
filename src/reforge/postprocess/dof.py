"""Physically plausible optical Depth-of-Field (DoF) and bokeh effect engine."""

import cv2
import numpy as np


class DepthOfFieldShader:
    """Simulates aperture lens bokeh based on scene depth disparity."""

    @staticmethod
    def apply_bokeh(
        image_rgb: np.ndarray,
        depth_map: np.ndarray,
        focal_depth: float = 0.5,
        aperture: float = 12.0,
        max_blur_kernel: int = 15,
    ) -> np.ndarray:
        """
        Applies depth-of-field blur using continuous circle-of-confusion weighting.
        Args:
            image_rgb: uint8 array (H, W, 3)
            depth_map: float32 array (H, W) normalized to [0.0, 1.0]
            focal_depth: Depth plane where the image remains tack-sharp (0.0 to 1.0)
            aperture: Size of circle of confusion (higher = shallower depth of field)
            max_blur_kernel: Maximum radius for bokeh blur filter
        Returns:
            np.ndarray: Blended RGB frame with natural optical bokeh
        """
        # Ensure odd kernel dimension
        if max_blur_kernel % 2 == 0:
            max_blur_kernel += 1

        # Circle of Confusion (CoC): distance from focal plane scaled by aperture
        coc = np.abs(depth_map - focal_depth) * aperture
        coc_weight = np.clip(coc, 0.0, 1.0)[:, :, None]

        # Generate blurred background layer using circular disc / Gaussian filter
        blurred_layer = cv2.GaussianBlur(image_rgb, (max_blur_kernel, max_blur_kernel), 0)

        # Alpha composite: sharp where near focal plane, blurred as depth deviates
        blended = (1.0 - coc_weight) * image_rgb.astype(np.float32) + coc_weight * blurred_layer.astype(np.float32)

        return np.clip(blended, 0, 255).astype(np.uint8)