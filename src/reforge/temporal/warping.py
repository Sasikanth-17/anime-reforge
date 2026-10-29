"""Frame warping and forward-backward consistency occlusion mask generation."""

import cv2
import numpy as np


class FlowWarper:
    """Performs spatial deformation and occlusion boundary detection using flow fields."""

    @staticmethod
    def warp_frame(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Warps an image backwards using a flow field.
        Args:
            image: np.ndarray (H, W, C) or (H, W)
            flow: np.ndarray (H, W, 2) where flow[:, :, 0] = dx, flow[:, :, 1] = dy
        Returns:
            Warped image of matching shape and dtype.
        """
        h, w = flow.shape[:2]
        
        # Build coordinate grid
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[:, :, 0]).astype(np.float32)
        map_y = (grid_y + flow[:, :, 1]).astype(np.float32)

        warped = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return warped

    @classmethod
    def compute_occlusion_mask(
        cls, forward_flow: np.ndarray, backward_flow: np.ndarray, threshold: float = 1.0
    ) -> np.ndarray:
        """
        Detects occlusions via Forward-Backward Consistency Check:
        || flow_fwd + warp(flow_bwd, flow_fwd) ||^2 > alpha1 * (||flow_fwd||^2 + ||warp(flow_bwd)||^2) + alpha2

        Returns:
            np.ndarray: float32 mask of shape (H, W), values in [0.0, 1.0].
            1.0 = visible/consistent surface, 0.0 = occluded/disoccluded (needs rediffusion).
        """
        # Warp backward flow into forward coordinates
        warped_bwd = cls.warp_frame(backward_flow, forward_flow)

        # Vector addition should cancel out to zero if consistent:
        flow_diff = forward_flow + warped_bwd
        sq_diff = np.sum(flow_diff ** 2, axis=-1)

        sq_fwd = np.sum(forward_flow ** 2, axis=-1)
        sq_bwd = np.sum(warped_bwd ** 2, axis=-1)

        criterion = 0.01 * (sq_fwd + sq_bwd) + 0.5 * threshold

        # 0 where occluded, 1 where motion is consistent
        valid_motion = (sq_diff <= criterion).astype(np.float32)

        # Morphological clean-up to remove isolated pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        valid_motion = cv2.morphologyEx(valid_motion, cv2.MORPH_OPEN, kernel)

        return valid_motion