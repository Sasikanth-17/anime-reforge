"""Multi-ControlNet diffusion synthesis engine built for bounded VRAM constraints."""

import logging
from typing import Optional, List
import numpy as np
import torch
from PIL import Image
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

from reforge.core.memory import VRAMManager
from reforge.synthesis.styling import StylePreset, PRESET_CINEMATIC_UNREAL

logger = logging.getLogger("reforge.synthesis.pipeline")


class SynthesisPipeline:
    """Manages the lifecycle and execution of Multi-ControlNet generation."""

    DEPTH_CONTROLNET_ID = "lllyasviel/control_v11f1p_sd15_depth"
    LINEART_CONTROLNET_ID = "lllyasviel/control_v11p_sd15_lineart"
    BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

    def __init__(
        self,
        base_model_id: str = BASE_MODEL_ID,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.base_model_id = base_model_id
        self.device = device or VRAMManager.get_optimal_device()
        self.dtype = dtype
        self.pipeline: Optional[StableDiffusionControlNetImg2ImgPipeline] = None

    def load(self) -> None:
        """Loads and configures Multi-ControlNet and UNet with CPU offloading."""
        if self.pipeline is not None:
            return

        logger.info("Initializing Depth & Lineart ControlNets (FP16)...")
        controlnets = [
            ControlNetModel.from_pretrained(
                self.DEPTH_CONTROLNET_ID,
                torch_dtype=self.dtype,
            ),
            ControlNetModel.from_pretrained(
                self.LINEART_CONTROLNET_ID,
                torch_dtype=self.dtype,
            ),
        ]

        logger.info("Initializing Base Diffusion Pipeline: %s...", self.base_model_id)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.base_model_id,
            controlnet=controlnets,
            torch_dtype=self.dtype,
            safety_checker=None,
        )

        # High-speed deterministic scheduler (converges in 20-25 steps)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        # Critical VRAM Optimizations for 4GB Envelope
        if hasattr(pipe, "vae") and pipe.vae is not None:
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
        
        if self.device.type == "cuda":
            pipe.enable_model_cpu_offload()

        self.pipeline = pipe
        logger.info("Synthesis pipeline loaded and offloaded.")

    def unload(self) -> None:
        """Purges models from memory and clears CUDA cache."""
        logger.info("Unloading Synthesis pipeline...")
        self.pipeline = None
        VRAMManager.purge()

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()

    @torch.inference_mode()
    def render_frame(
        self,
        base_frame_rgb: np.ndarray,
        depth_map: np.ndarray,
        lineart_map: np.ndarray,
        preset: StylePreset = PRESET_CINEMATIC_UNREAL,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Renders a single stylized frame conditioned on depth, lineart, and source RGB.
        Args:
            base_frame_rgb: Source frame (H, W, 3) uint8
            depth_map: Normalized depth (H, W) float32 [0.0, 1.0]
            lineart_map: Binary/grayscale edges (H, W) uint8 [0, 255]
            preset: Style configuration dataclass
            seed: RNG seed for reproducible generation
        Returns:
            np.ndarray: Stylized frame (H, W, 3) uint8
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Use with context manager or call .load().")

        h, w = base_frame_rgb.shape[:2]

        # Ensure dimension compatibility with 8x VAE downsampling
        target_w = (w // 8) * 8
        target_h = (h // 8) * 8

        # Prepare conditioning PIL inputs
        init_image = Image.fromarray(base_frame_rgb).resize((target_w, target_h), Image.BICUBIC)
        
        # Depth conditioning: convert float [0, 1] to 3-channel RGB PIL
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        depth_rgb = np.repeat(depth_uint8[:, :, None], 3, axis=-1)
        depth_pil = Image.fromarray(depth_rgb).resize((target_w, target_h), Image.BILINEAR)

        # LineArt conditioning: convert single-channel to 3-channel RGB PIL
        if lineart_map.ndim == 2:
            lineart_rgb = np.repeat(lineart_map[:, :, None], 3, axis=-1)
        else:
            lineart_rgb = lineart_map
        lineart_pil = Image.fromarray(lineart_rgb).resize((target_w, target_h), Image.NEAREST)

        generator = torch.Generator(device="cpu").manual_seed(seed)

        output = self.pipeline(
            prompt=preset.prompt,
            negative_prompt=preset.negative_prompt,
            image=init_image,
            control_image=[depth_pil, lineart_pil],
            num_inference_steps=preset.num_inference_steps,
            guidance_scale=preset.guidance_scale,
            controlnet_conditioning_scale=[
                preset.controlnet_depth_scale,
                preset.controlnet_lineart_scale,
            ],
            strength=preset.strength,
            generator=generator,
        ).images[0]

        # Resize back to exact original frame resolution if adjusted
        if (target_w, target_h) != (w, h):
            output = output.resize((w, h), Image.BICUBIC)

        return np.array(output, dtype=np.uint8)