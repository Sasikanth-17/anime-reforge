"""Style definitions, prompt injection templates, and generation presets."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StylePreset:
    """Encapsulates positive/negative prompts, guidance scale, and conditioning weights."""
    name: str
    prompt: str
    negative_prompt: str
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    controlnet_depth_scale: float = 0.85
    controlnet_lineart_scale: float = 0.65
    strength: float = 0.75  # For img2img style transfer


PRESET_CINEMATIC_UNREAL = StylePreset(
    name="cinematic_unreal",
    prompt=(
        "cinematic film still, photorealistic 8k, Unreal Engine 5 render, "
        "volumetric lighting, octane render, dynamic lighting, high subsurface scattering, "
        "metallic reflections, raytraced shadows, hyper-detailed, masterpiece"
    ),
    negative_prompt=(
        "flat colors, 2D anime illustration, low quality, cartoon, blurry, deformed, "
        "bad anatomy, watermark, text, artifacts, lowres, oversaturated"
    ),
    num_inference_steps=24,
    guidance_scale=7.0,
    controlnet_depth_scale=0.85,
    controlnet_lineart_scale=0.60,
    strength=0.70,
)

PRESET_MAKOTO_SHINKAI = StylePreset(
    name="makoto_shinkai",
    prompt=(
        "Makoto Shinkai aesthetic, gorgeous hyper-detailed background, golden hour lighting, "
        "dramatic cloud formations, vibrant atmospheric scattering, lens flare, 4k visual novel art, "
        "cinematic lighting composition, masterpiece"
    ),
    negative_prompt=(
        "low quality, dull lighting, muddy colors, deformed faces, flat shading, artifacts"
    ),
    num_inference_steps=24,
    guidance_scale=7.5,
    controlnet_depth_scale=0.80,
    controlnet_lineart_scale=0.70,
    strength=0.75,
)