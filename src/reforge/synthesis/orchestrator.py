"""Master execution engine controlling the entire pipeline lifecycle."""

import logging
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm

from reforge.core.types import SessionManifest
from reforge.core.memory import VRAMManager
from reforge.ingestion.pipeline import IngestionPipeline
from reforge.temporal.flow import OpticalFlowEstimator
from reforge.temporal.warping import FlowWarper
from reforge.synthesis.pipeline import SynthesisPipeline
from reforge.synthesis.styling import (
    StylePreset,
    PRESET_CINEMATIC_UNREAL,
    PRESET_MAKOTO_SHINKAI,
)
from reforge.postprocess.dof import DepthOfFieldShader
from reforge.postprocess.compositor import VideoCompositor
from reforge.ingestion.audio import AudioEngine

logger = logging.getLogger("reforge.synthesis.orchestrator")


class ReforgeEngine:
    """End-to-end multi-pass video re-rendering system."""

    STYLE_REGISTRY = {
        "cinematic_unreal": PRESET_CINEMATIC_UNREAL,
        "makoto_shinkai": PRESET_MAKOTO_SHINKAI,
    }

    def __init__(self, workspace_root: Path = Path("workspace")):
        self.workspace_root = workspace_root
        self.flow_estimator = OpticalFlowEstimator()

    def process_video(
        self,
        input_video_path: Path,
        output_video_path: Path,
        style_name: str = "cinematic_unreal",
        temporal_blend: float = 0.25,
        apply_dof: bool = True,
        seed: int = 42,
        clean_cache: bool = False,
    ) -> Path:
        input_video_path = Path(input_video_path)
        output_video_path = Path(output_video_path)
        
        session_name = input_video_path.stem
        preset = self.STYLE_REGISTRY.get(style_name, PRESET_CINEMATIC_UNREAL)

        # -------------------------------------------------------------
        # Phase 1: Ingestion & Feature Extraction (Zero-VRAM leakage)
        # -------------------------------------------------------------
        logger.info("Starting Phase 1: Ingestion & Feature Extraction...")
        ingestor = IngestionPipeline(self.workspace_root)
        manifest: SessionManifest = ingestor.execute(input_video_path, session_name)

        rendered_dir = manifest.workspace_dir / "rendered_frames"
        rendered_dir.mkdir(parents=True, exist_ok=True)
        rendered_frame_paths = []

        # -------------------------------------------------------------
        # Phase 2: Synthesis with Optical Flow Temporal Blending
        # -------------------------------------------------------------
        logger.info("Starting Phase 2: Multi-ControlNet Temporal Synthesis...")
        num_frames = len(manifest.frame_paths)
        prev_rendered_frame: np.ndarray = None
        prev_raw_frame: np.ndarray = None

        with SynthesisPipeline() as pipe:
            for idx in tqdm(range(num_frames), desc="Synthesis Pass"):
                raw_rgb = np.load(str(manifest.frame_paths[idx]))
                depth_map = np.load(str(manifest.depth_paths[idx]))
                lineart_map = np.load(str(manifest.lineart_paths[idx]))

                # Temporal Motion Feedback Loop
                if prev_rendered_frame is not None and prev_raw_frame is not None:
                    # Calculate flow from previous to current raw frame
                    fwd_flow, bwd_flow = self.flow_estimator.compute_flow(prev_raw_frame, raw_rgb)
                    occ_mask = FlowWarper.compute_occlusion_mask(fwd_flow, bwd_flow)
                    
                    # Warp previous stylized frame forward to current geometry
                    warped_prev = FlowWarper.warp_frame(prev_rendered_frame, fwd_flow)

                    # Blend raw input frame with warped previous stylized frame on non-occluded regions
                    # This anchors the latent space to the previous frame's color/texture
                    mask_3d = np.repeat(occ_mask[:, :, None], 3, axis=-1)
                    init_image = (
                        (1.0 - temporal_blend * mask_3d) * raw_rgb.astype(np.float32)
                        + (temporal_blend * mask_3d) * warped_prev.astype(np.float32)
                    ).astype(np.uint8)
                else:
                    init_image = raw_rgb

                # Diffusion synthesis pass
                current_render = pipe.render_frame(
                    base_frame_rgb=init_image,
                    depth_map=depth_map,
                    lineart_map=lineart_map,
                    preset=preset,
                    seed=seed,
                )

                # Optional Depth-guided Bokeh
                if apply_dof:
                    current_render = DepthOfFieldShader.apply_bokeh(
                        image_rgb=current_render,
                        depth_map=depth_map,
                        focal_depth=0.5,
                        aperture=8.0,
                    )

                out_frame_path = rendered_dir / f"rendered_{idx:05d}.npy"
                np.save(str(out_frame_path), current_render)
                rendered_frame_paths.append(out_frame_path)

                # Update temporal tracking pointers
                prev_rendered_frame = current_render
                prev_raw_frame = raw_rgb

        VRAMManager.purge()

        # -------------------------------------------------------------
        # Phase 3: Video Assembly & Lossless Audio Re-Multiplexing
        # -------------------------------------------------------------
        logger.info("Starting Phase 3: Encoding & Container Multiplexing...")
        temp_video_path = manifest.workspace_dir / "silent_render.mp4"
        VideoCompositor.write_video(
            frame_paths=rendered_frame_paths,
            output_path=temp_video_path,
            fps=manifest.video_meta.fps,
        )

        if manifest.audio_path and manifest.audio_path.exists():
            AudioEngine.mux_video_audio(
                rendered_video_path=temp_video_path,
                audio_path=manifest.audio_path,
                final_output_path=output_video_path,
                fps=manifest.video_meta.fps,
            )
        else:
            shutil.copy(temp_video_path, output_video_path)

        if clean_cache:
            logger.info("Purging temporary workspace cache for session: %s", session_name)
            shutil.rmtree(manifest.workspace_dir, ignore_errors=True)

        logger.info("Processing complete. Output saved to -> %s", output_video_path)
        return output_video_path