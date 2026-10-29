"""Orchestrator that demuxes video and runs headless feature extraction with zero VRAM leaks."""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm

from reforge.core.types import SessionManifest, VideoMetadata
from reforge.core.memory import VRAMManager
from reforge.ingestion.demuxer import VideoDemuxer
from reforge.ingestion.audio import AudioEngine
from reforge.extractors.depth import DepthExtractor
from reforge.extractors.lineart import LineartExtractor

logger = logging.getLogger("reforge.ingestion.pipeline")


class IngestionPipeline:
    """Pre-processes raw input video into synchronized frame and conditioning caches."""

    def __init__(self, workspace_root: Path = Path("workspace")):
        self.workspace_root = workspace_root

    def execute(self, video_path: Path, session_name: str) -> SessionManifest:
        session_dir = self.workspace_root / session_name
        frames_dir = session_dir / "raw_frames"
        depth_dir = session_dir / "depth"
        lineart_dir = session_dir / "lineart"
        audio_dir = session_dir / "audio"

        for d in [frames_dir, depth_dir, lineart_dir, audio_dir]:
            d.mkdir(parents=True, exist_ok=True)

        demuxer = VideoDemuxer(video_path)
        meta = demuxer.probe()
        logger.info(
            "Loaded video: %s | %dx%d @ %.2f fps | %d frames",
            meta.filepath.name, meta.width, meta.height, meta.fps, meta.total_frames
        )

        manifest = SessionManifest(
            session_id=session_name,
            workspace_dir=session_dir,
            video_meta=meta,
        )

        # 1. Audio Extraction
        if meta.has_audio:
            audio_out = audio_dir / "stream.aac"
            if AudioEngine.extract_audio(video_path, audio_out):
                manifest.audio_path = audio_out

        # 2. Unpack Raw Frames
        logger.info("Unpacking video frames...")
        frames_buffer = []
        for packet in tqdm(demuxer.stream_frames(), total=meta.total_frames, desc="Demuxing"):
            f_path = frames_dir / f"frame_{packet.index:05d}.npy"
            np.save(str(f_path), packet.image_rgb)
            manifest.frame_paths.append(f_path)
            frames_buffer.append(packet.image_rgb)

        # 3. Depth Extraction (Scoped VRAM load & purge)
        logger.info("Extracting Depth Maps...")
        with DepthExtractor() as depth_ext:
            for idx, img in enumerate(tqdm(frames_buffer, desc="Depth Passes")):
                depth_map = depth_ext.process_frame(img)
                d_path = depth_dir / f"depth_{idx:05d}.npy"
                np.save(str(d_path), depth_map)
                manifest.depth_paths.append(d_path)

        # 4. LineArt Extraction (Scoped VRAM load & purge)
        logger.info("Extracting LineArt...")
        with LineartExtractor() as lineart_ext:
            for idx, img in enumerate(tqdm(frames_buffer, desc="LineArt Passes")):
                lineart_map = lineart_ext.process_frame(img)
                l_path = lineart_dir / f"lineart_{idx:05d}.npy"
                np.save(str(l_path), lineart_map)
                manifest.lineart_paths.append(l_path)

        VRAMManager.purge()
        logger.info("Ingestion completed successfully for session: %s", session_name)
        return manifest