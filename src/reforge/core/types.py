"""Domain types, dataclasses, and immutable contracts for the reforge pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata extracted from the raw input video."""
    filepath: Path
    total_frames: int
    fps: float
    width: int
    height: int
    duration_sec: float
    has_audio: bool


@dataclass
class FramePacket:
    """Represents a single frame in memory or ready for serialization."""
    index: int
    timestamp_pts: int
    timestamp_sec: float
    image_rgb: np.ndarray  # Shape: (H, W, 3), dtype: uint8


@dataclass
class ExtractedFeatureBundle:
    """Intermediate extracted conditioning maps for a single frame."""
    frame_index: int
    depth_map: Optional[np.ndarray] = None    # Shape: (H, W), dtype: float32 [0.0, 1.0]
    pose_canvas: Optional[np.ndarray] = None  # Shape: (H, W, 3), dtype: uint8
    lineart_map: Optional[np.ndarray] = None  # Shape: (H, W), dtype: uint8 [0, 255]


@dataclass
class SessionManifest:
    """Tracks the state and paths of cached intermediate artifacts for a run."""
    session_id: str
    workspace_dir: Path
    video_meta: VideoMetadata
    frame_paths: List[Path] = field(default_factory=list)
    depth_paths: List[Path] = field(default_factory=list)
    pose_paths: List[Path] = field(default_factory=list)
    lineart_paths: List[Path] = field(default_factory=list)
    audio_path: Optional[Path] = None