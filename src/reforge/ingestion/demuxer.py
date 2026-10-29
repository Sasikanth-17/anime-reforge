"""Low-overhead video demuxing and frame extraction engine using PyAV."""

from pathlib import Path
from typing import Generator, Tuple
import av
import numpy as np

from reforge.core.types import VideoMetadata, FramePacket


class VideoDemuxer:
    """Extracts raw frames and technical metadata from container video files."""

    def __init__(self, input_path: Path):
        self.input_path = Path(input_path)
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Input video file not found: {self.input_path}")

    def probe(self) -> VideoMetadata:
        """Inspects container streams and extracts precise video specifications."""
        with av.open(str(self.input_path)) as container:
            video_stream = next((s for s in container.streams if s.type == "video"), None)
            if video_stream is None:
                raise ValueError(f"No valid video stream located inside {self.input_path}")

            audio_stream = next((s for s in container.streams if s.type == "audio"), None)
            
            # Extract duration and frame count safely
            fps = float(video_stream.average_rate or 24.0)
            duration = float(video_stream.duration * video_stream.time_base) if video_stream.duration else 0.0
            total_frames = video_stream.frames or int(duration * fps)

            return VideoMetadata(
                filepath=self.input_path,
                total_frames=total_frames,
                fps=fps,
                width=video_stream.width,
                height=video_stream.height,
                duration_sec=duration,
                has_audio=audio_stream is not None,
            )

    def stream_frames(self) -> Generator[FramePacket, None, None]:
        """Decodes the video stream frame-by-frame as RGB ndarrays."""
        with av.open(str(self.input_path)) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                return

            for idx, frame in enumerate(container.decode(stream)):
                # Convert directly to standard contiguous HWC RGB uint8
                rgb_array = frame.to_ndarray(format="rgb24")
                pts = frame.pts if frame.pts is not None else idx
                time_sec = float(pts * stream.time_base) if stream.time_base else idx / 24.0

                yield FramePacket(
                    index=idx,
                    timestamp_pts=pts,
                    timestamp_sec=time_sec,
                    image_rgb=rgb_array,
                )