"""High-efficiency video writer utilizing PyAV for lossless/high-bitrate container assembly."""

from fractions import Fraction
from pathlib import Path
from typing import List
import av
import numpy as np


class VideoCompositor:
    """Assembles individual RGB numpy frames into an H.264 encoded video container."""

    @staticmethod
    def write_video(
        frame_paths: List[Path],
        output_path: Path,
        fps: float,
        pixel_format: str = "yuv420p",
        crf: int = 17,
    ) -> Path:
        """
        Reads frame files from disk sequentially and writes an MP4 video file.
        Uses CRF 17 (visually lossless).
        """
        if not frame_paths:
            raise ValueError("No frames provided to assemble.")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load first frame to determine dimensions
        first_frame = np.load(str(frame_paths[0]))
        h, w = first_frame.shape[:2]

        # PyAV >= 12 requires a Fraction or integer rate
        rate_fraction = Fraction(str(round(float(fps), 3))).limit_denominator(1001)

        container = av.open(str(output_path), mode="w")
        stream = container.add_stream("libx264", rate=rate_fraction)
        stream.width = w
        stream.height = h
        stream.pix_fmt = pixel_format
        stream.options = {"crf": str(crf), "preset": "medium"}

        for idx, f_path in enumerate(frame_paths):
            frame_rgb = np.load(str(f_path))
            av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            av_frame.pts = idx
            for packet in stream.encode(av_frame):
                container.mux(packet)

        # Flush encoder buffers
        for packet in stream.encode():
            container.mux(packet)

        container.close()
        return output_path
