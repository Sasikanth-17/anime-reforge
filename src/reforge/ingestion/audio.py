"""Lossless audio extraction and container re-multiplexing using FFmpeg."""

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("reforge.ingestion.audio")


def _get_ffmpeg_bin() -> str:
    """Resolves the ffmpeg executable path across conda environments and PATH."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # Fallback directly into active Conda environment Library\bin on Windows
    conda_fallback = Path(shutil.which("python") or "").parent / "Library" / "bin" / "ffmpeg.exe"
    if conda_fallback.exists():
        return str(conda_fallback)
    return "ffmpeg"


class AudioEngine:
    """Coordinates audio demuxing and final video-audio multiplexing passes."""

    @staticmethod
    def extract_audio(video_path: Path, output_audio_path: Path) -> bool:
        """Extracts the audio track losslessly without re-encoding."""
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_bin = _get_ffmpeg_bin()
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(video_path),
            "-vn",
            "-c:a", "copy",
            str(output_audio_path)
        ]
        
        logger.info("Extracting audio: %s -> %s", video_path.name, output_audio_path.name)
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0

    @staticmethod
    def mux_video_audio(
        rendered_video_path: Path,
        audio_path: Path,
        final_output_path: Path,
        fps: float
    ) -> bool:
        """Multiplexes processed silent video frames with the preserved audio track."""
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_bin = _get_ffmpeg_bin()
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(rendered_video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "320k",
            "-shortest",
            str(final_output_path)
        ]

        logger.info("Multiplexing final package -> %s", final_output_path)
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0