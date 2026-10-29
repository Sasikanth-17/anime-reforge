"""Generates a synthetic 3-second 24fps test animation with audio for end-to-end verification."""

import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np


def generate_clip(output_path: Path = Path("test_clip.mp4"), fps: int = 24, duration: int = 3):
    output_path = output_path.resolve()
    temp_raw_video = output_path.parent / "temp_silent.mp4"
    width, height = 512, 512
    total_frames = fps * duration

    # 1. Generate frames
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_raw_video), fourcc, fps, (width, height))

    for i in range(total_frames):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :] = [40, 20, 20]  # Background
        
        # Ground line
        cv2.line(canvas, (0, 420), (512, 420), (180, 180, 180), 3)

        # Bouncing animated subject
        cx = int(100 + (312 * (i / total_frames)))
        cy = int(300 + 80 * np.sin(i * 0.4))
        
        # Silhouette
        cv2.circle(canvas, (cx, cy), 45, (230, 230, 240), -1)
        cv2.circle(canvas, (cx, cy), 47, (10, 10, 10), 3)
        cv2.rectangle(canvas, (cx - 25, cy + 45), (cx + 25, cy + 110), (50, 120, 220), -1)
        cv2.rectangle(canvas, (cx - 25, cy + 45), (cx + 25, cy + 110), (10, 10, 10), 3)

        writer.write(canvas)

    writer.release()

    # 2. Add synthetic audio track using ffmpeg
    ffmpeg_bin = shutil.which("ffmpeg") or r"C:\Users\HP\anaconda3\envs\reforge\Library\bin\ffmpeg.exe"
    cmd = [
        str(ffmpeg_bin), "-y",
        "-i", str(temp_raw_video),
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path)
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("FFmpeg stderr output:\n", res.stderr)
        # Fallback: if audio synthesis fails, keep silent video as test_clip.mp4
        if temp_raw_video.exists():
            temp_raw_video.replace(output_path)
            print(f"Fallback: created silent test clip at {output_path}")
            return
    else:
        if temp_raw_video.exists():
            temp_raw_video.unlink()

    if output_path.exists():
        print(f"Verified test clip exists: {output_path} ({output_path.stat().st_size} bytes)")
    else:
        raise RuntimeError("Failed to generate test_clip.mp4")


if __name__ == "__main__":
    generate_clip()