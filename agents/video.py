import subprocess
import numpy as np
from PIL import Image
from schemas import Frame


class VideoProcessor:
    def __init__(self, max_frames: int = 10):
        """Extract up to max_frames at 1fps from a video."""
        self.max_frames = max_frames

    def _get_video_properties(self, video_path: str) -> dict:
        """Get video fps and dimensions."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")

        lines = result.stdout.strip().split("\n")
        if len(lines) < 3:
            raise RuntimeError(f"Unexpected ffprobe output: {result.stdout!r}")

        width, height, fps_str = lines[0], lines[1], lines[2]

        # Safe fps parsing — handles both '30' and '30000/1001', never uses eval
        parts = fps_str.strip().split("/")
        fps = int(parts[0]) / int(parts[1]) if len(parts) == 2 else float(parts[0])

        return {"fps": fps, "width": int(width), "height": int(height)}

    def process(self, video_path: str) -> list[Frame]:
        """
        Extract frames at 1fps up to max_frames.
        Timestamp = frame index (seconds) since we extract at exactly 1fps.
        """
        print(f"Processing video: {video_path}")
        props = self._get_video_properties(video_path)
        print(f"  {props['width']}x{props['height']} @ {props['fps']:.2f}fps")

        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", "fps=1",
            "-f", "image2pipe",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo", "-"
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        frames = []
        frame_size = props["width"] * props["height"] * 3

        try:
            while len(frames) < self.max_frames:
                frame_data = process.stdout.read(frame_size)
                if len(frame_data) != frame_size:
                    break

                frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                    (props["height"], props["width"], 3)
                )

                idx = len(frames)
                frames.append(Frame(
                    frame_num=idx,
                    timestamp=float(idx),   # fps=1 so frame N = second N
                    image=Image.fromarray(frame_array, "RGB")
                ))

        finally:
            process.stdout.close()
            process.stderr.close()
            process.wait()
            if process.returncode not in (0, None) and not frames:
                raise RuntimeError(f"ffmpeg failed with return code {process.returncode}")

        print(f"Extracted {len(frames)} frames")
        return frames
