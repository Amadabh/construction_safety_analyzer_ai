import subprocess
import numpy as np
from PIL import Image
from schemas import Frame

class VideoProcessor:
    def __init__(self, max_frames: int = 10):
        """Extract max N key frames (usually 5-10 from a video)."""
        self.max_frames = max_frames
    
    def _get_video_properties(self, video_path: str) -> dict:
        """Get video fps and dimensions."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,width,height",
            "-of", "csv=p=0",
            video_path
        ]
        output = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        fps_str, width, height = output.split(",")
        fps = eval(fps_str) if "/" in fps_str else float(fps_str)
        return {"fps": fps, "width": int(width), "height": int(height)}
    
    def process(self, video_path: str) -> list[Frame]:
        """Extract key frames (5-10 max from video)."""
        print(f"Processing video: {video_path}")
        props = self._get_video_properties(video_path)
        
        # Extract only key frames (I-frames) - naturally gives ~5-10 frames
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", "select=eq(pict_type\\,I)",
            "-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-"
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frames = []
        frame_count = 0
        frame_size = props["width"] * props["height"] * 3
        
        while len(frames) < self.max_frames:
            frame_data = process.stdout.read(frame_size)
            if len(frame_data) != frame_size:
                break
            
            frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                (props["height"], props["width"], 3)
            )
            
            frame = Frame(
                frame_num=frame_count,
                timestamp=frame_count / props["fps"],
                image=Image.fromarray(frame_array, "RGB")
            )
            frames.append(frame)
            frame_count += 1
        
        process.wait()
        print(f"Extracted {len(frames)} key frames")
        return frames
