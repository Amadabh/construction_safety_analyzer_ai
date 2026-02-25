import sys
from pathlib import Path

# Add parent directory to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import Frame, Detection
from inference_sdk import InferenceHTTPClient
from config import Config  
from PIL import Image

class VisionDetector:
    def __init__(self):
        self.CLIENT = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key= Config.ROBOFLOW_API_KEY
        )

    def detect(self, frames: list[Frame]) -> list[Detection]:
        """Detect objects in video frames using Roboflow model."""
        print(f"Detecting objects in {len(frames)} frames")

        results = []
        for frame in frames:
            # Use PIL Image directly from frame - no disk I/O needed
            result = self.CLIENT.infer(frame.image, model_id="construction-site-safety/27")
            results.append(result)
        return results

