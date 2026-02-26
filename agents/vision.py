import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import Frame, Detection
from inference_sdk import InferenceHTTPClient
from config import Config
from PIL import Image

ROBOFLOW_CONFIDENCE_THRESHOLD = 0.30
MAX_DIMENSION = 1280


class VisionDetector:
    def __init__(self):
        self.roboflow_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=Config.ROBOFLOW_API_KEY
        )

    def _resize(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        if max(w, h) > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return image

    def _run_roboflow(self, frame: Frame) -> list[Detection]:
        try:
            result = self.roboflow_client.infer(
                self._resize(frame.image),
                model_id="construction-site-safety/27"
            )
            raw_preds = result.get("predictions", [])
            print(f"  [Roboflow] Frame {frame.frame_num} (t={frame.timestamp:.1f}s): "
                  f"{len(raw_preds)} raw predictions")

            detections = []
            for p in raw_preds:
                conf, label = p["confidence"], p["class"]
                ok = conf >= ROBOFLOW_CONFIDENCE_THRESHOLD
                print(f"    {'✓' if ok else '✗ filtered'} {label}: {conf:.3f}")
                if ok:
                    detections.append(Detection(
                        label=label,
                        confidence=conf,
                        bbox=[p["x"], p["y"], p["width"], p["height"]]
                    ))
            return detections

        except Exception as e:
            print(f"  [Roboflow] Frame {frame.frame_num} error: {e}")
            return []

    def detect(self, frames: list[Frame]) -> list[Detection]:
        if not frames:
            print("[VisionDetector] No frames to process")
            return []

        print(f"[VisionDetector] Processing {len(frames)} frames...")
        all_detections = []

        for frame in frames:
            all_detections.extend(self._run_roboflow(frame))

        print(f"\n[VisionDetector] ── Detection Summary ──")
        print(f"  Frames processed : {len(frames)}")
        print(f"  Total detections : {len(all_detections)}")
        for label, count in sorted(Counter(d.label for d in all_detections).items()):
            print(f"    {label}: {count}x")

        return all_detections