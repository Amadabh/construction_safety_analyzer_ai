import sys
import re
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import Frame, Detection
from inference_sdk import InferenceHTTPClient
from config import Config
from PIL import Image

ROBOFLOW_CONFIDENCE_THRESHOLD = 0.30
MAX_DIMENSION = 1280

CLAUDE_VISION_PROMPT = """You are a construction site safety inspector analyzing a video frame.

Carefully examine this image and identify ALL of the following:

1. Workers/people — are they wearing:
   - Hard hat / helmet? If NOT wearing one, that is a "NO-Hardhat" violation
   - Safety/high-visibility vest? If NOT wearing one, that is a "NO-Safety Vest" violation
   - Face mask if near dust/fumes? If needed and missing, that is a "NO-Mask" violation

2. Heavy machinery present: Excavator, Wheel Loader, Dump Truck, Crane, etc.

3. Other site objects: Ladders, Safety Cones, Vehicles, Trucks

Return ONLY a JSON array of detections, no other text, no markdown:
[
  {
    "label": "<use exact labels: NO-Hardhat, NO-Safety Vest, NO-Mask, Person, Hardhat, Safety Vest, Excavator, Wheel Loader, Dump Truck, Ladder, Safety Cone, Truck, Vehicle, Machinery>",
    "confidence": <0.0 to 1.0>,
    "bbox": [0, 0, 100, 100]
  }
]

Rules:
- Workers visible WITHOUT hard hats → include NO-Hardhat
- Workers visible WITHOUT safety vests → include NO-Safety Vest
- No people or safety issues → return []
- This is a safety system. Missing a real violation is dangerous. Be thorough."""


class VisionDetector:
    def __init__(self):
        self.roboflow_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=Config.ROBOFLOW_API_KEY
        )
        # Lazy-loaded — only initialised if Claude Vision fallback is needed
        self._bedrock_model = None

    def _get_model(self):
        """Get BedrockModel singleton — imported here to avoid circular imports."""
        if self._bedrock_model is None:
            from model import BedrockModel
            self._bedrock_model = BedrockModel.get_instance()
        return self._bedrock_model

    def _resize(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        if max(w, h) > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return image

    def _run_roboflow(self, frame: Frame) -> list[Detection]:
        """Roboflow YOLO — fast, good for ground-level footage."""
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

    def _run_claude_vision(self, frame: Frame) -> list[Detection]:
        """
        Claude Vision fallback — handles aerial, overhead, any camera angle.
        Uses BedrockModel.invoke_vision_json() so the same model ID and
        credentials from config are used everywhere.
        """
        print(f"  [ClaudeVision] Roboflow got nothing — running Claude on frame {frame.frame_num}...")
        try:
            model = self._get_model()
            raw = model.invoke_vision_json(CLAUDE_VISION_PROMPT, frame.image)

            detections = []
            for d in raw:
                label = d.get("label", "").strip()
                conf  = float(d.get("confidence", 0.5))
                bbox  = d.get("bbox", [0, 0, 100, 100])
                if label:
                    print(f"    ✓ [CV] {label}: {conf:.2f}")
                    detections.append(Detection(label=label, confidence=conf, bbox=bbox))

            if not detections:
                print(f"    [CV] No violations found in frame {frame.frame_num}")
            return detections

        except Exception as e:
            print(f"  [ClaudeVision] Frame {frame.frame_num} error: {e}")
            return []

    def detect(self, frames: list[Frame]) -> list[Detection]:
        """
        Two-stage detection:
          Stage 1 — Roboflow YOLO (fast, cheap, ground-level footage)
          Stage 2 — Claude Vision fallback (aerial/overhead/any angle)
        """
        if not frames:
            print("[VisionDetector] No frames to process")
            return []

        print(f"[VisionDetector] Processing {len(frames)} frames...")
        all_detections  = []
        roboflow_hits   = 0
        claude_fallback = 0

        for frame in frames:
            rf_detections = self._run_roboflow(frame)
            if rf_detections:
                roboflow_hits += 1
                all_detections.extend(rf_detections)
            else:
                claude_fallback += 1
                all_detections.extend(self._run_claude_vision(frame))

        print(f"\n[VisionDetector] ── Detection Summary ──")
        print(f"  Frames processed  : {len(frames)}")
        print(f"  Roboflow hits     : {roboflow_hits}")
        print(f"  Claude Vision used: {claude_fallback} frames")
        print(f"  Total detections  : {len(all_detections)}")
        for label, count in sorted(Counter(d.label for d in all_detections).items()):
            print(f"    {label}: {count}x")

        return all_detections
