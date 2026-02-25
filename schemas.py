from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime
from PIL import Image

class AlertLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class Frame(BaseModel):
    """Represents a single video frame extracted for analysis."""
    frame_num: int
    timestamp: float
    image: Image.Image  # PIL Image object directly
    
    class Config:
        arbitrary_types_allowed = True

class Detection(BaseModel):
    """Object detected in a frame."""
    label: str
    confidence: float
    bbox: List[float] # [x, y, w, h]

class Violation(BaseModel):
    """Safety violation detected."""
    type: str
    confidence: float
    duration_seconds: float = 0.0
    severity: AlertLevel
    timestamp_start: float
    timestamp_end: float
    
class RiskAssessment(BaseModel):
    """Overall risk assessment for a sequence or event."""
    risk_score: int = Field(..., ge=0, le=100)
    alert_level: AlertLevel
    violations: List[Violation] = []
    equipment_contex: List[str] = []

class Regulation(BaseModel):
    """OSHA regulation reference."""
    citation: str
    text: str
    source: str

class ProcessingResult(BaseModel):
    """Final output object for the reporting agent."""
    video_id: str
    risk_assessment: RiskAssessment
    regulations: List[Regulation]
    report_path: Optional[str] = None
    alerts_sent: List[str] = []
