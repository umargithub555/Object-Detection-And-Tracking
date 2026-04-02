from datetime import datetime
from typing import Dict, Optional
from beanie import Document
from pydantic import Field

class DetectionRecord(Document):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    counts: Dict[str, int] = Field(default_factory=dict)
    filename: str
    duration: str
    resolution: str
    fps: float
    processing_time: float # in seconds

    class Settings:
        name = "detection_records"
        indexes = [
            "timestamp",
            "filename"
        ]
