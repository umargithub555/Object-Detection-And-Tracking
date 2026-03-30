from beanie import Document
from pydantic import Field

class SystemSettings(Document):
    detection_humans: bool = True
    detection_vehicle: bool = True
    detection_animals: bool = True
    detection_birds: bool = True
    detection_sensitivity: float = Field(0.5, ge=0.0, le=1.0)
    count_humans: bool = True
    count_vehicle: bool = True
    count_animals: bool = True
    count_birds: bool = True
    counting_interval: int = Field(5, ge=1)

    class Settings:
        name = "system_settings"
