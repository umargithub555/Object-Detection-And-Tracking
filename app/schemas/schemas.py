from pydantic import BaseModel, EmailStr
from typing import Optional



class AdminSeedRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str


class SettingsResponse(BaseModel):
    detection_humans: bool
    detection_vehicle: bool
    detection_animals: bool
    detection_birds: bool
    detection_sensitivity: float
    count_humans: bool
    count_vehicle: bool
    count_animals: bool
    count_birds: bool
    counting_interval: int


class SettingsUpdate(BaseModel):
    detection_humans: Optional[bool] = None
    detection_vehicle: Optional[bool] = None
    detection_animals: Optional[bool] = None
    detection_birds: Optional[bool] = None
    detection_sensitivity: Optional[float] = None
    count_humans: Optional[bool] = None
    count_vehicle: Optional[bool] = None
    count_animals: Optional[bool] = None
    count_birds: Optional[bool] = None
    counting_interval: Optional[int] = None