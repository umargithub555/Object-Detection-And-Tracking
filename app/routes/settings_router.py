from fastapi import APIRouter, HTTPException
from app.models.settings import SystemSettings
from app.schemas.schemas import SettingsResponse, SettingsUpdate

router = APIRouter(prefix="/settings", tags=["Admin Settings"])

async def get_or_create_settings():
    settings = await SystemSettings.find_one()
    if not settings:
        settings = SystemSettings()
        await settings.insert()
    return settings

@router.get("/", response_model=SettingsResponse)
async def get_settings():
    return await get_or_create_settings()

@router.patch("/", response_model=SettingsResponse)
async def update_settings(update_data: SettingsUpdate):
    settings = await get_or_create_settings()
    
    update_dict = update_data.model_dump(exclude_unset=True)
    for key, value in update_dict.items():
        setattr(settings, key, value)
    
    await settings.save()
    return settings
