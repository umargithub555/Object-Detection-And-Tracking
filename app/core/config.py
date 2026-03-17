from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from fastapi_mail import ConnectionConfig
from pydantic import Field
import enum


class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "Object Detection API"
    APP_DESCRIPTION: str = "API for Object Detection system"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: enum = ['development','production']


    mongo_url: str =  Field(..., alias="MONGO_URL")
    database_name: str = Field(..., alias='DATABASE_NAME')

    secret_key:str = Field(..., alias="SECRET_KEY")
    algorithm:str = Field(..., alias="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(1440, alias="ACCESS_TOKEN_EXPIRE_MINUTES") # Default 24 hours


    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore'
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings class.
    This ensures that settings are loaded only once,
    improving performance.
    """
    return Settings()





conf = ConnectionConfig(
    
    MAIL_USERNAME="shayanumar277@gmail.com",
    MAIL_PASSWORD="ctax tfwk yqrs bche",  # use App Password for Gmail
    MAIL_FROM="shayanumar277@gmail.com",
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True
)



settings = get_settings()