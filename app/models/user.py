from beanie import Document
from pydantic import EmailStr, Field
from datetime import datetime, timezone
from typing import Optional

class User(Document):
    """User model for authentication and profile management"""
    
    email: EmailStr
    hashed_password: str
    full_name: str
    profile_picture: Optional[str] = None

    deleted_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    otp_code: Optional[str] = None
    otp_expires_at: Optional[datetime] = None


    class Settings:
        name = "users"
        indexes = [
            "email"
        ]

   