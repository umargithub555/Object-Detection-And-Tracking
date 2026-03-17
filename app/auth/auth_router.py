from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from app.models.user import User
from app.services.auth_service import auth_service
from app.services.email_service import send_otp_email
from datetime import datetime, timedelta, timezone
from app.utils.helper import generate_otp, get_otp_expire_time
import random

class Login(BaseModel):
    email: EmailStr
    password: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp: str

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.get("/admin-check")
def admin_check():
    return {"message":"Admin Auth APIs","status" : "Working Fine"}

@router.post("/login")
async def login(form_data: Login):
    try:
        user = await User.find_one(User.email == form_data.email)

        if not user or not auth_service.verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # print(user.email)

        access_token = auth_service.create_access_token(
            data={"sub": user.email}
        )
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))



@router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, background_tasks: BackgroundTasks):
    try:
        user = await User.find_one(User.email == request.email)
        if not user:
            # Prevent email enumeration by returning success even if user not found
            return {"message": "If this email is registered, an OTP will be sent."}
        
        # Generate 6 digit OTP
        otp = generate_otp()
        
        # Set expiration to 10 minutes from now
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
        
        user.otp_code = otp
        user.otp_expires_at = expires_at
        await user.save()
        
        # Send email in background
        background_tasks.add_task(send_otp_email, user.email, otp)
        
        return {"message": "If this email is registered, an OTP will be sent."}
        
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/verify-otp")
async def verify_otp(request: VerifyOTPRequest):
    try:
        user = await User.find_one(User.email == request.email)
        if not user:
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        
        if user.otp_code != request.otp:
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid OTP")
        
        now = datetime.now(timezone.utc)
             
        if user.otp_expires_at and user.otp_expires_at.replace(tzinfo=timezone.utc) < now:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OTP expired"
            )
            
        return {"message": "OTP verified successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest):
    try:
        user = await User.find_one(User.email == request.email)
        if not user:
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        if user.otp_code != request.otp:
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid OTP")
             
        now = datetime.now(timezone.utc)
             
        if user.otp_expires_at and user.otp_expires_at.replace(tzinfo=timezone.utc) < now:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OTP expired"
            )
            
        # Update password
        user.hashed_password = auth_service.hash_password(request.new_password)
        
        # Clear OTP fields
        user.otp_code = None
        user.otp_expires_at = None
        await user.save()
            
        return {"message": "Password reset successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


