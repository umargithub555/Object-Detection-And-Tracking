from typing import Optional

from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks, File, UploadFile
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, Field
from app.models.user import User
from app.services.auth_service import auth_service
from app.services.email_service import send_otp_email
from app.services.cloudinary_service import CloudinaryService
from datetime import datetime, timedelta, timezone
from app.utils.helper import generate_otp, get_otp_expire_time
import random

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
cloudinary_service = CloudinaryService()

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
    otp: Optional[str] = None
    new_password: str

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6)
    confirm_password: str



async def get_current_user(token: str = Depends(oauth2_scheme)):
    email = auth_service.decode_access_token(token)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = await User.find_one(User.email == email)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


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
        
        # Generate 4 digit OTP
        otp = generate_otp()

        print("OTP :", otp)
        
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
        
        # if user.otp_code != request.otp:
        #      raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid OTP")
             
        now = datetime.now(timezone.utc)
             
        # if user.otp_expires_at and user.otp_expires_at.replace(tzinfo=timezone.utc) < now:
        #     raise HTTPException(
        #         status_code=status.HTTP_400_BAD_REQUEST,
        #         detail="OTP expired"
        #     )
            
        # Update password
        user.hashed_password = auth_service.hash_password(request.new_password)
        
        # Clear OTP fields
        # user.otp_code = None
        # user.otp_expires_at = None
        await user.save()
            
        return {"message": "Password reset successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))




@router.post("/change-password")
async def change_password(request: ChangePasswordRequest, current_user: User = Depends(get_current_user)):
    try:
        user = await User.find_one(User.email == current_user.email)
        if not user:
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        if not auth_service.verify_password(request.current_password, user.hashed_password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect current password")
        
        if request.new_password != request.confirm_password:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Passwords do not match")
        
        user.hashed_password = auth_service.hash_password(request.new_password)
        await user.save()
        
        return {"message": "Password changed successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    try:
        return {"message": "Logout successful"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))



@router.post("/refresh-token")
async def refresh_token(current_user: User = Depends(get_current_user)):
    try:
        access_token = auth_service.create_access_token(
            data={"sub": current_user.email}
        )
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/add-profile-picture")
async def add_profile_picture(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    try:
        user = await User.find_one(User.email == current_user.email)
        if not user:
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        file_content = await file.read()
        profile_picture = cloudinary_service.upload_image(file_content, file.filename)
        user.profile_picture = profile_picture
        await user.save()
        
        return {"message": "Profile picture added successfully", "profile_picture": profile_picture}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))



@router.post("/remove-profile-picture")
async def remove_profile_picture(current_user: User = Depends(get_current_user)):
    try:
        user = await User.find_one(User.email == current_user.email)
        if not user:
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        user.profile_picture = None
        await user.save()
        
        return {"message": "Profile picture removed successfully"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/get-profile-picture")
async def get_profile_picture(current_user: User = Depends(get_current_user)):
    try:
        user = await User.find_one(User.email == current_user.email)
        if not user:
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        return {"profile_picture": user.profile_picture}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.put("/update-profile-picture")
async def update_profile_picture(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    try:
        user = await User.find_one(User.email == current_user.email)
        if not user:
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        file_content = await file.read()
        profile_picture = cloudinary_service.upload_image(file_content, file.filename)
        user.profile_picture = profile_picture
        await user.save()
        
        return {"message": "Profile picture updated successfully", "profile_picture": profile_picture}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))