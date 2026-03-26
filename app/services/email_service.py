from fastapi_mail import FastMail, MessageSchema, MessageType
from app.core.config import conf

async def send_otp_email(email: str, otp: str):
    html = f"""
    <p>Your OTP for password reset is: <strong>{otp}</strong></p>
    <p>This OTP will expire in 10 minutes.</p>
    """
    message = MessageSchema(
        subject="Password Reset OTP",
        recipients=[email],
        body=html,
        subtype=MessageType.html
    )

    fm = FastMail(conf)
    await fm.send_message(message)
