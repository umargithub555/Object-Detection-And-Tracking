from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from jose import jwt, JWTError
from app.core.config import settings
import bcrypt



pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    # def hash_password(self, password: str) -> str:
    #     return pwd_context.hash(password)

    
    def hash_password(self, password: str) -> str:
    # Convert password to bytes, salt it, and hash it
        pwd_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(pwd_bytes, salt)
        # Return as string for database storage
        return hashed_password.decode('utf-8')

    # def verify_password(self, plain_password: str, hashed_password: str) -> bool:
    #     return pwd_context.verify(plain_password, hashed_password)


    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )

    def create_access_token(self, data: dict, expires_delta: timedelta | None = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})

        encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.algorithm)

        return encoded_jwt
        
    def decode_access_token(self, token: str) -> str | None:
        try:
            payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
            email: str = payload.get("sub")
            if email is None:
                return None
            return email
        except JWTError:
            return None

auth_service = AuthService()
