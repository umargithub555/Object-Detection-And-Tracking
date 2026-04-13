from fastapi import Depends, FastAPI, HTTPException
from app.auth import auth_router
from app.detect import video_router
from app.routes import settings_router, dashboard_router
from app.core.logging import setup_logging, get_logger
from app.db.mongo_connection import mongo_lifespan
from app.models.user import User
from app.schemas.schemas import AdminSeedRequest
from app.services.auth_service import auth_service
from fastapi.middleware.cors import CORSMiddleware
from app.auth.auth_router import get_current_user




setup_logging()

logger = get_logger(__name__)

logger.info("AI Object Detection is starting...")

app = FastAPI(lifespan=mongo_lifespan)


#  Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def home():
    return {"message": "Object Detection API", "status" : "OK"}


@app.post("/seed-admin", status_code=201)
async def create_admin(admin_data: AdminSeedRequest):

    existing_admin = await User.find_one(User.email == admin_data.email.lower())

    if existing_admin:
        raise HTTPException(409, "Admin already exists")

    admin = User(
        email=admin_data.email.lower(),
        hashed_password=auth_service.hash_password(admin_data.password),
        full_name=admin_data.full_name,
        role="admin"
    )

    await admin.insert()

    return {
        "message": "Admin created successfully"
    }



@app.get("/current-user") 
async def get_current_user(current_user : User = Depends(get_current_user)):
    try:
        if current_user:
            email, full_name =  current_user.email, current_user.full_name
            
        return {"email": email, "full_name": full_name}

    except Exception as e:
        raise HTTPException(404, detail=f"Unable to fetch current user data {e}")


app.include_router(auth_router.router)
app.include_router(video_router.router)
app.include_router(settings_router.router)
app.include_router(dashboard_router.router)
