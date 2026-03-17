from fastapi import FastAPI
from app.auth import auth_router
from app.core.logging import setup_logging, get_logger
from app.db.mongo_connection import mongo_lifespan


setup_logging()

logger = get_logger(__name__)

logger.info("AI Object Detection is starting...")

app = FastAPI(lifespan=mongo_lifespan)



@app.get("/health")
def home():
    return {"message": "Object Detection API", "status" : "OK"}


app.include_router(auth_router.router)