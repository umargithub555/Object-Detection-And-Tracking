# database.py
import motor.motor_asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from beanie import init_beanie
import os
from app.core.config import settings
from app.core.logging import get_logger
from app.models.user import User
from app.models.settings import SystemSettings
from app.models.detection_record import DetectionRecord



logger = get_logger(__name__)


class DatabaseClient:
    """A singleton-like class to hold the Motor client and database instance."""
    client: AsyncIOMotorClient = None
    database: AsyncIOMotorDatabase = None

DB_CLIENT = DatabaseClient()


async def init_db():
    """Initialize Beanie with the User model"""
    await init_beanie(
        database=DB_CLIENT.database,
        document_models=[
            User,
            SystemSettings,
            DetectionRecord
        ]
    )
    logger.info("✅ Beanie initialized with User model")


# Recommended modern approach: FastAPI Lifespan context manager
@asynccontextmanager
async def mongo_lifespan(app: FastAPI):
    # Startup event: Connect to the database
    logger.info("Connecting to MongoDB...")
    DB_CLIENT.client = AsyncIOMotorClient(settings.mongo_url)
    DB_CLIENT.database = DB_CLIENT.client[settings.database_name]
    
    # Optional: Ping the database to confirm connection
    try:
        await DB_CLIENT.client.admin.command('ping')
        logger.info("✅ MongoDB connection established successfully")
        
        # Initialize Beanie
        await init_db()
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB: {e}")
        raise

    yield  # Application starts here

    # Shutdown event: Close the database connection
    logger.info("Closing MongoDB connection...")
    DB_CLIENT.client.close()
    logger.info("✅ MongoDB connection closed")


def get_mongo_db() -> AsyncIOMotorDatabase:
    """Dependency injection function to yield the database instance."""
    if DB_CLIENT.database is None:
        raise Exception("Database not initialized")
    return DB_CLIENT.database
