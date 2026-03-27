import os
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
from app.core.config import settings

load_dotenv()

class CloudinaryService:
    def __init__(self):
        cloudinary.config(
            cloud_name=settings.CLOUDINARY_CLOUD_NAME,
            api_key=settings.CLOUDINARY_API_KEY,
            api_secret=settings.CLOUDINARY_API_SECRET,
            secure=True
        )

    def upload_video(self, file_path: str):
        """
        Uploads a video to Cloudinary and returns the secure URL with optimization.
        """
        try:
            # Remove extension from filename to prevent .mp4.mp4 issue
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            
            result = cloudinary.uploader.upload_large(
                file_path,
                resource_type="video",
                public_id=f"processed_videos/{base_filename}",
                overwrite=True,
                # Transformation to ensure browser compatibility (H.264, optimized)
                transformation=[
                    {"fetch_format": "auto", "quality": "auto"}
                ]
            )
            return result.get("secure_url")
        except Exception as e:
            print(f"Cloudinary upload failed: {e}")
            return None
