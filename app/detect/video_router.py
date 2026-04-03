import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from app.services.video_service import VideoProcessingService
from app.services.cloudinary_service import CloudinaryService
import os

router = APIRouter(prefix="/video", tags=["Video Processing"])
video_service = VideoProcessingService()
cloudinary_service = CloudinaryService()

# Paths for temporary storage
UPLOAD_DIR = "input"
OUTPUT_DIR = "output"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@router.get("/class-names")
def model_detection_class_names():
    class_list = video_service.all_objects()
    return {"Detection Classes": class_list}


@router.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    """
    Upload a video file, process it using YOLO detection and tracking,
    upload to Cloudinary, and return the object counts and metadata.
    """
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    input_path = os.path.join(UPLOAD_DIR, unique_filename)
    output_filename = f"processed_{unique_filename}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1. Process the video and get summary counts
        result = await video_service.process_video(input_path, output_path)

        # 2. Upload to Cloudinary
        print(f"Uploading to Cloudinary: {output_path}")
        cloudinary_url = cloudinary_service.upload_video(output_path)
        print(f"Cloudinary Upload Complete: {cloudinary_url}")

        # 3. Return the summary and Cloudinary URL
        return JSONResponse(
            status_code=200,
            content={
                "message": "Processing and upload complete",
                "filename": file.filename,
                "metadata": result["metadata"],
                "summary": result["summary"],
                "counts": result["counts"],
                "cloudinary_url": cloudinary_url,
                "download_url": f"/video/download/{output_filename}"
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Processing failed: {str(e)}"}
        )
    finally:
        # Cleanup input file
        if os.path.exists(input_path):
            os.remove(input_path)
        # Optional: Cleanup output file as well if you only want to serve from Cloudinary
        # if os.path.exists(output_path):
        #     os.remove(output_path)

@router.get("/download/{filename}")
async def download_video(filename: str):
    """
    Download a processed video file.
    """
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"message": "File not found"})
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="video/mp4"
    )
