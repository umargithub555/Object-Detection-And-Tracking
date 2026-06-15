import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
import json
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
        result = await video_service.process_video(input_path, output_path, file.filename)

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

@router.post("/process-employee-sittings")
async def process_employee_sittings(file: UploadFile = File(...)):
    """
    Upload a video file, process it using YOLO Pose estimation and tracking
    specifically for employee sitting/standing detection.
    """
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"sitting_{uuid.uuid4()}{file_extension}"
    input_path = os.path.join(UPLOAD_DIR, unique_filename)
    output_filename = f"processed_{unique_filename}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1. Process the video for sittings
        result = await video_service.process_employee_sittings(input_path, output_path, file.filename)

        # 2. Upload to Cloudinary
        print(f"Uploading to Cloudinary: {output_path}")
        cloudinary_url = cloudinary_service.upload_video(output_path)
        print(f"Cloudinary Upload Complete: {cloudinary_url}")

        # 3. Return results
        return JSONResponse(
            status_code=200,
            content={
                "message": "Employee sitting detection and upload complete",
                "filename": file.filename,
                "metadata": result["metadata"],
                "counts": result["counts"],
                "summary": result["summary"],
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

@router.post("/process-desk-monitoring")
async def process_desk_monitoring(
    file: UploadFile = File(...),
    regions_file: UploadFile = File(None)
):
    """
    Upload a video file and optional regions JSON, process it for desk occupancy,
    upload to Cloudinary, and return the occupancy statistics.
    """
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"desk_{uuid.uuid4()}{file_extension}"
    input_path = os.path.join(UPLOAD_DIR, unique_filename)
    output_filename = f"processed_{unique_filename}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Save uploaded video file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    regions_data = None
    if regions_file:
        try:
            regions_content = await regions_file.read()
            regions_data = json.loads(regions_content)
        except Exception as e:
            return JSONResponse(status_code=400, content={"message": f"Invalid regions JSON: {str(e)}"})

    try:
        # 1. Process the video for desk monitoring
        result = await video_service.process_desk_monitoring(input_path, output_path, file.filename, regions_data)

        if "error" in result:
            return JSONResponse(status_code=500, content=result)

        # 2. Upload to Cloudinary
        print(f"Uploading to Cloudinary: {output_path}")
        cloudinary_url = cloudinary_service.upload_video(output_path)
        print(f"Cloudinary Upload Complete: {cloudinary_url}")

        # 3. Return results
        return JSONResponse(
            status_code=200,
            content={
                "message": "Desk monitoring and upload complete",
                "filename": file.filename,
                "metadata": result["metadata"],
                "desk_stats": result["desk_stats"],
                "overall_total_time": result["overall_total_time"],
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

