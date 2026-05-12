# Advanced Object Detection, Tracking & Counting System

🚀 A professional-grade video analytics platform leveraging **YOLO11** and **BoxMOT** for robust multi-object tracking, counting, and behavioral analysis.

---

## 🌟 Key Features

### 1. Advanced Multi-Object Tracking (MOT)
*   **BoxMOT Integration**: Utilizes state-of-the-art trackers like **StrongSORT** and **BoT-SORT** for consistent ID retention across occlusions.
*   **OSNet Re-Identification**: Employs deep appearance features to maintain track stability in complex environments.

### 2. Intelligent Behavioral Analysis
*   **Employee Activity Detection**: Specialized module for detecting sitting vs. standing postures using **YOLO11 Pose Estimation**.
*   **Stability Debouncing**: Implements temporal window voting and commitment thresholds to prevent label flickering in posture analysis.

### 3. Dynamic Configuration & Control
*   **UI-Driven Filtering**: Enable or disable detection for specific categories (Humans, Vehicles, Animals, Birds) via a central settings dashboard.
*   **Sensitivity Tuning**: Adjustable confidence thresholds to balance detection recall and precision.

### 4. Enterprise Video Pipeline
*   **Cloudinary Integration**: Automatic upload of processed and annotated videos for easy sharing and cloud storage.
*   **FastAPI Backend**: High-performance asynchronous API for video processing and data management.
*   **Comprehensive Dashboard**: Track historical records, summary statistics, and object counts in a structured MongoDB database.

---

## 🛠 Technology Stack

*   **Core**: Python 3.11+
*   **Detection**: Ultralytics YOLO11 (Detection & Pose)
*   **Tracking**: BoxMOT (StrongSORT)
*   **Visualization**: OpenCV, Supervision
*   **Web Framework**: FastAPI
*   **Database**: MongoDB with Beanie ODM
*   **Storage**: Cloudinary
*   **Environment**: Optimized for NVIDIA CUDA (RTX 3090 Tested)

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.11 or higher
*   NVIDIA GPU with CUDA support (Recommended)
*   MongoDB Instance (Local or Atlas)
*   Cloudinary Account

### Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/umargithub555/Object-Detection-And-Tracking.git
    cd Object-Detection-And-Tracking
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables**:
    Create a `.env` file in the root directory:
    ```env
    MONGO_URL=your_mongodb_connection_string
    DATABASE_NAME=object_detection
    CLOUDINARY_CLOUD_NAME=your_name
    CLOUDINARY_API_KEY=your_key
    CLOUDINARY_API_SECRET=your_secret
    ```

4.  **Weights Setup**:
    Ensure the `weights/` directory contains the necessary model files:
    *   `yolov8m.pt` (or YOLO11 equivalents)
    *   `yolo11m-pose.pt`
    *   `osnet_x0_25_msmt17.pt`

### Running the Application

**Start the FastAPI Server**:
```bash
python main.py
```
*   API Documentation: `http://localhost:8000/docs`
*   Video Processing Endpoints: `/video/process-video` and `/video/process-employee-sittings`

**Run Standalone Tracking Script**:
```bash
python app/services/video_service_v2.py
```

---

## 📂 Project Structure

*   `app/services/`: Core logic for video processing and tracking.
*   `app/models/`: Database schemas and settings models.
*   `app/utils/`: Custom wrappers for Detector, Tracker, and Visualizer.
*   `app/core/`: Configuration management (`config.yaml`).
*   `weights/`: AI model weights for detection and Re-ID.

---

## 📜 License
This project is for internal and educational use. See individual model licenses (Ultralytics, BoxMOT) for commercial usage details.
