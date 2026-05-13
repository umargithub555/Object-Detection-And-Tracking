import os
# Sanitize CUDA_VISIBLE_DEVICES before importing torch/boxmot
if os.environ.get('CUDA_VISIBLE_DEVICES') == 'cuda':
    del os.environ['CUDA_VISIBLE_DEVICES']

import sys
from pathlib import Path

# Add project root to sys.path to allow imports from 'utils'
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import time
import cv2
import torch
from utils.data_loader import load_config
from utils.detector import Detector
from utils.tracker import TrackerWrapper
from utils.visualizer import Visualizer
from utils.video_writer import VideoWriter



def track_video(video_path: str, output_path: str = None):
    # 1. Load config and setup paths
    cfg = load_config()
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"Error: Video file {video_path} not found.")
        return

    if output_path is None:
        output_dir = Path(cfg["paths"]["videos_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"tracked_{video_file.stem}.mp4"
    else:
        output_file = Path(output_path)

    # 2. Initialize components
    print("\n--- Initializing Components ---")
    detector   = Detector(cfg)
    tracker    = TrackerWrapper(cfg)
    visualizer = Visualizer(cfg)

    # 3. Open Video
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Input Video : {video_file}")
    print(f"  Resolution  : {width}x{height}")
    print(f"  FPS         : {fps:.2f}")
    print(f"  Total Frames: {total_frames}")

    # 4. Run Tracking
    print("\n--- Running Tracking ---")
    frame_id = 0
    start_time = time.time()

    with VideoWriter(cfg, output_file, fps, (width, height)) as vw:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            t0 = time.perf_counter()

            # Detection
            dets = detector.detect(frame)
            
            # Tracking
            trks = tracker.update(dets, frame, frame_id)
            
            # Visualization
            vis = visualizer.draw(frame, trks)
            
            # Writing
            vw.write(vis)

            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000

            if frame_id % 30 == 0 or frame_id == total_frames:
                progress = (frame_id / total_frames) * 100
                print(f"  [{progress:5.1f}%] Frame {frame_id:04d}/{total_frames} | {elapsed_ms:4.0f}ms", end="\r")

    cap.release()
    end_time = time.time()
    total_time = end_time - start_time
    avg_fps = frame_id / total_time

    print(f"\n\n--- Done ---")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Output saved to: {output_file}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Track objects in a custom video file.")
    # parser.add_argument("input", help="Path to the input video file (e.g. video.mp4)")
    # parser.add_argument("--output", help="Optional path to save the output video", default=None)
    
    # args = parser.parse_args()
    track_video(r"input\test.mp4", "results/res_botsort_2.mp4")

