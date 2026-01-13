import os
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Workaround for OpenMP multiple library initialization error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration
SOURCE_VIDEO_PATH = "traffic_analysis.mp4"
TARGET_VIDEO_PATH = "output.mp4"
MODEL_NAME = "yolov9c.pt"  # Upgraded to YOLOv9 for better accuracy

def process_video():
    # 1. Initialize Model
    model = YOLO(MODEL_NAME).to('cuda')
    model.fuse()
    
    # 2. Setup Video Info
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    # 3. Initialize Annotators
    box_annotator = sv.BoxAnnotator(thickness=1)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)
    
    # 4. Tracking Data
    # Persistent mapping to lock a class name to a specific tracker_id
    tracker_id_to_class = {}

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        # Inference & Tracking
        results = model.track(frame, persist=True, verbose=False, device=0)[0]
        
        # Check if any detections were made
        if results.boxes.id is None:
            return frame
            
        detections = sv.Detections.from_ultralytics(results)
        
        # Update stabilized mappings
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
            if tracker_id not in tracker_id_to_class:
                tracker_id_to_class[tracker_id] = model.model.names[class_id]
        
        # Build Labels using the stabilized/locked classes
        labels = [
            f"#{tracker_id} {tracker_id_to_class.get(tracker_id, model.model.names[class_id])} {conf:.2f}"
            for conf, tracker_id, class_id 
            in zip(detections.confidence, detections.tracker_id, detections.class_id)
        ]
        
        # Annotate Frame
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        # Calculate Current Totals from stabilized mapping
        class_counts = {}
        for class_name in tracker_id_to_class.values():
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Draw Historical Count Overlay (Top Left)
        for i, (class_name, count) in enumerate(class_counts.items()):
            text = f"Total {class_name}: {count}"
            cv2.putText(
                annotated_frame, text, (20, 40 + (i * 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
        # # Optional: Show live preview
        # cv2.imshow("Detection and Tracking", annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     return annotated_frame

        return annotated_frame

    # # 5. Execute Processing
    # print(f"Processing video: {SOURCE_VIDEO_PATH}")
    # sv.process_video(
    #     source_path=SOURCE_VIDEO_PATH,
    #     target_path=TARGET_VIDEO_PATH,
    #     callback=callback
    # )
    # cv2.destroyAllWindows()
    # 5. Execute Processing with Manual Loop
    print(f"Processing video: {SOURCE_VIDEO_PATH}")
    
    # Create a video sink to save the output while we process
    with sv.VideoSink(target_path=TARGET_VIDEO_PATH, video_info=video_info) as sink:
        # Generate frames one by one
        for index, frame in enumerate(sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)):
            # Use your existing callback logic
            annotated_frame = callback(frame, index)
            
            # Save the frame to the output file
            sink.write_frame(frame=annotated_frame)
            
            # Show the frame live
            cv2.imshow("Live Traffic Analysis", annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print(f"Finished. Output saved to: {TARGET_VIDEO_PATH}")

if __name__ == "__main__":
    process_video()
