import os
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import Counter, defaultdict

# Configuration & Fixes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class VideoProcessingService:
    def __init__(self, model_path="yolo11x.pt"):
        # Initialize Model
        self.model = YOLO(model_path).to('cuda')
        
        # --- IMPROVEMENT: Tune ByteTrack parameters ---
        # lost_track_buffer: How many frames to keep a lost track in memory. 
        # Increasing this helps prevent ID switches during occlusion.
        self.byte_tracker = sv.ByteTrack(lost_track_buffer=90)
        
        # Define Production-Style Annotators
        self.colors = sv.ColorPalette.from_hex(["#00FFCC", "#FF3366", "#33FF57", "#7B61FF"])
        
        self.corner_annotator = sv.BoxCornerAnnotator(
            color=self.colors,
            thickness=2,
            corner_length=30
        )
        
        self.label_annotator = sv.LabelAnnotator(
            color=self.colors,
            text_thickness=1,
            text_scale=0.5,
            text_padding=10,
            border_radius=4
        )
        
        self.trace_annotator = sv.TraceAnnotator(
            color=self.colors,
            thickness=2,
            trace_length=20,
            position=sv.Position.CENTER
        )

    def process_video(self, source_path: str, target_path: str, confidence_threshold: float = 0.35):
        # Get video metadata
        video_info = sv.VideoInfo.from_video_path(source_path)
        generator = sv.get_video_frames_generator(source_path)

        # Global tracking state
        # tracker_id -> list of class_names seen for this ID
        tracker_id_to_classes = defaultdict(list)
        
        print(f"Processing started: {source_path} ({video_info.resolution_wh})")
        
        with sv.VideoSink(target_path=target_path, video_info=video_info) as sink:
            for index, frame in enumerate(generator):
                # Inference (Using FP16 for speed)
                results = self.model(frame, device=0, half=True, verbose=False)[0]
                
                # Update Detections
                detections = sv.Detections.from_ultralytics(results)
                
                # --- IMPROVEMENT: Confidence Thresholding ---
                # Filter out low-confidence detections to reduce false positives (like "fire hydrant")
                detections = detections[detections.confidence > confidence_threshold]
                
                # Update Tracker
                detections = self.byte_tracker.update_with_detections(detections)

                # Store classes for each tracker ID for "Majority Vote" smoothing
                labels = []
                for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
                    class_name = self.model.model.names[class_id]
                    tracker_id_to_classes[tracker_id].append(class_name)
                    
                    # Live Smoothing: Use the most common class seen so far for this ID
                    smoothed_class = Counter(tracker_id_to_classes[tracker_id]).most_common(1)[0][0]
                    labels.append(f"ID:{tracker_id} {smoothed_class.upper()}")

                # Annotate Frame
                annotated_frame = frame.copy()
                
                # Layers: Trace -> Corners -> Labels
                annotated_frame = self.trace_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = self.corner_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                
                # Overlay Global Object Counts (Smoothed)
                # Determine "won" class for all unique tracker IDs seen so far
                live_unique_objects = {tid: Counter(classes).most_common(1)[0][0] 
                                      for tid, classes in tracker_id_to_classes.items()}
                global_counts = Counter(live_unique_objects.values())
                
                y_offset = 40
                cv2.putText(annotated_frame, "GLOBAL COUNTS (TOTAL):", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30
                for class_name, count in sorted(global_counts.items()):
                    cv2.putText(
                        annotated_frame, 
                        f"{class_name.upper()}: {count}", 
                        (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0, 255, 0), 
                        2, 
                        cv2.LINE_AA
                    )
                    y_offset += 25

                # Show real-time processing
                cv2.imshow("Video Processing preview", annotated_frame)
                
                # Save to File
                sink.write_frame(frame=annotated_frame)

                # Allow quitting via 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Processing interrupted by user.")
                    break

        # --- IMPROVEMENT: Class Smoothing (Majority Vote) ---
        # For each tracker_id, we determine the final class based on what was seen most often.
        # This prevents a "car" being briefly misclassified as a "bus" from creating a bus entry.
        final_unique_objects = {}
        for tracker_id, classes in tracker_id_to_classes.items():
            most_common_class = Counter(classes).most_common(1)[0][0]
            final_unique_objects[tracker_id] = most_common_class

        # Final counts summary
        final_counts = dict(Counter(final_unique_objects.values()))
        
        print(f"Processing Complete. File saved: {target_path}")
        print(f"Final Counts (Smoothed): {final_counts}")
        
        # Cleanup CV2 windows
        cv2.destroyAllWindows()
        
        return final_counts
