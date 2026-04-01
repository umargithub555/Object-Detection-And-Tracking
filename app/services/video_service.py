import os
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import Counter, defaultdict
from app.models.settings import SystemSettings
from app.models.detection_record import DetectionRecord
import time

# Configuration & Fixes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class VideoProcessingService:
    # Map UI categories to COCO categories
    CATEGORY_MAPPING = {
        "humans": ["person"],
        "vehicle": ["car", "motorcycle", "bus", "truck", "bicycle", "train"],
        "animals": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
        "birds": ["bird"]
    }

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
        

    def all_objects(self):
        return self.model.names

    async def process_video(self, source_path: str, target_path: str, confidence_threshold: float = 0.35):
        # Fetch Admin Settings from DB
        settings = await SystemSettings.find_one()
        if not settings:
            settings = SystemSettings()
            # We don't necessarily need to insert here, just use defaults
        
        # Override threshold if sensitivity is set (merging logic)
        # Using settings.detection_sensitivity as the primary threshold
        confidence_threshold = settings.detection_sensitivity

        # Map UI categories to COCO categories
        category_mapping = self.CATEGORY_MAPPING

        enabled_classes = []
        if settings.detection_humans: enabled_classes.extend(category_mapping["humans"])
        if settings.detection_vehicle: enabled_classes.extend(category_mapping["vehicle"])
        if settings.detection_animals: enabled_classes.extend(category_mapping["animals"])
        if settings.detection_birds: enabled_classes.extend(category_mapping["birds"])
        
        # Remove duplicates
        enabled_classes = list(set(enabled_classes))

        # Filter for counting
        counting_enabled_classes = []
        if settings.count_humans: counting_enabled_classes.extend(category_mapping["humans"])
        if settings.count_vehicle: counting_enabled_classes.extend(category_mapping["vehicle"])
        if settings.count_animals: counting_enabled_classes.extend(category_mapping["animals"])
        if settings.count_birds: counting_enabled_classes.extend(category_mapping["birds"])
        counting_enabled_classes = list(set(counting_enabled_classes))

        # Get video metadata
        video_info = sv.VideoInfo.from_video_path(source_path)
        generator = sv.get_video_frames_generator(source_path)

        # Global tracking state
        # tracker_id -> list of class_names seen for this ID
        tracker_id_to_classes = defaultdict(list)
        # tracker_id -> list of confidence scores seen for this ID
        tracker_id_to_confidences = defaultdict(list)
        
        print(f"Processing started: {source_path} ({video_info.resolution_wh})")
        start_time = time.time()
        
        with sv.VideoSink(target_path=target_path, video_info=video_info) as sink:
            for index, frame in enumerate(generator):
                # Inference (Using FP16 for speed)
                results = self.model(frame, device=0, half=True, verbose=False)[0]
                
                # Update Detections
                detections = sv.Detections.from_ultralytics(results)
                
                # --- IMPROVEMENT: Confidence Thresholding ---
                detections = detections[detections.confidence > confidence_threshold]
                
                # --- NEW: Class Filtering based on Admin Settings ---
                if enabled_classes:
                    mask = np.array([self.model.model.names[cid] in enabled_classes for cid in detections.class_id])
                    detections = detections[mask]
                else:
                    # If everything is disabled, we might want to return empty detections
                    # but usually, enabled_classes will have at least the defaults.
                    pass
                
                # Update Tracker
                detections = self.byte_tracker.update_with_detections(detections)

                # Store classes and confidences for each tracker ID
                labels = []
                for class_id, tracker_id, confidence in zip(detections.class_id, detections.tracker_id, detections.confidence):
                    class_name = self.model.model.names[class_id]
                    tracker_id_to_classes[tracker_id].append(class_name)
                    tracker_id_to_confidences[tracker_id].append(float(confidence))
                    
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
                live_unique_objects = {tid: Counter(classes).most_common(1)[0][0] 
                                      for tid, classes in tracker_id_to_classes.items()}
                
                # Filter counts based on counting rules
                filtered_live_objects = {tid: cls for tid, cls in live_unique_objects.items() if cls in counting_enabled_classes}
                global_counts = Counter(filtered_live_objects.values())
                
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

        # --- FINAL AGGREGATION ---
        # For each tracker_id, we determine the final class and average confidence
        final_unique_objects = {}
        for tracker_id, classes in tracker_id_to_classes.items():
            most_common_class = Counter(classes).most_common(1)[0][0]
            avg_conf = sum(tracker_id_to_confidences[tracker_id]) / len(tracker_id_to_confidences[tracker_id])
            final_unique_objects[tracker_id] = {
                "class": most_common_class,
                "confidence": avg_conf
            }

        # Calculate final counts and average confidence per UI category
        category_stats = defaultdict(lambda: {"count": 0, "total_confidence": 0.0})
        
        for obj in final_unique_objects.values():
            class_name = obj["class"]
            confidence = obj["confidence"]
            
            # Map the class to one or more UI categories
            for category, sub_classes in self.CATEGORY_MAPPING.items():
                if class_name in sub_classes:
                    category_stats[category]["count"] += 1
                    category_stats[category]["total_confidence"] += confidence

        detailed_summary = []
        for category, data in category_stats.items():
            if data["count"] > 0:
                detailed_summary.append({
                    "type": category.capitalize(),
                    "count": data["count"],
                    "avg_confidence": f"{(data['total_confidence'] / data['count']) * 100:.1f}%"
                })

        # Video metadata summary
        file_size_bytes = os.path.getsize(source_path)
        file_size_readable = f"{file_size_bytes / 1024:.2f} KB" if file_size_bytes < 1024 * 1024 else f"{file_size_bytes / (1024 * 1024):.2f} MB"
        
        duration_seconds = video_info.total_frames / video_info.fps
        duration_readable = f"{int(duration_seconds // 60)}:{int(duration_seconds % 60):02d}"

        metadata = {
            "file_size": file_size_readable,
            "duration": duration_readable,
            "resolution": f"{video_info.resolution_wh}",
            "fps": video_info.fps
        }
        
        print(f"Processing Complete. File saved: {target_path}")
        print(f"Final Detailed Summary: {detailed_summary}")
        
        # Cleanup CV2 windows
        cv2.destroyAllWindows()
        
        # Save record to DB
        counts = {cat: data["count"] for cat, data in category_stats.items()}
        processing_duration = time.time() - start_time
        record = DetectionRecord(
            counts=counts,
            filename=os.path.basename(source_path),
            duration=metadata["duration"],
            resolution=metadata["resolution"],
            fps=metadata["fps"],
            processing_time=round(processing_duration, 2)
        )
        await record.insert()
        print(f"Detection record saved to database: {record.id}")

        return {
            "metadata": metadata,
            "summary": detailed_summary,
            "counts": counts
        }
