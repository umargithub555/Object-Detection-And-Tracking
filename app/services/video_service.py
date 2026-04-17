import os
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import Counter, defaultdict, deque
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
        # Initialize Detection Model
        self.model = YOLO(model_path).to('cuda')
        
        # Initialize Pose Model for Employee Sittings
        # Fix 5: Upgraded to yolo11x-pose for significantly better keypoint accuracy in crowds
        self.pose_model = YOLO("yolo11m-pose.pt").to('cuda')
        
        # --- CONFIGURATION: ByteTrack parameters ---
        self.tracker_buffer = 90 # frames to keep lost tracks
        
        # Define Production-Style Annotators
        self.colors = sv.ColorPalette.from_hex(["#00FFCC", "#FF3366", "#33FF57", "#7B61FF"])
        
        self.box_annotator = sv.BoxAnnotator(
            color=self.colors,
            thickness=1
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

    def classify_posture(self, kp, box_xyxy):
        """
        Determine if a person is sitting or standing using keypoints.
        Simplified logic from employee_sittings.py
        """
        def get_pt(idx, min_conf=0.3):
            if kp[idx][2] >= min_conf:
                return kp[idx][0], kp[idx][1]
            return None

        x1, y1, x2, y2 = box_xyxy
        box_h = y2 - y1
        box_w = x2 - x1
        aspect = box_h / (box_w + 1e-5)

        hip      = get_pt(11) or get_pt(12)
        knee     = get_pt(13) or get_pt(14)
        ankle    = get_pt(15) or get_pt(16)
        shoulder = get_pt(5)  or get_pt(6)

        if hip and knee:
            if ankle:
                leg_len   = abs(ankle[1] - hip[1])
                torso_len = abs(hip[1] - shoulder[1]) if shoulder else box_h * 0.3
                if leg_len > torso_len * 1.2:
                    return "standing"
                else:
                    return "sitting"
            
            # If no ankle, use hip-to-knee vertical distance
            vertical_diff = abs(knee[1] - hip[1])
            if vertical_diff < box_h * 0.25:
                return "sitting"

        # Fallback to aspect ratio
        return "standing" if aspect > 1.8 else "sitting"

    def _draw_hud(self, frame, total, standing, sitting):
        """Shadowed HUD counter in the top-left corner."""
        STANDING_COLOR = (0, 230, 80)
        SITTING_COLOR  = (255, 140, 0)
        hud = [
            (f"TOTAL DETECTED: {total}", (220, 220, 220)),
            (f"STANDING      : {standing}", STANDING_COLOR),
            (f"SITTING       : {sitting}",  SITTING_COLOR),
        ]
        for idx, (text, color) in enumerate(hud):
            y = 40 + idx * 30
            # Shadow
            cv2.putText(frame, text, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            # Text
            cv2.putText(frame, text, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    def _draw_pill_label(self, frame, text, x1, y1, color):
        """Semi-transparent pill label above the corner box."""
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        pad_x, pad_y = 8, 5
        lx1 = x1
        ly1 = y1 - th - pad_y * 2 - baseline
        lx2 = x1 + tw + pad_x * 2
        ly2 = y1

        if ly1 < 0:
            ly1 = y1
            ly2 = y1 + th + pad_y * 2 + baseline

        overlay = frame.copy()
        cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, text, (lx1 + pad_x, ly2 - baseline - pad_y),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    async def process_video(self, source_path: str, target_path: str, original_filename, confidence_threshold: float = 0.35):
        # Fetch Admin Settings from DB
        settings = await SystemSettings.find_one()
        if not settings:
            settings = SystemSettings()
        
        confidence_threshold = settings.detection_sensitivity

        # Map UI categories to COCO categories
        category_mapping = self.CATEGORY_MAPPING

        enabled_classes = []
        if settings.detection_humans: enabled_classes.extend(category_mapping["humans"])
        if settings.detection_vehicle: enabled_classes.extend(category_mapping["vehicle"])
        if settings.detection_animals: enabled_classes.extend(category_mapping["animals"])
        if settings.detection_birds: enabled_classes.extend(category_mapping["birds"])
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

        # Fix 1: BoT-SORT is managed internally by Ultralytics .track(persist=True)
        # Reset predictor state to avoid leaking tracks from a previous video
        self.model.predictor = None

        # Global tracking state
        tracker_id_to_classes = defaultdict(list)
        tracker_id_to_confidences = defaultdict(list)
        
        print(f"Processing started: {source_path} ({video_info.resolution_wh})")
        start_time = time.time()
        
        with sv.VideoSink(target_path=target_path, video_info=video_info) as sink:
            for index, frame in enumerate(generator):
                # Fix 1+2: BoT-SORT via native .track(); iou=0.45 kills duplicate boxes at 1024px
                results = self.model.track(frame, imgsz=1024, iou=0.45, device=0, half=True,
                                           persist=True, tracker="botsort.yaml", verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)

                # Skip frame if BoT-SORT hasn't assigned IDs yet (first 1-2 frames)
                if detections.tracker_id is None:
                    sink.write_frame(frame=frame)
                    continue

                # Filter by enabled UI classes
                if enabled_classes:
                    mask = np.array([self.model.model.names[cid] in enabled_classes for cid in detections.class_id])
                    detections = detections[mask]

                # Filter by confidence threshold (post-tracking, so BoT-SORT already bridged weak frames)
                detections = detections[detections.confidence > confidence_threshold]

                labels = []
                for class_id, tracker_id, confidence in zip(detections.class_id, detections.tracker_id, detections.confidence):
                    class_name = self.model.model.names[class_id]
                    tracker_id_to_classes[tracker_id].append(class_name)
                    tracker_id_to_confidences[tracker_id].append(float(confidence))
                    smoothed_class = Counter(tracker_id_to_classes[tracker_id]).most_common(1)[0][0]
                    labels.append(f"ID:{tracker_id} {smoothed_class.upper()}")

                annotated_frame = frame.copy()
                annotated_frame = self.trace_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                
                live_unique_objects = {
                    tid: Counter(classes).most_common(1)[0][0] 
                    for tid, classes in tracker_id_to_classes.items() 
                    if len(classes) >= 15  # Avoid ghost trackers popping in out
                }
                filtered_live_objects = {tid: cls for tid, cls in live_unique_objects.items() if cls in counting_enabled_classes}
                global_counts = Counter(filtered_live_objects.values())
                
                y_offset = 40
                cv2.putText(annotated_frame, "GLOBAL COUNTS (TOTAL):", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30
                for class_name, count in sorted(global_counts.items()):
                    cv2.putText(annotated_frame, f"{class_name.upper()}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    y_offset += 25

                cv2.imshow("Video Processing preview", annotated_frame)
                sink.write_frame(frame=annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        final_unique_objects = {}
        for tracker_id, classes in tracker_id_to_classes.items():
            if len(classes) < 15: continue # Ghost tracks exclusion
            
            most_common_class = Counter(classes).most_common(1)[0][0]
            avg_conf = sum(tracker_id_to_confidences[tracker_id]) / len(tracker_id_to_confidences[tracker_id])
            final_unique_objects[tracker_id] = {"class": most_common_class, "confidence": avg_conf}

        category_stats = defaultdict(lambda: {"count": 0, "total_confidence": 0.0})
        for obj in final_unique_objects.values():
            class_name = obj["class"]
            confidence = obj["confidence"]
            for category, sub_classes in self.CATEGORY_MAPPING.items():
                if class_name in sub_classes:
                    category_stats[category]["count"] += 1
                    category_stats[category]["total_confidence"] += confidence

        detailed_summary = []
        for category, data in category_stats.items():
            if data["count"] > 0:
                detailed_summary.append({"type": category.capitalize(), "count": data["count"], "avg_confidence": f"{(data['total_confidence'] / data['count']) * 100:.1f}%"})

        duration_seconds = video_info.total_frames / video_info.fps
        metadata = {
            "duration": f"{int(duration_seconds // 60)}:{int(duration_seconds % 60):02d}",
            "resolution": f"{video_info.resolution_wh}",
            "fps": video_info.fps
        }
        
        cv2.destroyAllWindows()
        
        counts = {cat: data["count"] for cat, data in category_stats.items()}
        record = DetectionRecord(counts=counts, filename=original_filename, duration=metadata["duration"], resolution=metadata["resolution"], fps=metadata["fps"], processing_time=round(time.time() - start_time, 2))
        await record.insert()

        return {"metadata": metadata, "summary": detailed_summary, "counts": counts}

    async def process_employee_sittings(self, source_path: str, target_path: str, original_filename: str):
        """
        Specialized processing for employee sitting/standing detection.
        Uses Pose Estimation and Tracking.
        """
        video_info = sv.VideoInfo.from_video_path(source_path)
        generator = sv.get_video_frames_generator(source_path)
        
        # Fix 1: BoT-SORT managed internally by Ultralytics .track(persist=True)
        # Reset predictor state so old video tracks don't bleed into this one
        self.pose_model.predictor = None

        # Fix 3: Sliding window posture vote — majority over the last 45 frames (~1.5s at 30fps)
        # Prevents a single bad keypoint frame from flipping the posture label
        POSTURE_WINDOW = 45
        # Fix 4: Commitment threshold — how many consecutive frames of the new posture
        # are required before the label actually switches (suppresses transition flicker)
        COMMITMENT_THRESHOLD = 20
        tracker_id_to_postures = defaultdict(lambda: deque(maxlen=POSTURE_WINDOW))
        tracker_committed_posture = {}       # tid -> currently locked posture string
        tracker_consecutive_count = defaultdict(int)  # tid -> current streak length
        
        # Colors & Constants
        STANDING_COLOR = (0, 230, 80)
        SITTING_COLOR  = (255, 140, 0)
        SKELETON_COLOR = (0, 200, 255)
        
        # Skeleton connections (COCO)
        SKELETON = [(5, 6), (5, 11), (6, 12), (11, 12), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]

        start_time = time.time()
        print(f"Employee Sitting Detection started: {source_path}")

        # Use 11m-pose model for this task
        with sv.VideoSink(target_path=target_path, video_info=video_info) as sink:
            for index, frame in enumerate(generator):
                # Fix 1+2+5: BoT-SORT via native .track() with tight NMS; yolo11x-pose for cleaner keypoints
                results = self.pose_model.track(frame, imgsz=1024, iou=0.45, device=0, half=True,
                                                persist=True, tracker="botsort.yaml", verbose=False)[0]

                # Convert to supervision detections (tracker_id is already embedded from BoT-SORT)
                detections = sv.Detections.from_ultralytics(results)

                # Skip frame if BoT-SORT hasn't assigned IDs yet (first 1-2 frames)
                if detections.tracker_id is None:
                    sink.write_frame(frame=frame)
                    continue

                # Filter to confident detections only (post-tracking)
                detections = detections[detections.confidence > 0.3]

                annotated_frame = frame.copy()
                
                # We need to map tracker IDs to original pose result indices
                # ByteTrack might reorder detections, so we find matching boxes
                current_standing = 0
                current_sitting = 0

                # supervision.Detections stores xyxy in .xyxy
                # results.boxes stores xyxy in results.boxes.xyxy
                
                for i in range(len(detections)):
                    box = detections.xyxy[i]
                    tid = detections.tracker_id[i]
                    conf = detections.confidence[i]
                    
                    # Find matching individual in pose results to get keypoints
                    # Simple check: overlap or proximity since we don't have direct mapping easily
                    # Often, we can just iterate over the results if we match boxes
                    
                    # Let's find the pose index that matches this tracked box
                    pose_idx = -1
                    best_iou = 0
                    for p_idx, p_box in enumerate(results.boxes.xyxy):
                        iou = sv.box_iou_batch(box.reshape(1, 4), p_box.cpu().numpy().reshape(1, 4))[0][0]
                        if iou > best_iou:
                            best_iou = iou
                            pose_idx = p_idx
                    
                    if pose_idx != -1 and best_iou > 0.5:
                        kp = results.keypoints.data[pose_idx].cpu().numpy()
                        raw_posture = self.classify_posture(kp, box)
                        tracker_id_to_postures[tid].append(raw_posture)

                        # Fix 4: Committed posture debounce — only switch label after
                        # COMMITMENT_THRESHOLD consecutive frames of the new posture
                        committed = tracker_committed_posture.get(tid, raw_posture)
                        if raw_posture == committed:
                            tracker_consecutive_count[tid] = min(
                                tracker_consecutive_count[tid] + 1, COMMITMENT_THRESHOLD * 2
                            )
                        else:
                            tracker_consecutive_count[tid] -= 1
                            if tracker_consecutive_count[tid] <= 0:
                                tracker_committed_posture[tid] = raw_posture
                                tracker_consecutive_count[tid] = COMMITMENT_THRESHOLD // 2
                        # Use committed posture for display (stable label)
                        posture = tracker_committed_posture.get(tid, raw_posture)
                        tracker_committed_posture.setdefault(tid, raw_posture)

                        color = STANDING_COLOR if posture == "standing" else SITTING_COLOR

                        # Draw Skeleton
                        for a, b in SKELETON:
                            if kp[a][2] > 0.3 and kp[b][2] > 0.3:
                                pt1 = (int(kp[a][0]), int(kp[a][1]))
                                pt2 = (int(kp[b][0]), int(kp[b][1]))
                                cv2.line(annotated_frame, pt1, pt2, SKELETON_COLOR, 1, cv2.LINE_AA)
                        
                        # Draw Joints
                        for kidx in range(5, 17): # Focus on body joints
                            if kp[kidx][2] > 0.3:
                                cv2.circle(annotated_frame, (int(kp[kidx][0]), int(kp[kidx][1])), 2, (255, 255, 255), -1, cv2.LINE_AA)

                        # Draw Box Corners
                        x1, y1, x2, y2 = map(int, box)
                        self.box_annotator.annotate(scene=annotated_frame, detections=detections[i:i+1])
                        
                        # Draw Label
                        label = f"ID:{tid} {posture.upper()}"
                        self._draw_pill_label(annotated_frame, label, x1, y1, color)
                        
                        if posture == "standing": current_standing += 1
                        else: current_sitting += 1

                # HUD Overlay
                # Filter out "ghost" tracks (seen for fewer than 15 frames, i.e., 0.5s at 30fps)
                valid_tracker_ids = {tid: postures for tid, postures in tracker_id_to_postures.items() if len(postures) >= 15}
                
                unique_people = len(valid_tracker_ids)
                global_standing = 0
                global_sitting = 0
                for tid, postures in valid_tracker_ids.items():
                    # Fix 3+4: Prefer committed posture; fall back to sliding window majority vote
                    final_posture = tracker_committed_posture.get(
                        tid, Counter(postures).most_common(1)[0][0]
                    )
                    if final_posture == "standing": global_standing += 1
                    else: global_sitting += 1

                self._draw_hud(annotated_frame, unique_people, global_standing, global_sitting)
                
                # Show preview
                cv2.imshow("Employee Sitting Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
                sink.write_frame(frame=annotated_frame)

        cv2.destroyAllWindows()
        
        # Final Summary
        final_sitting = 0
        final_standing = 0
        valid_tracker_ids = {tid: postures for tid, postures in tracker_id_to_postures.items() if len(postures) >= 15}

        for tid, postures in valid_tracker_ids.items():
            # Fix 3+4: Use committed posture for definitive end-of-video classification
            final_posture = tracker_committed_posture.get(
                tid, Counter(postures).most_common(1)[0][0]
            )
            if final_posture == "standing":
                final_standing += 1
            else:
                final_sitting += 1

        total_person = final_sitting + final_standing
        
        metadata = {
            "duration": f"{int((video_info.total_frames / video_info.fps) // 60)}:{int((video_info.total_frames / video_info.fps) % 60):02d}",
            "resolution": f"{video_info.resolution_wh}",
            "fps": video_info.fps
        }
        
        counts = {
            "person": total_person,
            "sitting": final_sitting,
            "standing": final_standing
        }
        
        # Save to DB
        processing_duration = time.time() - start_time
        record = DetectionRecord(
            counts=counts,
            filename=original_filename,
            duration=metadata["duration"],
            resolution=metadata["resolution"],
            fps=metadata["fps"],
            processing_time=round(processing_duration, 2)
        )
        await record.insert()

        return {
            "metadata": metadata,
            "counts": counts,
            "summary": [
                {"type": "Total Persons", "count": total_person},
                {"type": "Sitting", "count": final_sitting},
                {"type": "Standing", "count": final_standing}
            ]
        }
