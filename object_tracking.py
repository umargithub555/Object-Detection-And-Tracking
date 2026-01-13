# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



# import numpy as np
# import supervision as sv

# from ultralytics import YOLO
# from supervision.assets import download_assets, VideoAssets
# import cv2

# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)



# SOURCE_VIDEO_PATH = "vehicles.mp4"

# generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
# frame = next(generator)


# print(sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH))




# model = YOLO("yolo11x.pt").to('cuda')

# # results = model(frame,device=0, verbose=False)[0]

# results = model(
#     frame,
#     device=0,
#     half=True,     # 🔥 FP16
#     verbose=False
# )[0]


# detections = sv.Detections.from_ultralytics(results)


# # bounding_box_annotator = sv.BoxAnnotator(thickness=6)



# labels = [
#     f"{results.names[class_id]} {confidence:0.2f}"
#     for class_id, confidence
#     in zip(detections.class_id, detections.confidence)
# ]

# # box_annotator = sv.BoxAnnotator(thickness=6)
# # label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)

# # annotated_frame = frame.copy()
# # annotated_frame = box_annotator.annotate(annotated_frame, detections)
# # annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)


# byte_tracker = sv.ByteTrack()




# START = sv.Point(0, 1500)
# END = sv.Point(3840, 1500)

# line_zone = sv.LineZone(start=START, end=END)

# line_zone_annotator = sv.LineZoneAnnotator(
#     thickness=4,
#     text_thickness=4,
#     text_scale=2)

# # annotated_frame = frame.copy()
# # annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)


# bounding_box_annotator = sv.BoxAnnotator(thickness=4)
# label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)
# trace_annotator = sv.TraceAnnotator(thickness=4)




# # def callback(frame: np.ndarray, index:int) -> np.ndarray:
# #     results = model(frame, verbose=False)[0]
# #     detections = sv.Detections.from_ultralytics(results)
# #     detections = byte_tracker.update_with_detections(detections)

# #     labels = [
# #         f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
# #         for confidence, class_id, tracker_id
# #         in zip(detections.confidence, detections.class_id, detections.tracker_id)
# #     ]

# #     annotated_frame = frame.copy()
# #     annotated_frame = trace_annotator.annotate(
# #         scene=annotated_frame,
# #         detections=detections)
# #     annotated_frame = bounding_box_annotator.annotate(
# #         scene=annotated_frame,
# #         detections=detections)
# #     annotated_frame = label_annotator.annotate(
# #         scene=annotated_frame,
# #         detections=detections,
# #         labels=labels)

# #     line_zone.trigger(detections)

# #     return  line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)


# def callback(frame: np.ndarray, index: int) -> np.ndarray:
#     results = model(
#         frame,
#         device=0,
#         half=True,
#         verbose=False
#     )[0]

#     detections = sv.Detections.from_ultralytics(results)
#     detections = byte_tracker.update_with_detections(detections)

#     labels = [
#         f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
#         for confidence, class_id, tracker_id
#         in zip(
#             detections.confidence,
#             detections.class_id,
#             detections.tracker_id
#         )
#     ]

#     annotated = frame.copy()
#     annotated = trace_annotator.annotate(annotated, detections)
#     annotated = bounding_box_annotator.annotate(annotated, detections)
#     annotated = label_annotator.annotate(annotated, detections, labels)

#     line_zone.trigger(detections)
#     return line_zone_annotator.annotate(annotated, line_zone)



# TARGET_VIDEO_PATH = f"vehicles_processed.mp4"




# sv.process_video(
#     source_path = SOURCE_VIDEO_PATH,
#     target_path = TARGET_VIDEO_PATH,
#     callback=callback
# )




import os
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# --- Configuration & Fixes ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

SOURCE_VIDEO_PATH = "vehicles.mp4"
TARGET_VIDEO_PATH = "vehicles_production.mp4"

# --- 1. Initialize Model & Tracker ---
model = YOLO("yolo11x.pt").to('cuda')
byte_tracker = sv.ByteTrack()

# --- 2. Define Production-Style Annotators ---
# Professional "Cyberpunk" Color Palette
COLORS = sv.ColorPalette.from_hex(["#00FFCC", "#FF3366", "#33FF57", "#7B61FF"])

corner_annotator = sv.BoxCornerAnnotator(
    color=COLORS,
    thickness=2,
    corner_length=30
)

# FIXED: Removed 'position' argument to avoid TypeError
label_annotator = sv.LabelAnnotator(
    color=COLORS,
    text_thickness=1,
    text_scale=0.5,
    text_padding=10,
    border_radius=4
)

trace_annotator = sv.TraceAnnotator(
    color=COLORS,
    thickness=2,
    trace_length=20,
    position=sv.Position.CENTER
)

# Coordinates for 4K video line counting
LINE_START = sv.Point(0, 1500)
LINE_END = sv.Point(3840, 1500)
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

line_zone_annotator = sv.LineZoneAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1.5,
    custom_in_text="ENTERING",
    custom_out_text="EXITING"
)

def process_video():
    # Get video metadata
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    print(f"Processing started: {SOURCE_VIDEO_PATH} ({video_info.resolution_wh})")
    
    with sv.VideoSink(target_path=TARGET_VIDEO_PATH, video_info=video_info) as sink:
        for index, frame in enumerate(generator):
            # 3. Inference (Using FP16 for speed)
            results = model(frame, device=0, half=True, verbose=False)[0]
            
            # 4. Update Detections & Tracker
            detections = sv.Detections.from_ultralytics(results)
            detections = byte_tracker.update_with_detections(detections)

            # 5. Build Labels
            labels = [
                f"ID:{tracker_id} {model.model.names[class_id].upper()}"
                for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
            ]

            # 6. Annotate Frame
            annotated_frame = frame.copy()
            
            # Layers: Trace -> Corners -> Labels
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = corner_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            # Line Zone Trigger and Annotation
            line_zone.trigger(detections)
            annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

            # 7. FIXED: Real-time Preview using OpenCV Resize
            # We calculate the 40% scale manually for better compatibility
            scale = 0.4
            width = int(annotated_frame.shape[1] * scale)
            height = int(annotated_frame.shape[0] * scale)
            preview_frame = cv2.resize(annotated_frame, (width, height))
            
            cv2.imshow("Production AI Analytics", preview_frame)

            # 8. Save to File
            sink.write_frame(frame=annotated_frame)

            # Press 'q' to stop processing
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print(f"Processing Complete. File saved: {TARGET_VIDEO_PATH}")

if __name__ == "__main__":
    process_video()