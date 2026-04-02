import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # fixes OpenMP crash on Windows

import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, no GUI conflict

from ultralytics import YOLO

if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0))
    
    model = YOLO("yolo11m.pt")
    
    results = model.train(
        data="Employee-Activity-Detection-7/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        patience=20,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        degrees=10.0,
        project="employee_activity",
        name="yolo11m_run1",
        device=0,
        workers=0,
        exist_ok=True,
        pretrained=True,
        save=True,
        plots=True,
    )
    
    print("Training complete!")
    print(f"Best mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")