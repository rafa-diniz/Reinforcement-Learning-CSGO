from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.export(format="engine",
             imgsz=864,
             half=True, # FP16
             device=0,
             dynamic=True
             )
