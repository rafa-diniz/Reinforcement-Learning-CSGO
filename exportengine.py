from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.export(format="engine",
             imgsz=864,
             batch=2,
             half=True,
             device=0,
             dynamic=True
             )
