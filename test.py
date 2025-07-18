from ultralytics import YOLO
from PIL import Image

img1 = Image.open("pngs/4_1.png")
img2 = Image.open("pngs/4_2.png")

# Load an official or custom model
model = YOLO("yolo11m.pt")  # Load an official Detect model

# Perform tracking with the model
results = model.track([img1, img2], classes=[0], tracker="custom.yaml", imgsz=864, conf=0.4)  # Tracking with custom tracker
for r in results:
    r.show()