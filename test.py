import time
import numpy as np

from ultralytics import YOLO
from PIL import Image

import include.computervision as computervision

detectionModel   = YOLO("yolo11m.engine", task="detect")

img1 = Image.open("pngs/2_1.png")
img2 = Image.open("pngs/2_2.png")

img1 = np.asarray(img1)
img2 = np.asarray(img2)

img_warmup = np.full_like(img1, np.random.randint(0, 255), np.uint8)

_ = detectionModel.predict(img1, classes=[0], save=False, verbose=False, device="cuda", imgsz=864, conf=0.4)

detectionDx, detectionDy = computervision.selectTarget(img1, img2, detectionModel, 1920, 1080)