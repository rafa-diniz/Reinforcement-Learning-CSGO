from PIL import Image
import numpy as np

img = Image.open("detect_test.png")
img = np.asarray(img)

print(img.shape)

# Y dos pixels que vão iniciar o run-length
pixels_of_interest = [
                [0.0, 0.2769],
                [0.0, 0.7692]
            ]

# Para fazer o crop
hudcoords = [
                [0.3744, 0.0],
                [0.4758, 0.0611]
            ]

hudcoords = (np.asarray(hudcoords) * np.array([1920, 1080])).astype(np.int32)
img       = img[hudcoords[0][1] : hudcoords[1][1], hudcoords[0][0] : hudcoords[1][0]]

print(img.shape)