from PIL import Image
import numpy as np

img = Image.open("detect_test.png")
img = np.asarray(img)

# X e Y dos pixels que vão iniciar o run-length
scanlineStartPoint = [
                        [0.384815, 0.002780],
                        [0.384815, 0.031510]
                    ]

gameResolutionX = 1923
gameResolutionY = 1079
pixeloffset     = 0.0156

test = []
for i in range(2):
    for j in range(6):
        test.append([
                        np.round(scanlineStartPoint[i][1] * gameResolutionY),
                        np.round((scanlineStartPoint[i][0] + pixeloffset * j) * gameResolutionX)
                    ])
        


test = np.asarray(test, dtype=np.int32)
pixels = img[test[..., 0], test[..., 1]]
print(np.all(pixels == [181, 212, 238], axis=1))