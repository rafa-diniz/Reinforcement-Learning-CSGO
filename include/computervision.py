import math
import numpy as np

def detectKills(frame, previousAlive, gameWindowWidth, gameWindowHeight):
    # In Counter Strike, each frame contains information about the status of each of the 12 bots alive.
    # Each bot has a "portrait" on the HUD. If this portrait is lit-up, it means the bot is alive, otherwise it has
    # been killed. Below is an AI-free way of detecting the number of kills.

    # This is more specific to Counter Strike, but the bot portraits are divided in two rows, each row with six portraits.
    # Instead of storing the location of each portrait, I added the location of the two portraits in the first column, and
    # then I take advantage of the fact that all portraits are spaced out by the same amount (portraitOffset). I simply
    # perform a scanline starting from the two portraits in the first row and add an offset to dinamically get the location
    # of the other portraits.
    scanlineStartPoint = [
                            [0.384815, 0.002780],
                            [0.384815, 0.031510]
                        ]

    portraitOffset = 0.0156

    # Generating the location of the other portraits. This is ugly, and runs everytime this function is called, but it's also
    # a for loop with 12 iterations. This really isn't the bottleneck of the program. Maybe optimize later.
    portraitLocation = []
    for i in range(2):
        for j in range(6):
            portraitLocation.append([
                            np.round(scanlineStartPoint[i][1] * gameWindowHeight),
                            np.round((scanlineStartPoint[i][0] + portraitOffset * j) * gameWindowWidth)
                        ])
    
    
    portraitLocation = np.asarray(portraitLocation, dtype=np.int32)
    # If the portrait is lit-up, the pixels in it should have the color [181, 212, 238]. Here we get currentAlive, an
    # array with True where the portrait is lit-up and False where it isn't
    currentAlive     = np.all(frame[portraitLocation[..., 0], portraitLocation[..., 1]] == [181, 212, 238], axis=1)
    
    # A bitwise xor of currentAlive and previousAlive tells us where the 'alive' status has changed. This is important because
    # sometimes a screenshot can happen so quick that some bots might not have had the chance to respawn. If a bot was killed in the
    # previous action and still hasn't respawned, its status will count as "killed" even if it wasn't the current action that killed it.
    # This could seriously mess up the reward function, so I coded this black magic that only counts as kill bots that:
    # a) were alive in the previous verification (previousAlive)
    # b) are killed now.
    confirmedKills = np.count_nonzero(np.bitwise_xor(previousAlive, currentAlive)[previousAlive])

    return confirmedKills


def detectTargets(frame: np.typing.NDArray, detectionModel: "Ultralytics model") -> np.typing.NDArray: # type: ignore
    """
    Receives an image and an object detector and outputs the bounding boxes for the detections in the frame.
    I don't like the default way that the boxes are exported and prefer working with them in this format:
    a) X and Y point to the top left corner of the bounding box
    b) W and H are the width and the height

    So I convert the bounding boxes to be in this format.

    Args:
        frame (np.typing.NDArray): The image as a numpy array
        detectionModel (Ultralytics model): The ultralytics object detector. Should be an instance of Ultralytics.YOLO

    Returns:
        np.typing.NDArray: Bounding boxes in the xywh format
    """
    
    results = detectionModel.predict(frame, classes=[0], save=False, verbose=False, device="cuda", imgsz=864, conf=0.4)
    
    # The X and Y values in boxes.xywh are centered. I'd rather have them as X and Y pointing
    # to the top-left corner and W,H as the pure width and height. This here converts them
    # to the format I'm more comfortable with
    boxes = results[0].boxes.xywh
    boxes[..., 0] = boxes[..., 0] - (boxes[..., 2] / 2)
    boxes[..., 1] = boxes[..., 1] - (boxes[..., 3] / 2)

    boxes = boxes.cpu().numpy()

    return boxes



def getHeadPositions(boxes: np.typing.NDArray, gameWindowWidth: int, gameWindowHeight: int) -> np.typing.NDArray:
    """Receives bounding boxes and outputs the estimated position of the head within that bounding box.
    The head positions are in the format that the network expects, that is:
    a) normalized in the -1, 1 range
    b) with an isValid flag appended to each detection.

    Args:
        boxes (np.typing.NDArray): The bounding boxes
        gameWindowWidth (int): Width of the game window
        gameWindowHeight (int): Height of the game window

    Returns:
        np.typing.NDArray: The positions of the heads
    """
    x = boxes[..., 0]
    y = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]
    
    x = ( x + (w * 0.50) ) / gameWindowWidth  # x + w * 0.50 moves the crosshair to the middle of the bounding box.
    y = ( y + (h * 0.12) ) / gameWindowHeight # y + h * 0.12 because this lowers the aim right onto the bot's head, increasing the chance of a headshot.
    
    # The head positions are normalized in the -1, 1 range. -1 is the leftmost pixel of the
    # the screen, and 1 is the rightmost pixel of the screen. In the Y axis, -1 = Topmost pixel, 1 = Bottommost pixel. 
    # This -1, 1 normalization makes it easier for the neural network to learn.
    x = (x * 2) - 1
    y = (y * 2) - 1

    # Generating the isValid flags. It's always valid (valid=1) because 'boxes' comes from a YOLO model; that is, it's always
    # outlining bounding boxes of true detections. 
    isValid       = np.ones_like(x)
    headPositions = np.stack([x, y, isValid], axis=1)

    return headPositions



def selectTarget(frame, detectionModel, gameWindowWidth, gameWindowHeight):
    boxes         = detectTargets(frame, detectionModel)
    headPositions = getHeadPositions(boxes, gameWindowWidth, gameWindowHeight)

    # Calculate the eucledian distance from the center of the screen (0,0) to each of the positions
    dists = [ np.hypot(x, y) for x, y, _ in headPositions ]
    
    # Get the bot with the shortest distance and return it along with the corresponding 
    # bounding box for that detection.
    if dists:
        idx = int(np.argmin(dists))
        return headPositions[idx], boxes[idx]
    
    else:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float32), None
    

def pixelsToCounts(dxNormalized, dyNormalized, gameWindowWidth, gameWindowHeight):
    dyNormalized = dyNormalized * -1
    """Screen (px,py) ➜ (dx,dy) in raw mouse counts."""
    # 1)  What the console says:
    BASE_H4_3 = 90
    VFOV_DEG = math.degrees(
            2 * math.atan( (3/4) * math.tan(math.radians(BASE_H4_3) / 2))
        )
    ASPECT   = gameWindowWidth / gameWindowHeight         

    # 2)  Derive the missing side:
    HFOV_DEG = math.degrees(
            2 * math.atan( ASPECT * math.tan(math.radians(VFOV_DEG) / 2))
        )
    
    SENS     = 1.0
    M_YAW    = 0.022
    M_PITCH  = 0.022
    
    fx = (gameWindowWidth  / 2) / math.tan(math.radians(HFOV_DEG) / 2)
    fy = (gameWindowHeight / 2) / math.tan(math.radians(VFOV_DEG) / 2)
    
    # dx, dy are in the [-1, 1] interval.
    dx = (dxNormalized + 1) * 0.5 * gameWindowWidth
    dy = (dyNormalized + 1) * 0.5 * gameWindowHeight
    dx = dx - gameWindowWidth  / 2
    dy = dy - gameWindowHeight / 2   # down is +y on screen

    # pixel → degrees
    yaw_deg   =  math.degrees(math.atan(dx / fx))
    pitch_deg = -math.degrees(math.atan(dy / fy))

    # degrees → raw counts   (round to int for SendInput / mouse_event)
    cnt_x = int(round(yaw_deg   / (SENS * M_YAW)))
    cnt_y = int(round(pitch_deg / (SENS * M_PITCH)))
    
    return cnt_x, cnt_y


def countsToPixels(cnt_x, cnt_y,
                   gameWindowWidth, gameWindowHeight,
                   sens=1.0, m_yaw=0.022, m_pitch=0.022,
                   return_normalised=False):
    """Raw mouse counts → (dx, dy) in pixels (centre-origin).
       If return_normalised=True it also returns (dxNorm, dyNorm)."""

    BASE_H4_3 = 90
    vfov_deg = math.degrees(
        2 * math.atan((3/4) * math.tan(math.radians(BASE_H4_3) / 2))
    )
    aspect = gameWindowWidth / gameWindowHeight
    hfov_deg = math.degrees(
        2 * math.atan(aspect * math.tan(math.radians(vfov_deg) / 2))
    )

    # pixel focal lengths
    fx = (gameWindowWidth  / 2) / math.tan(math.radians(hfov_deg) / 2)
    fy = (gameWindowHeight / 2) / math.tan(math.radians(vfov_deg) / 2)

    # counts -> degrees
    yaw_deg   = cnt_x * sens * m_yaw
    pitch_deg = cnt_y * sens * m_pitch

    # degrees ➜ pixels
    dx =  math.tan(math.radians(yaw_deg))   * fx
    dy = -math.tan(math.radians(pitch_deg)) * fy

    if not return_normalised:
        return dx, dy

    # centre-origin pixels ➜ normalised [–1,1]
    dx_norm =  dx / (gameWindowWidth  / 2)
    dy_norm = -dy / (gameWindowHeight / 2)   # undo earlier flip

    return dx_norm, dy_norm
