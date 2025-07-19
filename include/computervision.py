import math
import numpy as np


def readBotsAlive(frame, reader, botsAliveHudCoords):
    # --- Crop out the hud element with the number of bots alive ---
    botsAliveHudElement = frame.copy()[botsAliveHudCoords[1] : botsAliveHudCoords[3], botsAliveHudCoords[0] : botsAliveHudCoords[2]]
    
    # --- Use OCR to get the number in the image ---
    nAlive = reader.readtext(botsAliveHudElement, allowlist='0123456789')
    nAlive = int(nAlive[0][1])

    return nAlive


def detectTargets(frame: np.typing.NDArray, frame2: np.typing.NDArray, detectionModel) -> np.typing.NDArray:
    # Run object detection on the received frame to detect bot positions, measured in pixels
    results       = detectionModel.predict([frame, frame2], classes=[0], save=False, verbose=False, device="cuda", imgsz=864, conf=0.4)

    boundingBoxesFrame1         = results[0].boxes.xywh.cpu().numpy()
    boundingBoxesFrame1[..., 0] = boundingBoxesFrame1[..., 0] - (boundingBoxesFrame1[..., 2] / 2)
    boundingBoxesFrame1[..., 1] = boundingBoxesFrame1[..., 1] - (boundingBoxesFrame1[..., 3] / 2)
    
    boundingBoxesFrame2         = results[1].boxes.xywh.cpu().numpy()
    boundingBoxesFrame2[..., 0] = boundingBoxesFrame2[..., 0] - (boundingBoxesFrame2[..., 2] / 2)
    boundingBoxesFrame2[..., 1] = boundingBoxesFrame2[..., 1] - (boundingBoxesFrame2[..., 3] / 2)

    return boundingBoxesFrame1, boundingBoxesFrame2


def getHeadPositions(boundingBoxes, gameWindowWidth, gameWindowHeight):
    # The positions with bots are normalized in the -1, 1 range. -1 means the bot is on the left edge of the
    # the screen, and 1 means the bot is on the right edge of the screen.
    headPositions = np.empty((boundingBoxes.shape[0], 3), dtype=np.float32)

    # Compute head positions
    headPositions[..., 0] = (boundingBoxes[..., 0] + boundingBoxes[..., 2] * 0.50) / gameWindowWidth # x + w * 0.50 moves the crosshair to the middle of the bounding box, adjusted just a bit to the right
    headPositions[..., 1] = (boundingBoxes[..., 1] + boundingBoxes[..., 3] * 0.12) / gameWindowHeight # y + h * 0.12 because this puts the aim right on top of the bot's head. Headshots = good.

    # Map to [-1, 1]
    headPositions[..., :2] = headPositions[..., :2] * 2.0 - 1.0

    # Add isValid flag
    headPositions[..., 2] = 1.0

    return headPositions


def findClosestTarget(headPositions):
    # Calculate the eucledian distance from the center of the screen (0,0) to each of the head positions
    dists = [ np.hypot(x, y) for x, y, _ in headPositions ]

    # Return the idx of the closest target
    return int(np.argmin(dists))


def selectTarget(frame, frame2, detectionModel, gameWindowWidth, gameWindowHeight, timeElapsed):
    boundingBoxesFrame1, boundingBoxesFrame2 = detectTargets(frame, frame2, detectionModel)
    
    # If there's at least one detection in each frame
    if  boundingBoxesFrame1.shape[0] > 0 \
    and boundingBoxesFrame2.shape[0] > 0:
    
        headPositionsFrame1 = getHeadPositions(boundingBoxesFrame1, gameWindowWidth, gameWindowHeight)
        closestHeadFrame1   = headPositionsFrame1[findClosestTarget(headPositionsFrame1)]
        
        headPositionsFrame2 = getHeadPositions(boundingBoxesFrame2, gameWindowWidth, gameWindowHeight) 
        idx                 = findClosestTarget(headPositionsFrame2 - closestHeadFrame1)
        dx, dy, _ = (headPositionsFrame2 - closestHeadFrame1)[idx]

        vx = dx / timeElapsed
        vy = dy / timeElapsed

        predictedHeadPositionX = headPositionsFrame2[idx][0] + vx * (0.045)
        predictedHeadPositionY = headPositionsFrame2[idx][1] + vy * (0.045)

        return np.asarray([predictedHeadPositionX, predictedHeadPositionY, 1.0], dtype=np.float32)
    
    else:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float32)


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
    """Raw mouse counts -> (dx, dy) in pixels (centre-origin).
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
