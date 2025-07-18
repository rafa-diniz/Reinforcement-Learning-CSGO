import time
import win32api
import win32con
import win32gui

import numpy as np


def getCSGOWindowDimensions():
    handle = win32gui.FindWindow(None, "Counter-Strike: Global Offensive - Direct3D 9")

    if handle:
        rect = win32gui.GetWindowRect(handle)
        
        # These tell the pixel position of the start of the window
        # window on the screen. Useful to automatically detect the position of the CSGO window
        # in case it is in the middle of the screen or wherever else
        gameWindowX0 = rect[0]
        gameWindowY0 = rect[1]

        # width, height are the dimensions of the CSGO window
        gameWindowWidth  = rect[2] - rect[0]
        gameWindowHeight = rect[3] - rect[1]

        gameWindowX0     = gameWindowX0     + 3 # +3 because Windows uses some transparent padding on the border of the window
        gameWindowWidth  = gameWindowWidth  - 3 # -3 once again for the right padding

        gameWindowY0     = gameWindowY0     + 26  # +26 because Windows puts a white bar above the window when in windowed mode
        gameWindowHeight = gameWindowHeight - 30  # -30 to crop out the bottom padding


        return gameWindowX0, gameWindowY0, gameWindowWidth, gameWindowHeight
    else:
        raise Exception("CSGO Window not found!")
    

def moveMouse(dxMouseUnits, dyMouseUnits):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 
                        dxMouseUnits,
                        dyMouseUnits,
                        0,
                        0
                        )
    

def leftClick():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)


def multistepSchedule(progress_remaining: float) -> float:
    elapsed = 1.0 - progress_remaining
    if elapsed <= 0.25:
        return 3e-4
    elif elapsed <= 0.5:
        return 2e-4
    elif elapsed <= 0.75:
        return 1e-4
    else:
        return 7e-5



def xywh_to_xyxy(box):
    x, y, w, h = box

    x1, y1 = x, y
    x2, y2 = x1 + w, y1 + h
    return np.array([x1, y1, x2, y2], dtype=float)


def iou_xywh(box_a, box_b):
    x1a, y1a, x2a, y2a = xywh_to_xyxy(box_a)
    x1b, y1b, x2b, y2b = xywh_to_xyxy(box_b)

    # intersection
    xi1, yi1 = max(x1a, x1b), max(y1a, y1b)
    xi2, yi2 = min(x2a, x2b), min(y2a, y2b)
    inter_w, inter_h = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    # areas
    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)

    # IoU
    union = area_a + area_b - inter_area + 1e-6  # tiny eps to avoid /0
    return inter_area / union