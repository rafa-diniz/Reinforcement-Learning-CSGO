import time
import win32api
import win32con
import win32gui


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
    time.sleep(0.1)
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
