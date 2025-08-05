import cv2
import time
import dxcam
import easyocr
import numpy as np
import gymnasium as gym

from include import computervision
from include import utils


class CSGOAimEnv(gym.Env):
    def __init__(self, detectionModel):
        super().__init__()
        self.maxMove = 1 # The range of motion allowed. 

        self.nBotsInObservation = 1  # How many bots should be in the observation (more bots = more inputs in the MLP)
        self.nBotsInServer      = 12 # How many bots can be detected in-game at a given time
        
        # [dx, dy, shootProbability]
        self.action_space = gym.spaces.Box(
            low = np.array([-self.maxMove, -self.maxMove,  0.0], dtype=np.float32),
            high= np.array([ self.maxMove,  self.maxMove,  1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # X, Y and isValid flag for each bot + currentY position
        self.obsDim = self.nBotsInObservation * 3 + 1

        self.observation_space = gym.spaces.Box(
        low   =  np.array([-1.0, -1.0,  0.0, -1.0], dtype=np.float32),
        high  =  np.array([ 1.0,  1.0,  1.0,  1.0], dtype=np.float32),
        shape = (self.obsDim,),
        dtype = np.float32
        )

        self.detectionModel = detectionModel
        

        self.gameWindowX0,    self.gameWindowY0, \
        self.gameWindowWidth, self.gameWindowHeight = utils.getCSGOWindowDimensions()

        region = (self.gameWindowX0, self.gameWindowY0, self.gameWindowX0 + self.gameWindowWidth, self.gameWindowY0 + self.gameWindowHeight)
        self.cam = dxcam.create()
        self.cam.start(region=region, target_fps=120)

        self.tracker = cv2.TrackerVit.create()

        self.maxActionDx, self.maxActionDy = computervision.pixelsToCounts(1, 1, self.gameWindowWidth, self.gameWindowHeight)

        # These weird values indicate where we can find the number of bots alive in the HUD, relative to the screen.
        # It varies from resolution to resolution, so doing it this way makes it work
        # automatically for 480p, 720p, 900p, 1080p, etc.
        self.botsAliveHudCoords = [
            int(0.4500078 * self.gameWindowWidth), int(0.0027803 * self.gameWindowHeight),
            int(0.4758190 * self.gameWindowWidth), int(0.0417006 * self.gameWindowHeight)
        ]


        # The currentY position of the agent
        self.currentY           = 0.0
        self.previousAlive      = None
        self.numberKilled       = None
        self.detectionIsValid   = None
        self.detectionDx        = None
        self.detectionDy        = None
        self.numSteps           = 0
        self.invalidDetecs      = 0
        self.shoot              = True



    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        time.sleep(0.5)

        # Reset the number of steps taken
        self.numSteps = 0

        # Reset the mouse Y position to 0
        utils.moveMouse(0, int(-self.currentY * 4042))
        self.currentY = 0.0

        # Take a screenshot to read the rest of the needed information
        gameFrame = self.cam.get_latest_frame()

        # Run object detection on the screenshot and grab the dx and dy positions of the target
        (self.detectionDx, self.detectionDy, self.detectionIsValid), bb0 = computervision.selectTarget(gameFrame, 
                                                                                                self.detectionModel, 
                                                                                                self.gameWindowWidth,
                                                                                                self.gameWindowHeight                                  
                                                                                                )
        
        if bb0 is not None: # selectTarget() didn't find anything  
            self.tracker.init(gameFrame, bb0.astype(np.int32))

        # Run OCR to get the number of bots alive
        self.previousAlive = np.asarray([True] * 12)
        self.numberKilled  = 0

        obs = self._makeObs(self.detectionDx, self.detectionDy, self.detectionIsValid, self.currentY)

        return obs, {}


    def step(self, action):
        self.numSteps += 1

        if self.invalidDetecs > 10:
            raise Exception
        
        # Parse the action chosen by the network
        actionDx, actionDy, shootProbability = action

        dxMouseUnits, dyMouseUnits = np.ceil(actionDx * self.maxActionDx), np.ceil(actionDy * self.maxActionDy)
        dxMouseUnits, dyMouseUnits = int(dxMouseUnits), int(dyMouseUnits)

        utils.moveMouse(dxMouseUnits, dyMouseUnits)

        time.sleep(0.01)
        if np.random.rand() < shootProbability:
            utils.leftClick()

        gameFrame     = self.cam.get_latest_frame()
        confirmedKills, currentAlive  = computervision.detectKills(gameFrame, self.previousAlive, self.gameWindowWidth, self.gameWindowHeight)

        reward = 0.0
        if self.detectionIsValid:
            self.invalidDetecs = 0
            ok, bb1  = self.tracker.update(gameFrame)
            
            #  Add a big reward for getting a kill
            self.numberKilled += confirmedKills
            reward  += confirmedKills * 2.0
            # Distance Reward: decreases the penalty the closer the agent is to the target
            if ok:
                # Get the distance to the target in mouse units
                target   = computervision.getHeadPositions(np.asarray([bb1]), self.gameWindowWidth, self.gameWindowHeight)
                target   = target[0][ 0 : 2]
                distance = np.hypot(*target)

                reward    -= distance
            # Lost the target: decrease reward further
            else:
                reward -= 1
        else:
            self.invalidDetecs += 1
    

        # Penalize the agent for looking at the sky or at the ground
        if abs(self.currentY) > 0.85:
            reward -= 1
        
        done      = self.numberKilled  >= 12
        truncated = self.numSteps >= 200
        
        # update for next step
        (nextDetectionDx, nextDetectionDy, nextDetectionIsValid), bb0 = computervision.selectTarget(gameFrame, 
                                                                                            self.detectionModel,
                                                                                            self.gameWindowWidth,
                                                                                            self.gameWindowHeight
                                                                                            )
        
        if bb0 is not None: # selectTarget() didn't find anything  
            self.tracker.init(gameFrame, bb0.astype(np.int32))

        self.previousAlive     = currentAlive
        self.detectionDx       = nextDetectionDx
        self.detectionDy       = nextDetectionDy
        self.detectionIsValid  = nextDetectionIsValid

        # 0.41464621474517566 is how much of the screen the in-game character moves
        # in the Y axis when actionDy is 1.0. That is, if actionDy = 1.0, how much of the in-game Y axis
        # moves.
        self.currentY  += actionDy * 0.41464621474517566
        self.currentY  = np.clip(self.currentY, -1, 1) # Cap the number to be in the range [-1, 1]

        print(confirmedKills, self.numberKilled)
        obs  = self._makeObs(self.detectionDx, self.detectionDy, self.detectionIsValid, self.currentY)
        
        time.sleep(0.3)

        return obs, reward, done, truncated, {}


    def _makeObs(self, dx, dy, isValid, currentY):
        # --- Generate numpy array with the observations --- 
        return np.array([dx, dy, isValid, currentY], dtype=np.float32)