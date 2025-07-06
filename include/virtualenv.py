import numpy as np
import gymnasium as gym

from include import computervision


class VirtualEnv(gym.Env):
    def __init__(self, closenessTolerance, closenessToleranceDecay, closenessToleranceDecayFreq):
        super().__init__()
        self.maxMove = 1.0 # The range of motion allowed. 

        self.nBotsInObservation = 1  # How many bots should be in the observation (more bots = more inputs in the MLP)
        self.nBotsInServer      = 12 # How many bots can be detected in-game at a given time
        
        # [dx, dy, shootProbability]
        self.action_space = gym.spaces.Box(
            low = np.array([-self.maxMove, -self.maxMove, 0.0], dtype=np.float32),
            high= np.array([ self.maxMove,  self.maxMove, 1.0], dtype=np.float32),
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

        self.offsetX, self.offsetY, \
        self.gameWindowWidth, self.gameWindowHeight = 0, 0, 1920, 1080

        self.maxActionDx, self.maxActionDy = computervision.pixelsToCounts(1, 1, self.gameWindowWidth, self.gameWindowHeight)

        self.currentY                    = 0.0                          # The current orientation in the y axis. -1 means looking up high at the sky, 1 means looking straight down at the ground.
        self.previousAlive               = None                         # The number of bots that were alive before the action was undertaken
        self.detectionIsValid            = None                         # If a bot was detected in timestep t-1
        self.detectionDx                 = None                         # The position in the X axis for the bot detected in timestep t-1
        self.detectionDy                 = None                         # The position in the Y axis for the bot detected in timestep t-1
        self.numSteps                    = 0                            # The total number of steps in this episode
        self.numTotalSteps               = 0                            # The total number of steps overall
        self.closenessTolerance          = closenessTolerance           # How close an action has to be to count as a "kill". Will decay over time
        self.closenessToleranceDecay     = closenessToleranceDecay      # By how much the closeness decays
        self.closenessToleranceDecayFreq = closenessToleranceDecayFreq  # After how many total steps it should decay
        self.seed                        = None


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reduce the tolerance every self.closenessToleranceDecayFreq steps        
        if self.numTotalSteps > self.closenessToleranceDecayFreq:
            self.closenessTolerance = self.closenessTolerance * self.closenessToleranceDecay
            self.numTotalSteps      = 0

        # Resets the Y axis so the bot doesn't get stuck looking straight down
        self.currentY = 0.0
        
        # Generate a mock target. Can be valid or not
        self.detectionDx, self.detectionDy, self.detectionIsValid  = self._random_target()

        # The number of previous bots alive always resets to 12
        self.previousAlive  = 12

        # Reset the number of steps taken 
        self.numSteps       = 0

        # Generate an observation
        obs = self._makeObs(self.detectionDx, self.detectionDy, self.detectionIsValid, self.currentY)

        return obs, {}


    def step(self, action):
        self.numSteps      += 1
        self.numTotalSteps += 1

        # Parse the action chosen by the network
        actionDx, actionDy, shootProbability = action
        
        # 0.41464621474517566 is how much of the screen the in-game character moves
        # in the Y axis when actionDy is 1.0. That is, if actionDy = 1.0, how much of the in-game Y axis
        # moves.
        self.currentY  += actionDy * 0.41464621474517566
        self.currentY  = np.clip(self.currentY, -1, 1) # Cap the number to be in the range [-1, 1]

        reward = 0.0
        currentAlive = self.previousAlive

        target = np.asarray([self.detectionDx, self.detectionDy])
        # Calculate how far away the action is from the target
        dxPixels, dyPixels = actionDx * self.maxActionDx, actionDy * self.maxActionDy 
        pixelMovement      = np.asarray(computervision.countsToPixels(dxPixels, dyPixels, self.gameWindowWidth, self.gameWindowHeight, return_normalised=True))
        
        distance = (target - pixelMovement)
        distance = np.hypot(*distance)

        # Distance Reward: decreases the penalty the closer the agent is to the target
        if self.detectionIsValid:
            reward    -= distance

        # If the agent managed to successfully take a shot
        if np.random.rand() < shootProbability:
            # Reward for shooting and scoring a hit
            if self.detectionIsValid \
            and distance <= self.closenessTolerance:
                    currentAlive = self.previousAlive - 1

            # Penalty for shooting when there's no target
            elif not self.detectionIsValid:
                reward -= 0.5
            
        # Add a big reward for getting a kill
        reward  += (self.previousAlive - currentAlive) * 2.0

        # Penalize the agent for looking at the sky or at the ground
        if abs(self.currentY) > 0.85:
            reward -= 1

        done      = currentAlive  == 0
        truncated = self.numSteps >= 200

        # Print what happened in the last step of the episode
        if done or truncated:
            pass
            print(f"Target = {target}, Movement = {pixelMovement}, Shoot = {shootProbability:.2f}, Distance = {distance:.3f}, closeness = {self.closenessTolerance:.3f}, Reward = {reward:3f}, Steps Taken = {self.numSteps}, Enemies Killed = {12 - currentAlive}")


        detectionDx, detectionDy, isValid = self._random_target()

        # update for next step
        self.detectionIsValid  = isValid
        self.detectionDx       = detectionDx
        self.detectionDy       = detectionDy
        self.previousAlive     = currentAlive


        obs  = self._makeObs(self.detectionDx, self.detectionDy, self.detectionIsValid, self.currentY)
        
        return obs, reward, done, truncated, {}


    def _random_target(self):
        is_valid = np.random.rand() < 0.95  # 90 % chance of a visible bot
        
        if is_valid and abs(self.currentY) < 0.85:
            return np.random.uniform(-1, 1), np.random.uniform(-1, 1), 1
        else:
            return 0.0, 0.0, 0


    def _makeObs(self, dx, dy, isValid, currentY):
        # --- Generate numpy array with the observations --- 
        return np.array([dx, dy, isValid, currentY], dtype=np.float32)