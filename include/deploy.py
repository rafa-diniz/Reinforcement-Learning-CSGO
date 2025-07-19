import time
import dxcam
import numpy as np

from include import computervision
from include import utils


def deployAgent(agent, detectionModel):
     gameWindowX0,    gameWindowY0, \
     gameWindowWidth, gameWindowHeight = utils.getCSGOWindowDimensions()

     maxActionDx, maxActionDy = computervision.pixelsToCounts(1, 1, gameWindowWidth, gameWindowHeight)
     
     region = (gameWindowX0, gameWindowY0, gameWindowX0 + gameWindowWidth, gameWindowY0 + gameWindowHeight)
     
     cam    = dxcam.create()
     cam.start(region=region, target_fps=120)

     currentY      = 0.0
     invalidDetecs = 0

     while True:
          metrics_pipeline = time.perf_counter()

          metrics = []
          if invalidDetecs == 10:
               break
          
          # Take first screenshot
          screenshot1     = time.perf_counter()
          gameFrame       = cam.get_latest_frame()
          screenshot1_end = time.perf_counter()

          metrics.append({"Screenshot1": f"{(screenshot1_end - screenshot1) * 1000:.2f} ms"})
          time.sleep(0.01)
          # Take second screenshot
          screenshot2     = time.perf_counter()
          gameFrame2      = cam.get_latest_frame()
          screenshot2_end = time.perf_counter()

          metrics.append({"Screenshot2": f"{(screenshot2_end - screenshot2) * 1000:.2f} ms"})
          
          timeElapsed = screenshot2_end - screenshot1
          
          metrics_tracking = time.perf_counter()
          detectionDx, detectionDy, detectionIsValid = computervision.selectTarget(gameFrame, 
                                                                                   gameFrame2, 
                                                                                   detectionModel,
                                                                                   gameWindowWidth,
                                                                                   gameWindowHeight,
                                                                                   timeElapsed 
                                                                                   )
          
          metrics.append({"tracking": f"{(time.perf_counter() - metrics_tracking) * 1000:.2f} ms"})
          
          if detectionIsValid == 0:
               invalidDetecs += 1
          else:
               invalidDetecs = 0
     
          metrics.append({"pipeline": f"{(time.perf_counter() - metrics_pipeline) * 1000:.2f} ms"}) 
          
          metrics_inference = time.perf_counter()
          output, _ = agent.predict(np.array([detectionDx, detectionDy, detectionIsValid, currentY]),
                                   deterministic=True
                                   )
          metrics.append({"inference": f"{(time.perf_counter() - metrics_inference) * 1000:.2f} ms"})

          
          print(metrics)

          
          actionDx, actionDy, shootProbability = output

          dxMouseUnits, dyMouseUnits = np.ceil(actionDx * maxActionDx), np.ceil(actionDy * maxActionDy)
          dxMouseUnits, dyMouseUnits = int(dxMouseUnits), int(dyMouseUnits)
          

          utils.moveMouse(dxMouseUnits, dyMouseUnits)

          # O flick do mouse é muito rápido, e pode jogar o ponteiro do mouse para fora da janela! 
          # Esse time.sleep é para o jogo não perder o foco do ponteiro do mouse.
          time.sleep(0.006)
          if np.random.rand() < shootProbability:
               utils.leftClick()
    
          currentY  += actionDy * 0.41464621474517566
          currentY  = np.clip(currentY, -1, 1)
          
          time.sleep(0.2)