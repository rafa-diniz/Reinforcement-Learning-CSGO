import cv2
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
     cam = dxcam.create()
     cam.start(region=region, target_fps=120)


     currentY      = 0.0
     invalidDetecs = 0
     tracker       = cv2.TrackerVit.create()

     while True:
          metrics = []
          if invalidDetecs == 10:
               break
          
          metrics_screenshot1 = time.perf_counter()
          gameFrame           = cam.get_latest_frame()
          metrics_screenshot1 = time.perf_counter() - metrics_screenshot1
          metrics.append({"Screenshot1": f"{metrics_screenshot1 * 1000:.2f} ms"})
          
          metrics_detection = time.perf_counter()
          (detectionDx, detectionDy, detectionIsValid), bb0 = computervision.selectTarget(gameFrame, 
                                                                                         detectionModel,
                                                                                         gameWindowWidth,
                                                                                         gameWindowHeight
                                                                                         )
          metrics_detection = time.perf_counter() - metrics_detection
          metrics.append({"Detection": f"{metrics_detection * 1000:.2f} ms"})

          if detectionIsValid == 0:
               invalidDetecs += 1
          else:
               invalidDetecs = 0
               
               tracker.init(gameFrame, bb0.astype(np.int32))

               metrics_screenshot2 = time.perf_counter()
               gameFrame2          = cam.get_latest_frame()
               metrics_screenshot2 = time.perf_counter() - metrics_screenshot2
               metrics.append({"Screenshot2": f"{metrics_screenshot2 * 1000:.2f} ms"})

               metrics_tracking = time.perf_counter()
               ok, bb1          = tracker.update(gameFrame2)
               metrics_tracking = time.perf_counter() - metrics_tracking
               metrics.append({"Tracking": f"{metrics_tracking * 1000:.2f} ms"})

               if ok:
                    bb0Center = np.asarray([bb0[0] + bb0[2] / 2, bb0[1] + bb0[3] / 2])
                    bb1Center = np.asarray([bb1[0] + bb1[2] / 2, bb1[1] + bb1[3] / 2])

                    needsTracker = np.hypot(*(bb0Center - bb1Center)) > 5
               else:
                    needsTracker = False
               

               print(needsTracker)

               if needsTracker:
                    
                    if not ok:
                         continue
                    
                    dt = metrics_screenshot1 + metrics_detection + metrics_screenshot2
                    vx = (bb1[0] - bb0[0]) / dt # Pixels / segundo
                    vy = (bb1[1] - bb0[1]) / dt # Pixels / segundo

                    detectionDx = bb0[0] + vx * (metrics_screenshot1 + metrics_detection + metrics_screenshot2 + metrics_tracking + 0.015) # Prevê o novo X levando em conta a velocidade
                    detectionDy = bb0[1] + vy * (metrics_screenshot1 + metrics_detection + metrics_screenshot2 + metrics_tracking + 0.015) # Prevê o novo Y levando em conta a velocidade
                              
                    # Normaliza no intervalo [-1, 1]
                    detectionDx = (detectionDx + bb0[2] * 0.52)  / gameWindowWidth
                    detectionDx = detectionDx * 2 - 1
                    
                    # Normaliza no intervalo [-1, 1]
                    detectionDy = (detectionDy + bb0[3] * 0.12)  / gameWindowHeight
                    detectionDy = detectionDy * 2 - 1

          
          print(metrics)          
          output, _ = agent.predict(np.array([detectionDx, detectionDy, detectionIsValid, currentY]),
                                   deterministic=True
                                   )

          actionDx, actionDy, shootProbability = output

          dxMouseUnits, dyMouseUnits = np.ceil(actionDx * maxActionDx), np.ceil(actionDy * maxActionDy)
          dxMouseUnits, dyMouseUnits = int(dxMouseUnits), int(dyMouseUnits)
          

          utils.moveMouse(dxMouseUnits, dyMouseUnits)

          time.sleep(0.01)
          if np.random.rand() < shootProbability:
               utils.leftClick()


          currentY  += actionDy * 0.41464621474517566
          currentY  = np.clip(currentY, -1, 1)

          time.sleep(0.13)