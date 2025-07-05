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
          
          screenshot1 = time.perf_counter()
          gameFrame   = cam.get_latest_frame()
          screenshot1_end = time.perf_counter()

          metrics.append({"Screenshot1": f"{(screenshot1_end - screenshot1) * 1000:.2f} ms"})
          
          det = screenshot1_end
          (detectionDx, detectionDy, detectionIsValid), bb0 = computervision.selectTarget(gameFrame, 
                                                                                         detectionModel,
                                                                                         gameWindowWidth,
                                                                                         gameWindowHeight
                                                                                         )
          det_end = time.perf_counter()
          metrics.append({"Detection": f"{(det_end - det) * 1000:.2f} ms"})

          if detectionIsValid == 0:
               invalidDetecs += 1
          else:
               invalidDetecs = 0
               
               tracker.init(gameFrame, bb0.astype(np.int32))

               screenshot2 = time.perf_counter()
               gameFrame2  = cam.get_latest_frame()
               screenshot2_end = time.perf_counter()
               metrics.append({"Screenshot2": f"{(screenshot2_end - screenshot2) * 1000:.2f} ms"})

               tracking = time.perf_counter()

               ok, bb1  = tracker.update(gameFrame2)
               tracking_end = time.perf_counter()
               metrics.append({"Tracking": f"{(tracking_end - tracking) * 1000:.2f} ms"})

               if ok:
                    bb0Center = np.asarray([bb0[0] + bb0[2] / 2, bb0[1] + bb0[3] / 2])
                    bb1Center = np.asarray([bb1[0] + bb1[2] / 2, bb1[1] + bb1[3] / 2])

                    needsTracker = np.hypot(*(bb0Center - bb1Center)) > 2.8

               else:
                    needsTracker = False
               
               if needsTracker:
                    
                    if not ok:
                         continue

                    dt = screenshot2 - screenshot1_end
                    vx = (bb1[0] - bb0[0]) / dt # Pixels / segundo
                    vy = (bb1[1] - bb0[1]) / dt # Pixels / segundo

                    detectionDx = bb1[0] + vx * (tracking_end - screenshot2 + 0.001) # Prevê o novo X levando em conta a velocidade
                    detectionDy = bb1[1] + vy * (tracking_end - screenshot2 + 0.001) # Prevê o novo Y levando em conta a velocidade
                              
                    # Normaliza no intervalo [-1, 1]
                    detectionDx = (detectionDx + bb1[2] * 0.52)  / gameWindowWidth
                    detectionDx = detectionDx * 2 - 1
                    
                    # Normaliza no intervalo [-1, 1]
                    detectionDy = (detectionDy + bb1[3] * 0.12)  / gameWindowHeight
                    detectionDy = detectionDy * 2 - 1

                    end = time.perf_counter()

          
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

          time.sleep(0.12)