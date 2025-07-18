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
     
     cam    = dxcam.create()
     cam.start(region=region, target_fps=120)

     currentY      = 0.0
     invalidDetecs = 0
     tracker       = cv2.TrackerVit.create()
     pngID         = -1

     while True:
          metrics_pipeline = time.perf_counter()

          pngID += 1
          metrics = []
          if invalidDetecs == 10:
               break
          
          # Take first screenshot
          screenshot1 = time.perf_counter()
          gameFrame   = cam.get_latest_frame()
          screenshot1_end = time.perf_counter()

          metrics.append({"Screenshot1": f"{(screenshot1_end - screenshot1) * 1000:.2f} ms"})
          
          time.sleep(0.03)

          # Take second screenshot
          screenshot2 = time.perf_counter()
          gameFrame2   = cam.get_latest_frame()
          screenshot2_end = time.perf_counter()
          
          # Run detection
          det = screenshot1_end
          (detectionDx, detectionDy, detectionIsValid), bb0 = computervision.selectTarget(gameFrame,
                                                                                          gameFrame2, 
                                                                                          detectionModel,
                                                                                          gameWindowWidth,
                                                                                          gameWindowHeight
                                                                                          )
          
          det_end = time.perf_counter()
          metrics.append({"Detection": f"{(det_end - det) * 1000:.2f} ms"})
          
          # If no target is found
          if detectionIsValid == 0:
               invalidDetecs += 1
          else:
               invalidDetecs = 0
               tracking = time.perf_counter()
               # Initialize the tracker
               tracker.init(gameFrame, bb0.astype(np.int32))

               screenshot2 = time.perf_counter()
               gameFrame2  = cam.get_latest_frame()
               screenshot2_end = time.perf_counter()
               metrics.append({"Screenshot2": f"{(screenshot2_end - screenshot2) * 1000:.2f} ms"})

               
               ok, bb1  = tracker.update(gameFrame2)
               tracking_end = time.perf_counter()

               metrics.append({"Tracking": f"{(tracking_end - tracking) * 1000:.2f} ms"})

               if ok:
                    iou = utils.iou_xywh(bb0, bb1)
                    metrics.append({"IoU": f"{iou:.2f}"})
                    needsTracker = True if iou < 1000 else False
               else:
                    needsTracker = False


               metrics.append({"needsTracker": needsTracker})

               if needsTracker:
                    if not ok:
                         continue

                    t = screenshot2_end - screenshot1
                    vx = (bb1[0] - bb0[0]) / t # Pixels / segundo, não normalizado
                    vy = (bb1[1] - bb0[1]) / t # Pixels / segundo, não normalizado

                    detectionDx = bb1[0] + vx * (tracking_end - screenshot2 + 0.08) # Prevê o novo X levando em conta a velocidade
                    detectionDy = bb1[1] # Prevê o novo Y levando em conta a velocidade
                    
                    # Normaliza no intervalo [-1, 1]
                    detectionDx = (detectionDx + bb0[2] * 0.52)  / gameWindowWidth
                    detectionDx = detectionDx * 2 - 1
                    
                    # Normaliza no intervalo [-1, 1]
                    detectionDy = (detectionDy + bb0[3] * 0.12)  / gameWindowHeight
                    detectionDy = detectionDy * 2 - 1

          metrics.append({"pipeline": f"{(time.perf_counter() - metrics_pipeline) * 1000:.2f} ms"})
          
          print(metrics)
          continue 
          output, _ = agent.predict(np.array([detectionDx, detectionDy, detectionIsValid, currentY]),
                                   deterministic=True
                                   )

          actionDx, actionDy, shootProbability = output

          dxMouseUnits, dyMouseUnits = np.ceil(actionDx * maxActionDx), np.ceil(actionDy * maxActionDy)
          dxMouseUnits, dyMouseUnits = int(dxMouseUnits), int(dyMouseUnits)
          

          utils.moveMouse(dxMouseUnits, dyMouseUnits)

          # O flick do mouse é muito rápido, e pode jogar o ponteiro do mouse para fora da janela! 
          # Esse time.sleep é para o jogo não perder o foco do ponteiro do mouse.
          time.sleep(0.004)
          if np.random.rand() < shootProbability:
               utils.leftClick()
          
          currentY  += actionDy * 0.41464621474517566
          currentY  = np.clip(currentY, -1, 1)
          
          time.sleep(0.13)


'''
def deployAgent(agent, detectionModel):
     gameWindowX0,    gameWindowY0, \
     gameWindowWidth, gameWindowHeight = utils.getCSGOWindowDimensions()

     maxActionDx, maxActionDy = computervision.pixelsToCounts(1, 1, gameWindowWidth, gameWindowHeight)
     
     region = (gameWindowX0, gameWindowY0, gameWindowX0 + gameWindowWidth, gameWindowY0 + gameWindowHeight)
     
     cam    = dxcam.create()
     cam.start(region=region, target_fps=120)

     currentY      = 0.0
     invalidDetecs = 0
     tracker       = cv2.TrackerVit.create()
     pngID         = -1

     while True:
          metrics_pipeline = time.perf_counter()

          pngID += 1
          metrics = []
          if invalidDetecs == 10:
               break
          
          # Take first screenshot
          screenshot1 = time.perf_counter()
          gameFrame   = cam.get_latest_frame()
          screenshot1_end = time.perf_counter()

          metrics.append({"Screenshot1": f"{(screenshot1_end - screenshot1) * 1000:.2f} ms"})
          
          # Run detection
          det = screenshot1_end
          (detectionDx, detectionDy, detectionIsValid), bb0 = computervision.selectTarget(gameFrame, 
                                                                                         detectionModel,
                                                                                         gameWindowWidth,
                                                                                         gameWindowHeight
                                                                                         )
          
          det_end = time.perf_counter()
          metrics.append({"Detection": f"{(det_end - det) * 1000:.2f} ms"})

          # If no target is found
          if detectionIsValid == 0:
               invalidDetecs += 1
          else:
               invalidDetecs = 0
               tracking = time.perf_counter()
               # Initialize the tracker
               tracker.init(gameFrame, bb0.astype(np.int32))

               screenshot2 = time.perf_counter()
               gameFrame2  = cam.get_latest_frame()
               screenshot2_end = time.perf_counter()
               metrics.append({"Screenshot2": f"{(screenshot2_end - screenshot2) * 1000:.2f} ms"})

               
               ok, bb1  = tracker.update(gameFrame2)
               tracking_end = time.perf_counter()

               metrics.append({"Tracking": f"{(tracking_end - tracking) * 1000:.2f} ms"})

               if ok:
                    iou = utils.iou_xywh(bb0, bb1)
                    metrics.append({"IoU": f"{iou:.2f}"})
                    needsTracker = True if iou < 1000 else False
               else:
                    needsTracker = False


               metrics.append({"needsTracker": needsTracker})

               if needsTracker:
                    if not ok:
                         continue

                    t = screenshot2_end - screenshot1
                    vx = (bb1[0] - bb0[0]) / t # Pixels / segundo, não normalizado
                    vy = (bb1[1] - bb0[1]) / t # Pixels / segundo, não normalizado

                    detectionDx = bb1[0] + vx * (tracking_end - screenshot2 + 0.08) # Prevê o novo X levando em conta a velocidade
                    detectionDy = bb1[1] # Prevê o novo Y levando em conta a velocidade
                    
                    # Normaliza no intervalo [-1, 1]
                    detectionDx = (detectionDx + bb0[2] * 0.52)  / gameWindowWidth
                    detectionDx = detectionDx * 2 - 1
                    
                    # Normaliza no intervalo [-1, 1]
                    detectionDy = (detectionDy + bb0[3] * 0.12)  / gameWindowHeight
                    detectionDy = detectionDy * 2 - 1
                         

          from PIL import Image
          gameFrame  = Image.fromarray(gameFrame)
          gameFrame2 = Image.fromarray(gameFrame2)

          gameFrame.save(f"./pngs/{pngID}_1.png")
          gameFrame2.save(f"./pngs/{pngID}_2.png")
          raise Exception

          metrics.append({"pipeline": f"{(time.perf_counter() - metrics_pipeline) * 1000:.2f} ms"})
          
          print(metrics)
          continue 
          output, _ = agent.predict(np.array([detectionDx, detectionDy, detectionIsValid, currentY]),
                                   deterministic=True
                                   )

          actionDx, actionDy, shootProbability = output

          dxMouseUnits, dyMouseUnits = np.ceil(actionDx * maxActionDx), np.ceil(actionDy * maxActionDy)
          dxMouseUnits, dyMouseUnits = int(dxMouseUnits), int(dyMouseUnits)
          

          utils.moveMouse(dxMouseUnits, dyMouseUnits)

          # O flick do mouse é muito rápido, e pode jogar o ponteiro do mouse para fora da janela! 
          # Esse time.sleep é para o jogo não perder o foco do ponteiro do mouse.
          time.sleep(0.004)
          if np.random.rand() < shootProbability:
               utils.leftClick()
          
          currentY  += actionDy * 0.41464621474517566
          currentY  = np.clip(currentY, -1, 1)
          
          time.sleep(0.13)
'''