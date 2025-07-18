# 🎯 Deep Reinforcement Learning for Counter-Strike: Global Offensive

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12.10-blue.svg)](https://www.python.org/)

**TL;DR**

This repo contains my code and trained weights for the Deep Reinforcement Learning network described in my paper **"Deep Reinforcement Learning for Counter-Strike: Global Offensive"**.  
The pipeline captures game frames, detects valid targets and learns mouse movements + "shoot" via PPO. On the community map *Aim Botz* it averages **93.6 ± 2.8 kills-per-minute (KPM) on stationary targets and 39.2 ± 1.7 KPM on moving targets** - roughly 2× faster than human players (Master Guardian I - Dintinguished Master Guardian rank).

## Technical Info
In operating systems, cursor movement is tracked in "Mouse Units", which correspond exactly to screen pixels when interacting with a flat 2D interface - web browsers, for example. In 3D games, however, the scene is rendered in three dimensions and then projected onto a 2D plane, with said projection being shaped by camera parameters such as horizontal/vertical FOV, focal distance, and aspect ratio (4:3, 16:9, 21:9, etc.). As a result, moving the mouse 100 units in a 3D game won't necessarily move the on-screen crosshair by 100 pixels, because the projection math has to be “undone,” and that can only be done perfectly if the exact camera settings are known.

Sadly, using Supervised Learning for this task is very challenging because commercial games rarely expose the in-engine settings for the game camera. Without the exact intrinsic camera settings it is impossible to set up a Supervised Learning training loop that undoes the 3D -> 2D projection.

This is, however, a great use case for Deep Reinforcement Learning. DRL algorithms learn from multiple experiences, and can be taught to approximate these values by using a reliable metric, such as a fixed target. This is the main idea behind my DRL agent - using screen capture and fixed targets inside Counter-Strike to approximate the 2D -> 3D counter-projection function. By training the agent with the appropriate reward function, the DRL network can eventually learn to aim in Counter-Strike.


## Disclaimer
I do not recommend using this as an aimbot. If you are looking for that, look for "YOLO Aimbot" or similar on GitHub and you'll find plenty of such projects. This one is a proof of concept for how Reinforcement Learning can be combined with Computer Vision to teach RL agents to aim in 3D games :P


## Demo
| | |
|---------------------------|-------|
| Stationary Targets | ![stationary](demos/stationary.gif) |
| Moving Targets | ![moving](demos/moving.gif) |
| | |
---


## Key Features
|  |  |
|------|---------------|
| **Plug-and-play PPO agent** | Can theoretically be used to teach aiming mechanics in any 3D game |
| **State of the Art** | Uses Proximal Policy Optimization, ultra-lightweight Visual Transformers and one-shot detection models.
| **Vision-only pipeline** | No game memory access -> portable & encourages fair-play |
| **Two-stage training** | 2M steps in a **VirtualEnv** for speed, then 100k steps in the real game to fine-tune |
| **Frame-based Object detection** | Fast head-box detection on each frame |
| **Optional ViT tracker** | Predicts target motion to compensate vision latency |
---


## 🏁 Quick Start

```bash
# 1) Clone
git clone https://github.com/RafaelAmauri/Reinforcement-Learning-CSGO
cd Reinforcement-Learning-CSGO

# 2) Create a virtual env with python version 3.12.10 (You can get it here: https://www.python.org/downloads/release/python-31210/)
python -m venv myenv
# 3) Activate the env (my example uses CMD. Check if your terminal is using CMD or PowerShell)
myenv/Scripts/activate.bat

# 4) Install deps
pip install -r requirements.txt

# 5) Compile the tensorRT detection model
python exportengine.py

# 6) Run the demo on CS:GO
# If loading the trained weights, make sure to use the same game settings as me - the weights learned the 2D -> 3D mapping for my game settings; if you want to use different game settings, you'll have to train the agent from scratch. My settings: windowed, 1920×1080, FOV 90, mouse sentivity 1.0, mouse yaw and mouse pitch = 0.022, disable raw_input in mouse settings)
python main.py
```

## ⚙️ Design

The general architecture has two main components: A Computer Vision pipeline (Object Detection + Optical Character Recognition (OCR) + an optional ViT Tracker), and a Proximal Policy Optimization (PPO) component. 

![architecture](demos/architecture-tracker.png)

The Computer Vision component is responsible for extracting information from the environment, such as the positions of enemies, the velocity of a target and registering successful kills. All this information is used to evaluate the PPO network's actions and drive it towards the desired behavior.

The reward function is simple as it only contains three components, but it still proved itself to be extremely powerful at teaching the agent the intended behavior.

The Distance Penalty term penalizes the agent for aiming the crosshair away from the target with the $-\mu$. In case the agent picks a mouse movement that moves the crosshair so much that the target is no longer in the frame, a -0.6 fixed reward is discounted.

The Kill Bonus rewards the agent for getting kills.

The View Angle Penalty penalizes the agent for aiming at the ground or at the sky and encourages it to look forward.

The full reward function is: 

$$
\mathcal{R} =
\underbrace{
  \begin{cases}
    -\mu, & \text{if detection is valid and target was not lost by tracker} \\
    -0.6, & \text{if detection is valid and target was lost by tracker} \\
    0,    & \text{if detection is not valid}
  \end{cases}
}_{\text{Distance Penalty}}
\;+\;
\underbrace{2.0 \cdot (\text{Number of Kills})}_{\text{Kill Bonus}}
\;+\;
\underbrace{
  \begin{cases}
    -0.6,             & \text{if }abs(\text{Current Y}) > 0.85 \\ 
    0,                & \text{otherwise}
  \end{cases}
}_{\text{View Angle Penalty}}
$$
