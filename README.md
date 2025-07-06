# 🎯 Deep Reinforcement Learning for Counter-Strike: Global Offensive

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**TL;DR**

This repo contains my code and trained weights for the Deep Reinforcement Learning network described in my paper **"Deep Reinforcement Learning for Counter-Strike: Global Offensive"**.  
The pipeline captures game frames, detects valid targets and learns mouse movements + "shoot" via PPO. On the community map *Aim Botz* it averages **93.6 ± 2.8 kills-per-minute (KPM) on stationary targets and 39.2 ± 1.7 KPM on moving targets** - roughly 2× faster than human players (Master Guardian I - Dintinguished Master Guardian rank).

## Technical Info
Teaching a Machine Learning network to aim in a 3D game can be challenging because games often hide the settings that were used for creating the in-game camera; Knowing these settings can allow the player to map every pixel in a game frame to their corresponding position in the 3D scene, "undoing" the 3D -> 2D projection that happens in 3D games. Doing this can help create a Machine Learning network that can theoretically aim in any 3D game.

Sadly, using Supervised Learning for this task is very challenging because commercial games rarely expose the in-engine settings for the game camera. Without the exact values for focal distance, yaw/pitch acceleration and horizontal/vertical FOV, it is impossible to set up a Supervised Learning training loop that undoes the 3D -> 2D projection.

This is, however, a great use case for Deep Reinforcement Learning. DRL algorithms learn from multiple experiences, and can be taught to approximate these values by using a reliable metric, such as a fixed target. This is the main idea behind my DRL agent - using screen capture and fixed targets inside Counter-Strike to approximate the 2D -> 3D counter-projection function. By training for thousands of steps with the appropriate reward function, the DRL network can eventually learn to aim in Counter-Strike.

## Disclaimer
I do not recommend using this as an aimbot. If you are looking for that, look for "YOLO Aimbot" or similar on GitHub and you'll find plenty of such projects. This one is a proof of concept for how Reinforcement Learning can be combined with Computer Vision to teach RL agents to aim in 3D games :P


## Demo
| | |
|---------------------------|-------|
| Moving Targets | ![test](demos/moving.mp4) |
| | |
---


## Key Features
|  |  |
|------|---------------|
| **Plug-and-play PPO agent** | Can theoretically be used to teach aiming mechanics in any 3D game |
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