# Vision-Based Reinforcement Learning for 3D Aiming Control

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12.10-blue.svg)](https://www.python.org/)

**TL;DR**

This repo contains my code and trained weights for a Deep Reinforcement Learning agent that learns aiming mechanics in Counter-Strike using visual input and PPO.

The pipeline captures game frames, detects valid targets and uses PPO to learn the mouse movements required to move the crosshair towards them. On the community map *Aim Botz* it averages **93.6 ± 2.8 kills-per-minute (KPM) on stationary targets and 39.2 ± 1.7 KPM on moving targets**. In my tests, this was roughly 2x faster than the human players I compared it against (Master Guardian I - Distinguished Master Guardian rank).


## Demo
| | |
|---------------------------|-------|
| Stationary Targets | ![stationary](demos/stationary.gif) |
| Moving Targets | ![moving](demos/moving.gif) |
| | |
---

## Motivation

In operating systems, cursor movement is tracked in "Mouse Units", which correspond exactly to screen pixels when interacting with a flat 2D interface - web browsers, for example. In 3D games, however, the scene is rendered in three dimensions and then projected onto a 2D plane, with said projection being shaped by camera parameters such as horizontal/vertical FOV, focal distance, and aspect ratio (4:3, 16:9, 21:9, etc.). As a result, a target being 100 pixels away from the crosshair does not necessarily mean that moving the mouse by 100 units will place the crosshair on it. The mapping depends on the game's camera and input settings.

One way to solve this would be to explicitly model or calibrate the relationship between screen-space distance and mouse movement. The problem is that this mapping depends on several game and camera settings, and building a clean supervised dataset for every possible game configuration would be pretty annoying.

This is, however, a pretty nice use case for Deep Reinforcement Learning. Instead of explicitly deriving this mapping, an RL agent can learn it by interacting with the game: move the mouse, observe how the target moved relative to the crosshair, and use that feedback to improve its next action.

This is the main idea behind my Reinforcement Learning agent, using screen capture and targets inside Counter-Strike to learn the mapping between 2D screen-space target distance and the mouse movements required to aim at them. By training the agent with the appropriate reward function, the network can eventually learn to aim in Counter-Strike.


## Disclaimer
This project was built as a proof of concept for combining Reinforcement Learning and Computer Vision to teach an agent visual control in a real 3D game. All the experiments and benchmarks in this repo were performed in the Aim Botz training environment.

## Key Features
|  |  |
|------|---------------|
| **PPO aiming agent** | The same training idea can theoretically be adapted to other 3D environments |
| **Modern ML stack** | Uses Proximal Policy Optimization, lightweight Visual Transformers for tracking and real-time object detection models.
| **Vision-only pipeline** | Works from captured frames without reading the game's internal memory |
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

# 6) Run the agent on the Aim Botz training map
# If loading the trained weights, make sure to use the same game settings as me - the weights learned the screen-space -> mouse-movement mapping for my game settings; if you want to use different game settings, you'll have to train the agent from scratch. My settings: windowed, 1920×1080, FOV 90, mouse sentivity 1.0, mouse yaw and mouse pitch = 0.022, disable raw_input in mouse settings)
python main.py
```

## ⚙️ Design

The general architecture has two main components: A Computer Vision pipeline (Object Detection + Optical Character Recognition (OCR) + an optional ViT Tracker), and a Proximal Policy Optimization (PPO) component. 

![architecture](demos/architecture-tracker.png)

The Computer Vision component is responsible for extracting information from the environment, such as the positions of enemies, the velocity of a target and registering successful kills.

The reward function is simple as it only contains three components, but it still proved itself to be extremely powerful at teaching the agent the intended behavior.

The Distance Penalty term penalizes the agent for aiming the crosshair away from the target using $-\mu$. If the agent picks a movement large enough to lose the target, it receives a fixed -0.6 penalty instead.

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
