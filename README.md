# 🎯 Vision-Only DRL Aimbot for Counter-Strike: Global Offensive

> Deep Reinforcement Learning + Computer Vision = 💥 super-human aim without reading game memory

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**TL;DR**  
This repo contains my code and pretrained weights for the DRL agent described in my paper **“Deep Reinforcement Learning for Counter-Strike: Global Offensive”**.  
The agent uses screenshots only, detects enemies in the game frame, and learns mouse movements + “shoot” via PPO. On the aim-trainer map *Aim Botz* it averages **93.6 ± 2.8 kills-per-minute (KPM) on stationary targets and 39.2 ± 1.7 KPM on moving targets** —roughly 2× faster than human players (Master Guardian I - Dintinguished Master Guardian rank).

## Demo
| | |
|---------------------------|-------|
| Stationary Targets | [stationary](https://private-user-images.githubusercontent.com/49131143/462860001-a9778091-6546-42d3-a44d-74c1334d4042.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTE3NzE4ODIsIm5iZiI6MTc1MTc3MTU4MiwicGF0aCI6Ii80OTEzMTE0My80NjI4NjAwMDEtYTk3NzgwOTEtNjU0Ni00MmQzLWE0NGQtNzRjMTMzNGQ0MDQyLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA3MDYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNzA2VDAzMTMwMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTNhZTA1OWM1NjUxYmYxODNjOTE0MGJmNmJjMDBkOGQ4YzMxMzg3NmM4NzM5Y2M2NzhmNzJhNjViMjE4NGE5OTQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.MOzLNwbgvk7fdnlZSU9gDeA8jDnFSkH9i_m10CmJkbU) |
| Moving Targets | [moving](https://private-user-images.githubusercontent.com/49131143/462860001-a9778091-6546-42d3-a44d-74c1334d4042.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTE3NzE4ODIsIm5iZiI6MTc1MTc3MTU4MiwicGF0aCI6Ii80OTEzMTE0My80NjI4NjAwMDEtYTk3NzgwOTEtNjU0Ni00MmQzLWE0NGQtNzRjMTMzNGQ0MDQyLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA3MDYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNzA2VDAzMTMwMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTNhZTA1OWM1NjUxYmYxODNjOTE0MGJmNmJjMDBkOGQ4YzMxMzg3NmM4NzM5Y2M2NzhmNzJhNjViMjE4NGE5OTQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.MOzLNwbgvk7fdnlZSU9gDeA8jDnFSkH9i_m10CmJkbU) |
| | |

---

## Key Features
|  |  |
|------|---------------|
| **Vision-only pipeline** | No game memory access -> portable & encourages fair-play |
| **Two-stage training** | 2M steps in a **VirtualEnv** for speed, then 100k steps in the real game to fine-tune |
| **Frame-based Object detection** | Fast head-box detection on each frame |
| **Optional ViT tracker** | Predicts target motion to compensate vision latency|
| **Plug-and-play PPO agent** | Can theoretically be used to teach aiming mechanics in any 3D game |
---

## Demo



## 🏁 Quick Start

```bash
# 1) Clone
git clone https://github.com/RafaelAmauri/Reinforcement-Learning-CSGO
cd Reinforcement-Learning-CSGO

# 2) Create env with python version 3.12.10 (You can get it here: https://www.python.org/downloads/release/python-31210/)
python -m venv myenv
# 3) Activate the env (my example uses CMD. Check if your terminal is using CMD or PowerShell)
myenv/Scripts/activate.bat

# 4) Install deps
pip install -r requirements.txt

# 5) Compile the tensorRT detection model
python exportengine.py

# 6) Run demo on CS:GO (game settings: windowed, 1920×1080, FOV 90, mouse sentivity 1.0, disable raw_input in mouse settings)
python main.py