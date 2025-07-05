import pathlib
import stable_baselines3 as sb3
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import load_results, Monitor
from ultralytics import YOLO

from include import csgoaimenv, utils, virtualenv, deploy


trainVirtualEnv  = False
trainCSGOAimEnv  = True
chartVirtualEnv  = False
chartCSGOAimEnv  = False

deployAgent      = False


detectionModel   = YOLO("yolo11m.engine", task="detect")

if trainVirtualEnv:
    totalSteps = 2_000_000
    
    log_dir = pathlib.Path("logs/virtual_env")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    env = virtualenv.VirtualEnv(closenessTolerance          = 0.015,
                                closenessToleranceDecay     = 0.70,
                                closenessToleranceDecayFreq = totalSteps // 4)

    env = Monitor(env, str(log_dir))

    agent = sb3.PPO("MlpPolicy",
                    env,
                    device="cpu",
                    verbose=1,
                    batch_size=256,
                    gamma=0.5,
                    clip_range=0.2,
                    target_kl=0.5,
                    learning_rate=utils.multistepSchedule
                    )

    agent.learn(total_timesteps=totalSteps)
    agent.save("ppo_virtualEnv.zip")


if chartVirtualEnv:
    log_dir = pathlib.Path("logs/virtual_env")
    df = load_results(str(log_dir))
    x  = df['l'].cumsum()
    y  = df['r']

    plt.figure(figsize=(8,4))
    plt.plot(x, y, '.', alpha=0.3, label='raw')
    plt.plot(x, y.rolling(50).mean(), linewidth=2, label='smooth (50 eps)')
    plt.xlabel("Environment steps")
    plt.ylabel("Episode reward")
    plt.legend()
    plt.tight_layout()
    plt.show()


if trainCSGOAimEnv:
    log_dir = pathlib.Path("logs/csgoaim_env")
    log_dir.mkdir(parents=True, exist_ok=True)

    env = csgoaimenv.CSGOAimEnv(detectionModel)
    env = Monitor(env, str(log_dir))

    agent    = sb3.PPO.load("ppo_virtualEnv.zip", env=env, device="cpu")
    
    agent.learn(total_timesteps=100_000)
    agent.save("ppo_csgoaimenv.zip")


if chartCSGOAimEnv:
    log_dir = pathlib.Path("logs/csgoaim_env")
    df = load_results(str(log_dir))
    x  = df['l'].cumsum()
    y  = df['r']

    plt.figure(figsize=(8,4))
    plt.plot(x, y, '.', alpha=0.3, label='raw')
    plt.plot(x, y.rolling(50).mean(), linewidth=2, label='smooth (50 eps)')
    plt.xlabel("Environment steps")
    plt.ylabel("Episode reward")
    plt.legend()
    plt.tight_layout()
    plt.show()



if deployAgent:
    print("Deploying agent into the game...")

    agent = sb3.PPO.load("ppo_virtualEnv.zip", env=None, device="cpu")
    
    deploy.deployAgent(agent, detectionModel)