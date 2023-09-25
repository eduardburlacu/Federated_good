#import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.vec_env import VecEnv
#from stable_baselines3.common.envs import SimpleMultiObsEnv
from src.RL_training import MONITOR_PATH

# Environment instantiate
env= Monitor(
    gym.make("CartPole-v1"),
    filename=MONITOR_PATH
)


model = PPO(
    policy = MlpPolicy,
    env=env,
    verbose=1
)
print(model)

#Evaluate initial policy
init_mean_reward, init_std_reward = evaluate_policy(
    model=model,
    env=env,
    n_eval_episodes=100,
    warn=False
)
print(f"Initial mean reward: {init_mean_reward:.2f}\n Initial std reward: {init_std_reward:.2f}")

# Train the agent for 10000 steps
model.learn(total_timesteps=10_000)

#Evaluate final policy
mean_reward, std_reward = evaluate_policy(
    model=model,
    env=env,
    n_eval_episodes=100,
    warn=False
)
print(f"Initial mean reward: {mean_reward:.2f}\n Initial std reward: {std_reward:.2f}")
