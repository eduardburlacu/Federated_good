import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy, ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from src.RL.policy import LSTMExtractor
from src.RL import MONITOR_PATH

# Environment instantiate
env= Monitor(
    gym.make("CartPole-v1"),
    filename=MONITOR_PATH
)

model = PPO(
    policy = MlpPolicy,
    env=env,
    policy_kwargs= dict(
        features_extractor_class = LSTMExtractor,
        features_extractor_kwargs = dict(features_dim=18)
    ),
    batch_size=1,
    learning_rate=3E-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range= 0.2,
    verbose=1,
    seed=0
)

print(model)

#Evaluate initial policy
init_mean_reward, init_std_reward = evaluate_policy(
    model=model,
    env=env,
    n_eval_episodes=100,
    warn=False
)
print(f"Initial mean reward: {init_mean_reward:.2f} \n Initial std reward: {init_std_reward:.2f}")

# Train the agent for 10000 steps
#model.learn(total_timesteps=10_000)

#Evaluate final policy
#mean_reward, std_reward = evaluate_policy(
#    model=model,
#    env=env,
#    n_eval_episodes=100,
#    warn=False
#)
#print(f"Initial mean reward: {mean_reward:.2f}\n Initial std reward: {std_reward:.2f}")
