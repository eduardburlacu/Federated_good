from stable_baselines3.common.env_checker import check_env
from src.RL.environment import EnvFL

if __name__=="__main__":
    env =EnvFL(num_rounds=100)
    check_env(env)