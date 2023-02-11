import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack


def make_env(env_name="ALE/Pong-v5", seed=42):
    env = gym.make(env_name, render_mode="rgb_array", full_action_space=False, frameskip=1)
    env = AtariPreprocessing(env)
    # env = RecordEpisodeStatistics(env)
    env = FrameStack(env, 4)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)

    return env
