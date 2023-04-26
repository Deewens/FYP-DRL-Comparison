import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack, RecordEpisodeStatistics, RecordVideo


def make_env(env_name="ALE/Pong-v5", seed=42):
    def step_trigger(step: int):
        return step % 500_000 == 0

    env = gym.make(env_name, render_mode="rgb_array", full_action_space=False, frameskip=1)
    env = AtariPreprocessing(env)
    env = FrameStack(env, 4)
    env = RecordEpisodeStatistics(env)
    # A video will be recorded every 400,000 steps.
    # (Can't use lambda expression because it is not supported by cloud pickle when saving checkpoint model)
    env = RecordVideo(env, "runs/videos/", step_trigger=step_trigger, video_length=1000)

    env.observation_space.seed(seed)
    env.action_space.seed(seed)

    return env
