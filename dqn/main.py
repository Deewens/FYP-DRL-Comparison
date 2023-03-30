from double_dqn import DoubleDQNAgent
from environment import make_env

if __name__ == "__main__":
    train_env = make_env()

    agent = DoubleDQNAgent(train_env)
    # agent.load_checkpoint("checkpoints")
    agent.train(from_checkpoint=False)