from double_dqn import make_env, DoubleDQNAgent

if __name__ == 'main':
    train_env = make_env()

    agent = DoubleDQNAgent(train_env)
    agent.train()
