from double_dqn import make_env, DoubleDQNAgent


def main():
    train_env = make_env()

    agent = DoubleDQNAgent(train_env)
    agent.train()


if __name__ == 'main':
    main()
