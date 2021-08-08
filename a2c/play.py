import sys
import os

sys.path.append(os.path.abspath(".."))
from a2c.agent import TestAgent
from environment.cartpole import CartPole

def main():
    env = CartPole()
    for _ in range(10):
        env.reset()
        state = env.get_screen()
        agent = TestAgent(env.get_n_actions())

        total_reward = 0
        while True:
            action = agent.select(state)
            _, reward, done, _  = env.step(action)
            state = env.get_screen()
            total_reward += reward

            if done:
                break

        print("reward: {}".format(total_reward))

if __name__=='__main__':
    main()
