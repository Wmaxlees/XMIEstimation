import gym
from scipy.stats import norm
import numpy as np
import math
from sklearn.feature_selection import mutual_info_regression

gamma = 0.99

def main():
    env = gym.make('CartPole-v1')


    states = []
    rewards = []
    for _ in range(500):
        non_discounted_rewards = []

        state = env.reset()
        states.append(state)
        non_discounted_rewards.append(0)
        done = False
        while not done:
            action = env.action_space.sample()

            state, reward, done, _ = env.step(action)

            states.append(state)
            non_discounted_rewards.append(reward)

        for rew_idx in range(len(non_discounted_rewards)-1, -1, -1):
            if rew_idx == len(non_discounted_rewards)-1:
                continue

            non_discounted_rewards[rew_idx] = non_discounted_rewards[rew_idx] + (gamma * non_discounted_rewards[rew_idx+1])

        rewards = rewards + non_discounted_rewards

    rewards = np.array(rewards)
    states = np.array(states)

    mis = mutual_info_regression(states, rewards)

    print('Mean: {}'.format(np.mean(mis)))


if __name__ == '__main__':
    main()