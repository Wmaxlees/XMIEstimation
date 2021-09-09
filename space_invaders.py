import gym
from scipy.stats import norm
import numpy as np
import math
from skimage.color import rgb2gray
import tensorflow as tf
from sklearn.feature_selection import mutual_info_regression

gamma = 0.99

def main():
    env = gym.make('SpaceInvaders-v0')

    states = []
    rewards = []
    for _ in range(500):
        non_discounted_rewards = []

        state = env.reset()
        state = process_state(state)
        states.append(state)
        non_discounted_rewards.append(0)
        done = False
        while not done:
            action = env.action_space.sample()

            state, reward, done, _ = env.step(action)

            state = process_state(state)
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


def process_state(state):
    state = rgb2gray(state)
    state = np.expand_dims(state, 2)
    state = tf.image.resize(state, (84, 84))
    state = np.reshape(state, (7056,))
    state = state[-1764:] # Just keep 1/4 of the input
    return state



if __name__ == '__main__':
    main()