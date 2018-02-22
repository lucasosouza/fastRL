
import sys
sys.path.append('../agents')
sys.path.append('../environments')
sys.path.append('../tools')

import gym
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
from utils import *
from QLearning import QLearning
from SARSA import SARSA

# initialize env
env = Experiment(gym.make('FrozenLake-v0'))

tries = 100
episodes = 1000
results = np.zeros((tries, episodes))


#### run with QLearning

for t in range(tries):

    # define learning settings
    epsilon_decay = 1-(1/episodes)*6
    learning_decay = 1-(1/episodes)*3
    agent = QLearning(env.env, learning_rate =0.5, discount_factor=0.9, 
                      exploration_rate=0,
                      epsilon_decay_func = lambda x: x*epsilon_decay,
                      alpha_decay_func = lambda x: x*learning_decay,
                      qtable_default=1
                     )


    # fit and save results
    env.fit(agent, episodes)
    results[t, :] = agent.rewards_per_episode

# plot rewards
plot_rewards(np.mean(results, axis=0), smoothing=0.1, color='blue')

#### run with SARSA

# define learning settings

for t in range(tries):

    epsilon_decay = 1-(1/episodes)*6
    learning_decay = 1-(1/episodes)*3
    agent = SARSA(env.env, learning_rate =0.5, discount_factor=0.9, 
                      exploration_rate=0,
                      epsilon_decay_func = lambda x: x*epsilon_decay,
                      alpha_decay_func = lambda x: x*learning_decay,
                      qtable_default=1
                     )


    # fit and save results
    env.fit(agent, episodes)
    results[t, :] = agent.rewards_per_episode

# plot rewards
plot_rewards(np.mean(results, axis=0), smoothing=0.1, color='green')

#### show plot
plt.show()

"""
There is a component of randomness, but overall, it seems SARSA takes longer to converge
What if I run this a lot of times, and average over them
Can I remove the randomness component and have a better picture of what is going on?
"""
