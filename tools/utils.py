import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# basic plt settings

def plot_rewards(ts, smoothing=0, color='b', legend=None):
    smoothing = min(1, smoothing)
    window = max(1, int(len(ts)*smoothing))
    ts = pd.Series(ts).rolling(window).mean()
    plt.axhline(y=0.78, linewidth=1, color='r', linestyle='dashed')
    if legend:
        plt.plot(ts, color=color, label=legend)
        plt.legend(loc='upper left')
    else:
        plt.plot(ts, color=color)

class Experiment():

    def __init__(self, env):
        self.env = env

    def fit(self, agent, episodes):
        """ Agent interacts with the environment """

        for _ in range(episodes):
            prev_obs = self.env.reset()
            agent.reset()
            done = False
            steps =0    
            while not done:
                steps+=1
                # self.env.render()
                action = agent.act(prev_obs)
                next_obs, reward, done, info = self.env.step(action)
                agent.learn(prev_obs, action, next_obs, reward, done)
                prev_obs = next_obs
