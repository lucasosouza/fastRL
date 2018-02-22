import numpy as np
from gym import spaces

class BaseAgent():
    
    def __init__(self, env):
        
        # define limits for continuous case and n for discrete case        
        if type(env.action_space) != spaces.discrete.Discrete:
            self.action_discrete = False
            self.action_high = env.action_space.low 
            self.action_low = env.action_space.high
        else:
            self.action_discrete = True
            self.action_range = env.action_space.n

        if type(env.observation_space) != spaces.discrete.Discrete:
            self.observation_discrete = False
            self.observation_high = env.observation_space.low 
            self.observation_low = env.observation_space.high
        else:
            self.observation_discrete = True
            self.observation_range = env.observation_space.n
        
        # keep track of metrics
        self.total_reward = 0
        self.rewards_per_episode = []
        
    def reset(self):
        self.rewards_per_episode.append(self.total_reward)
        self.total_reward= 0

    def act(self, current_state):
        if self.action_discrete:
            return np.random.randint(self.action_range)        
        
    def learn(self, obs, reward, done):
        pass
    
    def total_rewards(self):
        return sum(self.rewards)
