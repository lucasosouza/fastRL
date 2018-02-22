from BaseAgent import BaseAgent
import numpy as np

class QLearning(BaseAgent):
    
    def __init__(self,env, qtable_default=0, learning_rate = 0.1, exploration_rate = 0.1, 
                 epsilon_decay = True, epsilon_decay_func= lambda x: x*.99,
                 discount_factor=0.9, 
                 alpha_decay=True, alpha_decay_func = lambda x: x*.999):
        BaseAgent.__init__(self, env)
        if self.action_discrete and self.observation_discrete:
            # init qtable with the default value for qtable
            # set to higher than 0 for optimistic initialization
            self.qtable={}
            for obs in range(self.observation_range):
                self.qtable[obs] = {}
                for action in range(self.action_range):
                    self.qtable[obs][action] = qtable_default
        else:
            raise TypeError("Environment not discrete.")
            
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_func = epsilon_decay_func
        self.alpha_decay = alpha_decay
        self.alpha_decay_func = alpha_decay_func
        self.discount_factor = discount_factor
                
    # learn and act are separate steps
    # act is just getting the argmax of qtable

    
    def reset(self):
        BaseAgent.reset(self)
        
        # if set to decay, decay exploration rate according to predefined decay function
        # don't decay during planning, since it is only look ahead
        if self.epsilon_decay:
            self.exploration_rate = self.epsilon_decay_func(self.exploration_rate)
            
        if self.alpha_decay:
            self.learning_rate = self.alpha_decay_func(self.learning_rate)

    def act(self, state, look_ahead=False):        
        """ Runs argmax on qtable to define next best action """
    
        # select randomly according to a fixed exploration rate
        if np.random.rand() < self.exploration_rate and not look_ahead:
            action = np.random.randint(self.action_range)
        
        else:
            # select all possible actions
            possible_actions = list(self.qtable[state].items())

            # shuffle before sorting, to ensure randomness in case of tie
            np.random.shuffle(possible_actions)
            action = sorted(possible_actions, key=lambda x:-x[1])[0][0]
            
        return action
    
    def learn(self, prev_state, prev_action, next_state, reward, done):
        """ Update qtable. Does not return anything. 
            Independent from agent's current state, except for qtable
        """
        
        # update reward count
        self.total_reward += reward
        
        if not done:        
            # select optimal action in next step, calc td target based on it
            next_action = self.act(next_state, look_ahead=True)
            td_target = reward + self.discount_factor * self.qtable[next_state][next_action]
            
        else:
            # if final state, td target equals rewards
            td_target = reward            
            
        # update q-table
        td_error = td_target - self.qtable[prev_state][prev_action]
        self.qtable[prev_state][prev_action] += self.learning_rate * td_error 
        
