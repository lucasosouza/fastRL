from BaseAgent import BaseAgent
import numpy as np

class MonteCarloEV(BaseAgent):
    """ Monte Carlo agent, every visit version 
        MonteCarlo defaults with 0 exploration rate, and no decay
    """
    
    def __init__(self,env, qtable_default=0, exploration_rate = 0.1, 
                 epsilon_decay = True, epsilon_decay_func= lambda x: x*.99,
                 discount_factor=0.9):

        BaseAgent.__init__(self, env)

        # init policy, qtable, and qcount (#times a state,action pair is visited)
        self.qtable = {}
        self.qcount = {}
        self.policy = {}
        if self.observation_discrete:
            for obs in range(self.observation_range):
                # initialize policy at random
                self.policy[obs] = env.action_space.sample()
                self.qtable[obs] = {}
                self.qcount[obs] = {}
                for action in range(self.action_range):
                    # set default value higher than 0 for optimistic initialization
                    self.qtable[obs][action] = qtable_default
                    # initialize count of visited state,action to 0
                    self.qcount[obs][action] = 0
        else:
            raise TypeError("Environment not discrete.")
        
        # exploration rate + decay
        self.exploration_rate = exploration_rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_func = epsilon_decay_func

        self.discount_factor = discount_factor
        self.transitions = []
                    
    def reset(self):
        
        BaseAgent.reset(self)
        
        # if set to decay, decay exploration rate according to predefined decay function
        # don't decay during planning, since it is only look ahead
        if self.epsilon_decay:
            self.exploration_rate = self.epsilon_decay_func(self.exploration_rate)
            
    def act(self, state):        
        """ Selects action from policy or according to e-greedy exploration strategy """
    
        # select randomly according to a fixed exploration rate
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_range)
        else:
            action = self.policy[state]
            
        return action
    
    def learn(self, prev_state, prev_action, next_state, reward, done):
        """ Store transitions to learn latter 
        """
        
        # update reward count
        self.total_reward += reward
        # save transitions
        self.transitions.append((prev_state, prev_action, reward))
        if done:
            self.evaluate()
            self.improve()

    def evaluate(self):
        """ Policy evaluation step. Loop backwards through all stored transitions, and update value function according to the reward perceived from the environment.

            Future: qcount is not reseted between episodes. Hence, it decreases vey fast. Verify option with learning rate instead of average
        """

        ret = 0
        for state, action, reward in self.transitions[::-1]:
            ret = reward + self.discount_factor * ret
            error = ret - self.qtable[state][action] 
            self.qcount[state][action] += 1
            self.qtable[state][action] += error/self.qcount[state][action]



    def improve(self):
        """ Policy improvement step. 
            Update policy according to greedy policy w.r.t. the value function.
            
            E-greedy exploration strategy delegated to act function
        """        

        for state in range(self.observation_range):
            possible_actions = list(self.qtable[state].items())
            # shuffle before sorting, to ensure randomness in case of tie
            np.random.shuffle(possible_actions)
            action = sorted(possible_actions, key=lambda x:-x[1])[0][0]
            self.policy[state] = action


class MonteCarloFV(BaseAgent):
    """ Monte Carlo agent, first visit version 
        MonteCarlo defaults with 0 exploration rate, and no decay
    """
    
    def __init__(self,env, qtable_default=0, exploration_rate = 0.1, 
                 epsilon_decay = True, epsilon_decay_func= lambda x: x*.99,
                 discount_factor=0.9):

        BaseAgent.__init__(self, env)

        # init policy, qtable, and qcount (#times a state,action pair is visited)
        self.qtable = {}
        self.qcount = {}
        self.policy = {}
        if self.observation_discrete:
            for obs in range(self.observation_range):
                # initialize policy at random
                self.policy[obs] = env.action_space.sample()
                self.qtable[obs] = {}
                self.qcount[obs] = {}
                for action in range(self.action_range):
                    # set default value higher than 0 for optimistic initialization
                    self.qtable[obs][action] = qtable_default
                    # initialize count of visited state,action to 0
                    self.qcount[obs][action] = 0
        else:
            raise TypeError("Environment not discrete.")
        
        # exploration rate + decay
        self.exploration_rate = exploration_rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_func = epsilon_decay_func

        self.discount_factor = discount_factor
        self.transitions = []
                    
    def reset(self):
        
        BaseAgent.reset(self)
        
        # if set to decay, decay exploration rate according to predefined decay function
        # don't decay during planning, since it is only look ahead
        if self.epsilon_decay:
            self.exploration_rate = self.epsilon_decay_func(self.exploration_rate)
            
    def act(self, state):        
        """ Selects action from policy or according to e-greedy exploration strategy """
    
        # select randomly according to a fixed exploration rate
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_range)
        else:
            action = self.policy[state]
            
        return action
    
    def learn(self, prev_state, prev_action, next_state, reward, done):
        """ Store transitions to learn latter 
        """
        
        # update reward count
        self.total_reward += reward
        # save transitions
        self.transitions.append((prev_state, prev_action, reward))
        if done:
            self.evaluate()
            self.improve()

    def evaluate(self):
        """ Policy evaluation step. Loop backwards through all stored transitions, and update value function according to the reward perceived from the environment.

            Future: qcount is not reseted between episodes. Hence, it decreases very fast. Verify option with learning rate instead of average
        """

        ret = 0
        # loop once through transitions to capture first visits
        # avoid the "unless" loop through past transitions in Sutton's algorithm
        visited_states = set()
        first_visit_indexes = {}
        for idx, (state, action, reward) in enumerate(self.transitions):
            if state not in visited_states:
                first_visit_indexes[state] = idx
                visited_states.add(state)

        # evaluate step for first visit version
        for idx, (state, action, reward) in enumerate(self.transitions[::-1]):
            ret = reward + self.discount_factor * ret
            if idx == first_visit_indexes[state]:
                error = ret - self.qtable[state][action] 
                self.qcount[state][action] += 1
                self.qtable[state][action] += error/self.qcount[state][action]


    def improve(self):
        """ Policy improvement step. 
            Update policy according to greedy policy w.r.t. the value function.
            
            E-greedy exploration strategy delegated to act function
        """        

        for state in range(self.observation_range):
            possible_actions = list(self.qtable[state].items())
            # shuffle before sorting, to ensure randomness in case of tie
            np.random.shuffle(possible_actions)
            action = sorted(possible_actions, key=lambda x:-x[1])[0][0]
            self.policy[state] = action





