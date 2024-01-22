import numpy as np
from .states import State
from .actions import Action
from .transitions import Transition
from .rewards import Reward
        

class Environnement():
    """Environnement class for the RL problem"""

    def __init__(self, states: State, actions: Action, transitions:Transition, rewards:Reward):

        self.state_space = states
        self.action_space = actions
        self.transition_model = transitions
        self.reward_model = rewards

    def p(self, state_index, action_index):
        """get the possible next states, their transition probabilities, and the reward"""
        transition = []
        
        reward = self.reward_model.reward_matrix[state_index, action_index]
        
        next_state_probs = self.transition_model.transition_matrix[state_index, action_index, :]
        
        # fill the transition list with the tuples (P[s'], s', r)
        for next_state_index, next_state_prob in enumerate(next_state_probs):
            # optional verification
            if next_state_prob == 0:
                continue
            transition.append((next_state_prob, next_state_index, reward))
        
        return transition
    
    def reset(self):
        self.state_space.initialize()
    
    def step(self, action_index):
        
        action_dictionary = self.action_space.get_action_dictionary(action_index)
        list_actions = list(action_dictionary.values())
        # Execute the action list for the state users
        self.state_space.execute_action(list_actions)
        # Update the state information and transition to a new one
        exceeded_delays = self.state_space.update_state()

        reward = - sum(exceeded_delays)
        next_state_index = self.state_space.get_state_index()

        return next_state_index, reward
