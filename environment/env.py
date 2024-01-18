import inspect
import numpy as np
from .states import State
from .actions import Action
        

class Environnement():
    """Environnement class for the RL problem"""

    def __init__(self, states: State, actions: Action, transitions, rewards):

        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
    
    def reset(self):
        raise NotImplementedError("The reset method is not implemented !")
    
    def step():
        raise NotImplementedError("The step method is not implemented !")