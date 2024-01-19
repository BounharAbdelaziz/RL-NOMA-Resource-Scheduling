import math
import numpy as np
from .states import State
from .actions import Action


class Reward():
    def __init__(self, states: State, actions: Action):
        self.states = states
        self.actions = actions
        self.reward_matrix_dims = (self.states.n_states, self.action.n_actions)
        self.reward_matrix = np.zeros(self.reward_matrix_dims)
        
        self.reward_matrix = self.__compute_rewards()

    def __compute_rewards(self):
        
        for s in range(self.reward_matrix_dims[0]):
            # current_state = [(data_packets, snr_level, battery_level), (...), ...]
            current_state = self.states.get_state_from_index(state_index=s)
            for a in range(self.reward_matrix_dims[1]):
                action_dictionary = self.actions.get_action_dictionary(a)
                noma_users_snr = []
                # Give a reward of (-oo) to actions that are impossible to execute
                for l, user_current_state in enumerate(current_state):
                    # the action number of packets to execute for the user are not available in the user buffer
                    if action_dictionary[f'user_{l+1}'] > user_current_state[0]:
                        self.reward_matrix[s, a] = - math.inf
                    
                    if action_dictionary[f'action_{l+1}'] == 'communicate':
                        # log the communicating users to verify multiple communications
                        noma_users_snr.append(user_current_state[1])
                        # the user action dictates that the user communicates, but its SNR does not allow it 
                        if user_current_state[1] == 0:
                            self.reward_matrix[s, a] = - math.inf
                    
                    # the user battery is not sufficient to execute the action number of packets
                    if action_dictionary[f'user_{l+1}'] > user_current_state[2]:
                        self.reward_matrix[s, a] = - math.inf
                
                # the joint action dictates multiple communications
                if len(noma_users_snr) > 1:
                    for user_snr in noma_users_snr:
                        # one of the user SNRs does not permit it
                        if user_snr < len(noma_users_snr):
                            self.reward_matrix[s, a] = - math.inf

                if not math.isinf(self.reward_matrix[s, a]):
                    for l, user_current_state in enumerate(current_state):
                        # if the number of user packets to execute is smaller than the number of packets in the user buffer
                        # a reward of -1 is associated with each packet that stays in the user buffer
                        if user_current_state[0] > action_dictionary[f'user_{l+1}']:
                            self.reward_matrix[s, a] += - (user_current_state[0] - action_dictionary[f'user_{l+1}'])
                