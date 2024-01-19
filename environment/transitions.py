import numpy as np
from .states import State
from .actions import Action


class Transition():
    def __init__(self, states: State, actions: Action):
        self.states = states
        self.actions = actions
        self.transition_matrix_dims = (self.states.n_states, self.action.n_actions, self.states.n_states)
        self.transition_matrix = np.ones(self.transition_matrix_dims)
        
        self.transition_matrix = self.__compute_transitions()

    def __compute_transitions(self):
        
        for s in range(self.transition_matrix_dims[0]):
            # current_state = [(data_packets, snr_level, battery_level), (...), ...]
            current_state = self.states.get_state_from_index(state_index=s)
            for a in range(self.transition_matrix_dims[1]):
                action_dictionary = self.actions.get_action_dictionary(a)
                noma_users_snr = []

                # Verify if the action can be performed, exclude transition if :
                for l, user_current_state in enumerate(current_state):
                    # the action number of packets to execute for the user are not available in the user buffer
                    if action_dictionary[f'user_{l+1}'] > user_current_state[0]:
                        self.transition_matrix[s, a, :] = 0
                    
                    if action_dictionary[f'action_{l+1}'] == 'communicate':
                        # log the communicating users to verify multiple communications
                        noma_users_snr.append(user_current_state[1])
                        # the user action dictates that the user communicates, but its SNR does not allow it 
                        if user_current_state[1] == 0:
                            self.transition_matrix[s, a, :] = 0
                    
                    # the user battery is not sufficient to execute the action number of packets
                    if action_dictionary[f'user_{l+1}'] > user_current_state[2]:
                        self.transition_matrix[s, a, :] = 0
                
                # the joint action dictates multiple communications
                if len(noma_users_snr) > 1:
                    for user_snr in noma_users_snr:
                        # one of the user SNRs does not permit it
                        if user_snr < len(noma_users_snr):
                            self.transition_matrix[s, a, :] = 0
                
                if self.transition_matrix[s, a, :].all() != 0:
                    for p in range(self.transition_matrix_dims[2]):
                        next_state = self.states.get_state_from_index(state_index=p)

                        # Verify if the action can be performed, exclude transition if :
                        for l, (user_current_state, user_next_state) in enumerate(current_state, next_state):
                            
                            if action_dictionary[f'action_{l+1}'] == 'idle':
                                # The user battery level in the next state is less then the user battery level in the current state
                                if user_next_state[2] < user_current_state[2]:
                                    self.transition_matrix[s, a, p] = 0
                            
                            elif action_dictionary[f'action_{l+1}'] == 'communicate':
                                energy_to_execute = action_dictionary[f'user_{l+1}'] # Each packet executed consumed 1 energy unit
                                # The user battery level in the next state is less then the user battery level in the current state after taking the action
                                if user_next_state[2] < (user_current_state[2] - energy_to_execute):
                                    self.transition_matrix[s, a, p] = 0
                        
                        if self.transition_matrix[s, a, p] != 0:
                            # Compute the transition probability from state `s` to state `p` after taking action `a`
                            for _ in range(self.states.n_users):
                                # Buffer transition probability
                                self.transition_matrix[s, a, p] *= (1 / (self.maximum_number_of_packets + 1))
                                # SNR transition probability
                                self.transition_matrix[s, a, p] *= (1 / self.snr_levels_cardinality)
                                # Battery transition probability
                                self.transition_matrix[s, a, p] *= (1 / (self.maximum_battery_level + 1))
