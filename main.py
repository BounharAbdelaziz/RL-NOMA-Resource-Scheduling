import sys
import numpy as np
# from learning.policy_iteration import PolicyIterAgent
from learning.value_iteration import ValueIterationAgent
from environment.states import State
from environment.actions import Action
from environment.transitions import Transition
from environment.rewards import Reward
from environment.env import Environnement


np.set_printoptions(threshold=sys.maxsize)


if __name__ == '__main__':
    #----------------------------------------------------- Hyperparameters -----------------------------------------------------#
    # Discount Factor
    gamma = 0.99                                
    # Value / Policy Iteration Convergence criterion 
    convergence_threshold = 1e-7
    
    # states initialization
    data_packets = 0
    maximum_number_of_packets = 1
    
    snr_level = 1
    snr_levels_cardinality = 3
    
    maximum_delay = 0
    
    maximum_battery_level = 1
    battery_level = 0
    
    data_arrival_probability = 0.5
    energy_arrival_probability = 0.5
    n_users = 2
    
    #----------------------------------------------------- Environment -----------------------------------------------------#
    
    states = State(data_packets=data_packets, 
                    maximum_number_of_packets=maximum_number_of_packets,
                    snr_level=snr_level, 
                    maximum_delay=maximum_delay, 
                    maximum_battery_level=maximum_battery_level, 
                    battery_level=battery_level,
                    data_arrival_probability=data_arrival_probability,
                    snr_levels_cardinality=snr_levels_cardinality,
                    energy_arrival_probability=energy_arrival_probability,
                    n_users=n_users,
            )
    
    # Actions space initialization
    actions = Action(n_users=n_users)
    # Transitions and Rewards initialization
    transitions = Transition(states=states, actions=actions)
    rewards = Reward(states=states, actions=actions)
    # Environment initialization
    environment = Environnement(states=states, actions=actions, transitions=transitions, rewards=rewards)
    
    #----------------------------------------------------- Agent -----------------------------------------------------#
    
    # Agent
    value_iteration_agent = ValueIterationAgent(environment=environment, 
                                                gamma=gamma, 
                                                convergence_threshold=convergence_threshold)
    
    # Start training
    value_iteration_agent.train()