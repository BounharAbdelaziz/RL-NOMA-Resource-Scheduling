import sys
import numpy as np
from learning.policy_iteration import PolicyIterationAgent
from learning.value_iteration import ValueIterationAgent
from learning.qlearning import QLearningAgent
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
        convergence_threshold = 1e-5

        # states initialization
        data_packets = 0
        maximum_number_of_packets = 1

        snr_level = 1
        snr_levels_cardinality = 3

        maximum_delay = 1

        maximum_battery_level = 2
        battery_level = 0

        data_arrival_probability = 0.85
        energy_arrival_probability = 0.25
        n_users = 2

        agent_to_use = str(input("Which agent do you want to use ? (V: value_iteration / P: policy_iteration/ Q: q_learning): "))

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
                        unavailable_action_penalty=2,
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
        if agent_to_use.upper() == 'P':
                agent = PolicyIterationAgent(environment=environment,
                                                gamma=gamma,
                                                convergence_threshold=convergence_threshold)
        elif agent_to_use.upper() == 'V':
                agent = ValueIterationAgent(environment=environment, 
                                                gamma=gamma, 
                                                convergence_threshold=convergence_threshold)
        elif agent_to_use.upper() == 'Q':
                agent = QLearningAgent(environment=environment, 
                                                gamma=gamma, 
                                                learning_rate=1e-2)     
        else:
                raise NotImplementedError(f"Agent {agent_to_use} is not implemented yet ! Please choose between P (Policy Iteration) and V (Value Iteration)")
        # Start training
        agent.train()