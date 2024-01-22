import numpy as np
from environment.env import Environnement
from matplotlib import pyplot as plt
class ValueIterationAgent():
    
    def __init__(self, environment: Environnement, gamma: float = 0.99, convergence_threshold: float = 1e-5):
        """
        Class for Value Iteration Algorithm

        Parameters
        ----------
            - environment : Model for the State : (User 1: (buffer1, channel 1), User 2 : (buffer2, channel2))
            - gamma : Discount Factor (default = 0.99)
            - convergence_threshold : Convergence Criterion (default = 1e-5)
        """
        
        self.env = environment
        self.gamma = gamma
        self.convergence_threshold = convergence_threshold
        
    def train(self,):
        """
        Train the Value Iteration Agent.
        """
        print("Value Iteration Training : Process Initiated ... ")
        V, policy, delta_array = self.value_iteration()
        print("Value Iteration Training : Process Completed !")
        print(f'Value function: {V}')
        print(f'Policy: {policy}')
        
        plt.plot(delta_array)
        plt.title("Convergence of Value Iteration Algorithm")
        plt.xlabel("Iteration")
        plt.ylabel("Delta")
        plt.savefig("./figures/convergence_value_iteration.png")
        
        return V, policy
    
    def value_iteration(self,):
        """
        Value Iteration Algorithm.
            
        Returns:
            A tuple (V, policy) of the optimal value function and the optimal policy.
        """
        
        # array to save the delta between each iteration
        delta_array = []
        
        # state and action space sizes
        n_states = self.env.state_space.n_states
        n_actions = self.env.action_space.n_actions
        
        def one_step_lookahead(state_index, V):
            """
            Function to calculate the value for all actions [A] in given state s.
            
            Args:
                state_index: The state to consider (int)
                V: The value to use as an estimator, Vector of length env.nS
            
            Returns:
                A vector of length env.nA containing the expected value of each action.
            """
            A = np.zeros(n_actions)
            # transitions = self.env.p(state_index, action_index)
            for action_index in range(n_actions):
                # Since env.p returns a list of tuples (transition probability, next state, reward)
                transitions = self.env.p(state_index, action_index)
                for (next_state_prob, next_state_index, reward) in transitions:
                    A[action_index] += next_state_prob * (reward + self.gamma * V[next_state_index])
                
            return A

        #Initialization value function
        V = np.ones(n_states)
        policy = np.zeros([n_states, n_actions])

        #While not optimal value function/optimal policy
        while True:
            # Stopping condition
            delta = 0
            
            # Update each state
            for state_index in range(n_states):
                # Do a one-step lookahead to find the best action 
                A = one_step_lookahead(state_index, V)
                # choose the action that maximizes the state-value function
                best_action_value = np.max(A)
                # Calculate delta across all states 
                delta = max(delta, np.abs(best_action_value - V[state_index]))
                delta_array.append(delta)
                # Update the value function: Bellman optimality eqn 
                V[state_index] = best_action_value
                
            # Check for convergence 
            if delta < self.convergence_threshold:
                break
        
        # Create a deterministic policy using the optimal value function
        for state_index in range(n_states):
            # One step lookahead to find the best action for this state
            A = one_step_lookahead(state_index, V)
            best_action = np.argmax(A)
            # Always take the best action
            policy[state_index, best_action] = 1.0
        return V, policy, delta_array