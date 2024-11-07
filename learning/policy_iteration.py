import numpy as np
from environment.env import Environnement
from matplotlib import pyplot as plt
class PolicyIterationAgent():
    
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

    def policy_evaluation(self, policy):
            """
            Evaluate a policy given an environment and a full description of the environment's dynamics.\n",
                       
            Returns:
                Vector of length n_states representing the value function.
            """
            
            # array to save the delta between each iteration
            delta_array = []
            
            # state space size
            n_states = self.env.state_space.n_states
            
            # Start with a random (all 0) value function,
            V = np.zeros(n_states)
            while True:
                delta = 0
                # For each state, perform a "full backup"
                for state_index in range(n_states):
                    v = 0
                    # Look at the possible next actions
                    for action_index, action_prob in enumerate(policy[state_index]):
                        # For each action, look at the possible next states...
                        transitions = self.env.p(state_index, action_index)
                        for (next_state_prob, next_state_index, reward) in transitions:
                            # Calculate the expected value
                            v += action_prob * next_state_prob * (reward + self.gamma * V[next_state_index])
                    # How much our value function changed (across any states)
                    delta = max(delta, np.abs(v - V[state_index]))
                    delta_array.append(delta)
                    V[state_index] = v
                # Stop evaluating once our value function change is below a threshold\n",
                if delta < self.convergence_threshold:
                    break
            return np.array(V), delta_array
        

    def policy_iteration(self):
        """
        Policy Improvement Algorithm.
        
        Returns:
            A tuple (policy, V) of the optimal value function and the optimal policy.
            policy is the optimal policy, a matrix of shape [S, A] where each state s
            contains a valid probability distribution over actions.
            V is the value function for the optimal policy.
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
                V: The value to use as an estimator, Vector of length n_states
            
            Returns:
                A vector of length n_actions containing the expected value of each action.
            """
            A = np.zeros(n_actions)
            for action_index in range(n_actions):
                transitions = self.env.p(state_index, action_index)
                for (next_state_prob, next_state_index, reward) in transitions:
                    A[action_index] = A[action_index] + next_state_prob * (reward + self.gamma * V[next_state_index])
            return A

        # Start with a random policy
        policy = np.ones([n_states, n_actions]) / n_actions
        print(f'[INFO] Starting Policy: {policy}')
        
        # While not optimal policy
        while True:
            # Evaluate the current policy
            V, delta_array_policy_evaluation = self.policy_evaluation(policy)
            
            delta_array = delta_array + delta_array_policy_evaluation        
            # Will be set to false if we make any changes to the policy
            policy_stable = True

            # For eac state, do the following
            for state_index in range(n_states):
                # Compute the best action we would take under the current policy
                chosen_a = np.argmax(policy[state_index])
                
                # Do a one-step lookahead to find the best action 
                action_values = one_step_lookahead(state_index, V)
                best_action = np.argmax(action_values)

                # Greedily update the policy
                if chosen_a != best_action:
                    policy_stable = False
                policy[state_index] = np.eye(n_actions)[best_action]
                
            # If the policy is stable, then this is the optimal one
            if policy_stable:
                return V, policy, delta_array
            
    def train(self,):
        """
        Train the Policy Iteration Agent.
        """
        print("[INFO] Policy Iteration Training : Process Initiated ... ")
        V, policy, delta_array = self.policy_iteration()
        print("[INFO] Policy Iteration Training : Process Completed !")
        print(f'[INFO] Policy: {policy}')
        
        plt.plot(delta_array)
        plt.title("Convergence of Policy Iteration Algorithm")
        plt.xlabel("Iteration")
        plt.ylabel("Delta")
        plt.savefig("./figures/convergence_policy_iteration.png")
        
        return V, policy