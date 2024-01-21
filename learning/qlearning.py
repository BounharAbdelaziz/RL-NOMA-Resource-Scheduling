import numpy as np
from tqdm import tqdm
from environment.env import Environnement
from matplotlib import pyplot as plt


class QLearningAgent():
    """Class for Q-Learning & Double Q-Learning Algorithm"""

    def __init__(self, environment: Environnement, gamma:float = 0.99, learning_rate: float = 1e-3, initial_q_value: float = -1):
        """
        Class for Q-Learning Algorithm

        """
        
        self.environment = environment
        self.n_states = environment.state_space.n_states
        self.n_actions = environment.action_space.n_actions
        
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        self.Q_matrix = initial_q_value * np.ones((self.n_states, self.n_actions))
        self.initial_value_Q_matrix = initial_q_value
        
        
    def update_Q_matrix(self, reward: float, current_state: int, next_state: int, action: int) -> None:
        """Updates the Q-matrix using the Bellman equation.

        Args:
            reward (float): _description_
            current_state (int): _description_
            next_state (int): _description_
            action (int): _description_
        Returns:
            None
        """
        # The old Q value
        old_Q_value = self.Q_matrix[current_state, action]
        # The new Q value (learned value)
        learned_value = reward + self.discount_factor * np.max(self.Q_matrix[next_state, :])
        # Update the Q-matrix of the current state and action pair
        self.Q_matrix[current_state, action] = (1- self.learning_rate) * old_Q_value + self.learning_rate * learned_value

    def choose_action(self, state_index, epsilon):
        """
        Choose an action following Epsilon Greedy Policy.
        """
        if np.random.random() < epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.Q_matrix[state_index])
    
    def train(self, n_episodes: int = 100, n_time_steps: int = 100, epsilon_decay: float = 0.95, epsilon_min: float = 0.01):
        """
        Train the Q-Learning agent.
        
        Parameters
        ----------
            - n_episodes : Number of Episodes
            - n_time_steps : Maximum number of time steps per episode
            - epsilon : Exploration Rate
            - epsilon_decay : Decay Rate for the Exploration Rate
            - epsilon_min : Minimum Exploration Rate
            - double : Boolean for Double Q-Learning (default = False)
        """
        
        # Exploration Rate Decay Curve
        epsilons = np.ones(int(n_episodes))

        for i in range(n_episodes):
            if i >= 1e3 and (i+1) % 1e2 == 0:
                epsilons[i] = max(epsilons[i-1] * epsilon_decay, epsilon_min)
            elif i > 0:
                epsilons[i] = epsilons[i-1]

        # Learning Rate Decay Curve
        alphas = np.ones(int(n_episodes)) * 1e-3

        for i in range(n_episodes):
            if 0 < i <= 100:
                alphas[i] = alphas[i-1] * 10**0.01
            elif alphas[i-1] > 1e-3:
                alphas[i] = max(alphas[i-1] * 0.999, 1e-3)
            elif i > 0:
                break

        print("[INFO] Q-Learning Training: Process Initiated ... ")

        for episode in tqdm(range(n_episodes)):
            
            # Generate a Random Intial State
            _ = self.environment.reset()

            # Exploration Probabilities
            epsilon = np.random.random()
            percentage_unvisited_states = self.computes_percentage_unvisited_states()
            print(f'Percentage of unvisited states : {percentage_unvisited_states:.2f}%')
            if percentage_unvisited_states == 0:
                print(f'INFO] Q-Learning exploration : All states have been visited ! Setting epsilon to epsilon_min = {epsilon_min} ...')
                epsilon = epsilon_min
            
            reward_episode = []
            
            for time_step in range(n_time_steps):
                # Get the State Index
                current_state = self.environment.state_space.get_state_index()

                # Choose an action following Epsilon Greedy Policy
                best_action = self.choose_action(current_state, epsilon)

                # Update State
                next_state, reward = self.environment.step(best_action)
                reward_episode.append(reward)

                # Update Q(s, a)
                self.update_Q_matrix(reward, current_state, next_state, best_action)

            avg_reward = np.mean(reward_episode[-100:])
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            
            print(f'Episode: {episode+1}/{n_episodes}, Average Reward: {avg_reward}, Epsilon: {epsilon:.2f}, Learning Rate: {alphas[episode]:.2e}')
            print('---------------------------------------------------')

        print("[INFO] Q-Learning Training : Process Completed !")

        plt.plot(avg_reward)
        plt.title("Convergence of Value Iteration Algorithm")
        plt.xlabel("Iteration")
        plt.ylabel("Average Reward")
        plt.savefig("convergence_q_learning.png")
    
    def computes_percentage_unvisited_states(self) -> float:
        """Computes the percentage of unvisited states in the Q-matrix.

        Returns:
            percentage_unvisited_states (float): The percentage of unvisited states in the Q-matrix.
        """
        # compute the number of unvisited states
        number_of_unvisited_states = np.count_nonzero(self.Q_matrix == self.initial_value_Q_matrix)
        # compute the percentage of unvisited states
        percentage_unvisited_states = 100 * number_of_unvisited_states / self.Q_matrix.size
        return percentage_unvisited_states