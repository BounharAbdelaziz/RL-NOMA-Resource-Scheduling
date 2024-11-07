import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from environment.env import Environnement
from matplotlib import pyplot as plt

class DQN(nn.Module):
    """Class for Deep Q-Learning Network"""

    def __init__(self, input_dims=1, n_actions=4, learning_rate=1e-3, params_list=[32, 32], loss_fct='huber'):
        super(DQN, self).__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.layers = []

        for i in range(len(params_list)-1): # until layer before last
            # feed-forward layers
            if i == 0:
                self.layers.append(nn.Linear(input_dims, params_list[i+1]))
            else:
                self.layers.append(params_list[i], params_list[i+1])

            # ReLU non-linearity
            self.layers.append(nn.ReLU())

        # do not add ReLU for the last layer
        self.layers.append(nn.Linear(params_list[i+1], n_actions))

        self.network = nn.Sequential(*self.layers).to(self.device)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        if loss_fct.lower() == 'huber':
            # To avoid large gradients as is the case with MSELoss
            self.loss = nn.HuberLoss(reduction='mean', delta=5.0).to(self.device)
        elif loss_fct.lower() == 'mse':
            self.loss = nn.MSELoss(reduction='mean').to(self.device)
        else:
            raise Exception("Loss Function not available ! Choose between 'mse' or 'huber' !")    
        
    def forward(self, state):
        state_action_values = self.network(state)
        return state_action_values
    
class ReplayBuffer():
    """Class for Experience Replay Buffer"""

    def __init__(self, memory_size, batch_size, state_input_shape, device='cpu'):
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory_counter = 0
        # Initialize the replay buffer tensors
        self.current_state_memory = torch.zeros((self.memory_size, state_input_shape), dtype=torch.float32).to(device)
        self.next_state_memory = torch.zeros((self.memory_size, state_input_shape), dtype=torch.float32).to(device)
        self.action_memory = torch.zeros(self.memory_size, dtype=torch.int64).to(device)
        self.reward_memory = torch.zeros(self.memory_size, dtype=torch.float32).to(device)

    def store_transition(self, current_state, action, reward, next_state):
        # Update the index, wrapping around when it exceeds the memory size
        index = self.memory_counter % self.memory_size
        # Store the transitions
        self.current_state_memory[index] = current_state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        # Increment the counter
        self.memory_counter += 1

    def sample_buffer(self):

        max_value = min(self.memory_counter, self.memory_size)

        batch_indices = np.random.choice(max_value, self.batch_size, replace=False)

        current_state_batch = self.current_state_memory[batch_indices]
        next_state_batch = self.next_state_memory[batch_indices]
        action_batch = self.action_memory[batch_indices]
        reward_batch = self.reward_memory[batch_indices]

        return current_state_batch, action_batch, reward_batch, next_state_batch


class DeepQLearningAgent():
    """Class for Deep Q-Learning Algorithm"""

    def __init__(self, 
                    environment: Environnement, 
                    gamma:float = 0.99, 
                    learning_rate: float = 1e-2, 
                    params_list=[32, 32], 
                    replay_buffer_memory_size=32*10, # to get 10 batches
                    batch_size=32, 
                    loss_fct='mse',
                    epsilon_min=0.01,
                    n_epochs=100, 
                    n_time_steps=1000, 
                    freq_update_target=5,
                    epsilon_decay=0.999,
                ):

        self.environment = environment
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.params_list = params_list
        self.replay_buffer_memory_size = replay_buffer_memory_size
        self.batch_size = batch_size

        self.input_dims = 1 # as we are just inputting the state index
        self.n_states = self.environment.state_space.n_states
        self.n_actions = self.environment.action_space.n_actions

        self.evaluation_q_network = DQN(input_dims=self.input_dims, n_actions=self.n_actions, learning_rate=learning_rate, params_list=params_list, loss_fct=loss_fct)
        self.target_q_network = DQN(input_dims=self.input_dims, n_actions=self.n_actions, learning_rate=learning_rate, params_list=params_list, loss_fct=loss_fct)
        
        # the target network is initialized with the same weights as the evaluation network
        self.target_q_network.load_state_dict(self.evaluation_q_network.state_dict())

        self.replay_buffer = ReplayBuffer(replay_buffer_memory_size, batch_size, self.input_dims, device=self.evaluation_q_network.device)

        self.policy = np.zeros((self.n_states, self.n_actions))

        self.n_epochs = n_epochs
        self.n_time_steps = n_time_steps
        self.freq_update_target = freq_update_target
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state_index, epsilon):
        """Choose an action following Epsilon Greedy Policy."""
        if np.random.random() > epsilon:
            state = torch.tensor(state_index, dtype=torch.float32).reshape(-1, 1).to(self.evaluation_q_network.device)
            state_action_values = self.evaluation_q_network(state)
            best_action = torch.argmax(state_action_values).item()
            return best_action
        else:
            random_action = torch.tensor(np.random.choice(self.n_actions), dtype=torch.int64).unsqueeze(0).to(self.evaluation_q_network.device)
            return random_action

    def train(self):

        losses = []

        # Exploration Rate Curve if needed
        # epsilons = np.ones(int(self.n_epochs))

        # for i in range(self.n_epochs):
        #     epsilons[i] = max(epsilons[i-1] * self.epsilon_min**(2/self.n_epochs), self.epsilon_min)

        print(f"[INFO] Initialize Training for DQN Network on Device : {self.target_q_network.device}")

        epsilon = 1
        epsilons = []

        for epoch in tqdm(range(self.n_epochs)):
            
            epsilons.append(epsilon)

            if (epoch+1) % self.freq_update_target == 0:
                self.target_q_network.load_state_dict(self.evaluation_q_network.state_dict())

            # epsilon = epsilons[epoch]
            loss_value = 0

            # Generate a Random Initial State
            _ = self.environment.reset()

            for _ in range(self.n_time_steps):

                # <--- Collect Experience and store them in the Replay Buffer --->
                
                # get the current state index
                current_state = self.environment.state_space.get_state_index()
                
                # Choose an action following Epsilon Greedy Policy
                action = self.choose_action(current_state, epsilon)

                # Update State
                next_state, reward = self.environment.step(action)

                # Store the transition in the Replay Buffer
                self.replay_buffer.store_transition(current_state, action, reward, next_state)
                if self.replay_buffer.memory_counter >= self.replay_buffer.batch_size:
                    # Sample a batch from the Replay Buffer
                    current_state_batch, action_batch, reward_batch, next_state_batch = self.replay_buffer.sample_buffer()

                    # Start the training process
                    loss_value = self.learn(current_state_batch, action_batch, reward_batch, next_state_batch)
            
            losses.append(loss_value)
            avg_loss = np.mean(losses)
            print(f'[INFO] Last loss: {loss_value}')
            print(f'[INFO] Average loss: {avg_loss}')
            print(f'[INFO] epsilon: {epsilon}')

            # update the epsilon value
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

        print(f'[INFO] Average loss: {avg_loss}')
        print("[INFO] Deep Q-Learning Training : Process Completed !")
        
        # Extract policy
        for s in range(self.n_states):
            state = torch.tensor(s, dtype=torch.float32).reshape(-1, 1).to(self.evaluation_q_network.device)
            best_action_index = self.evaluation_q_network(state).argmax().item()
            self.policy[s, best_action_index] = 1
        print(f"Policy : {self.policy}")

        plt.plot(losses, 'b', label='loss')
        # plt.plot(epsilons, 'r', label='Exploration Rate')
        plt.title("Convergence of DQN Algorithm")
        plt.xlabel("Epoch")
        plt.ylabel("Average loss")
        plt.savefig("./figures/convergence_deep_q_learning.png")

        return losses
    
    
    def learn(self, current_state_batch, action_batch, reward_batch, next_state_batch):
        """Train the Deep Q-Learning Network."""
        self.evaluation_q_network.train()
        self.target_q_network.eval()

        # Get Q-values for the current state-action pairs
        eval_net_state_action_values = self.evaluation_q_network(current_state_batch)
        eval_net_state_action_values = eval_net_state_action_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Get Q-values for the next state
        next_state_action_values = self.target_q_network(next_state_batch)

        # Get the maximum Q-value for each next state
        max_next_state_action_values = next_state_action_values.max(1)[0]

        # Calculate the target Q-values using the Bellman equation
        target_net_state_action_value = reward_batch + self.gamma * max_next_state_action_values

        # Calculate the loss between the predicted and target Q-values
        loss = self.evaluation_q_network.loss(eval_net_state_action_values, target_net_state_action_value)

        # Backpropagation
        self.evaluation_q_network.optimizer.zero_grad()
        loss.backward()
        self.evaluation_q_network.optimizer.step()

        return loss.item()

         
      