import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import json

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class AdvancedAI2048:
    def __init__(self):
        self.actions = ['Up', 'Down', 'Left', 'Right']
        self.learning_rate = 0.005
        self.discount_factor = 0.99
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
        self.memory = deque(maxlen=100000)  # Increased memory size
        self.batch_size = 64
        self.model = NeuralNetwork(16, 512, 4)  # Increased hidden layer size
        self.target_model = NeuralNetwork(16, 512, 4)  # Added target network
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.episode_rewards = []
        self.save_thresholds = [512, 1024, 2048]
        self.saved_states = []
        self.update_target_every = 1000
        self.steps = 0
        self.action_rewards = {action: deque(maxlen=50) for action in self.actions}  # Reward history for each action
        self.low_reward_threshold = 300  # Define low reward threshold
        self.ignore_threshold = 3  # Number of bad performances to stop action temporarily

    def get_state(self, matrix):
        state = np.array(matrix).flatten() / 2048  # Normalize
        return torch.FloatTensor(state)

    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return random.choice(self.actions)  # Random action

        valid_actions = self.get_valid_actions()
        if not valid_actions:
            valid_actions = self.actions  # Fallback if no valid action left

        with torch.no_grad():
            q_values = self.model(state)

        valid_q_values = [(action, q_values[self.actions.index(action)]) for action in valid_actions]
        best_action = max(valid_q_values, key=lambda x: x[1])[0]
        return best_action

    def get_valid_actions(self):
        valid_actions = []
        for action in self.actions:
            if len(self.action_rewards[action]) > 0:
                avg_reward = sum(self.action_rewards[action]) / len(self.action_rewards[action])
                if avg_reward > self.low_reward_threshold:
                    valid_actions.append(action)
            else:
                valid_actions.append(action)
        return valid_actions

    def learn(self, state, action, reward, next_state, done):
        # Calculate adjusted reward with custom logic
        matrix = state.view(4, 4) * 2048  # Denormalize
        previous_matrix = next_state.view(4, 4) * 2048  # Denormalize
        adjusted_reward = self.calculate_reward(matrix.cpu().numpy(), action, previous_matrix.cpu().numpy())

        self.memory.append((state, action, adjusted_reward, next_state, done))
        action_name = self.actions[action]
        self.action_rewards[action_name].append(adjusted_reward)

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor([self.actions.index(self.actions[a]) for a in actions])
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

        loss = self.criterion(current_q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if self.check_for_threshold(next_state):
            self.save_current_state(next_state)

    def calculate_reward(self, matrix, action, previous_matrix):
        reward = 0

        # Penalize if a lot of low-value tiles (2s and 4s) are on the board
        low_tiles_count = np.sum((matrix == 2) | (matrix == 4) | (matrix == 8))
        med_tiles_count = np.sum((matrix == 8) | (matrix == 16) | (matrix == 32))
        high_tile_count = np.sum((matrix == 64) | (matrix == 128) | (matrix == 256))

        # Encourage merging higher tiles
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] != previous_matrix[i][j]:
                    reward += matrix[i][j] * 10  # Stronger reward for higher tile merges

        # Strong reward for keeping the highest tile in the corner
        highest_tile = max(max(row) for row in matrix)
        if matrix[3][0] == highest_tile:  # Focus on bottom-left corner strategy
            reward += 20000  # Boost the reward for achieving this


        # Encourage merging same tiles together in a row/column
        for i in range(len(matrix)):
            for j in range(len(matrix[0]) - 1):
                if matrix[i][j] == matrix[i][j + 1] and matrix[i][j] != 0:
                    reward += 5000  # Strong reward for merging similar tiles
            
            # Add new logic: Reward for combining tiles of value 64 or higher
            for i in range(len(matrix)):
                for j in range(len(matrix[0]) - 1):
                    if matrix[i][j] == matrix[i][j + 1] and matrix[i][j] >= 64:
                        reward += 10000  # Additional reward for merging tiles of 64 or higher

            # Vertical merges
            for i in range(len(matrix) - 1):
                for j in range(len(matrix[0])):
                    if matrix[i][j] == matrix[i + 1][j] and matrix[i][j] != 0:
                        reward += 5000

                    # Add new logic: Reward for combining vertical tiles of value 64 or higher
                    if matrix[i][j] == matrix[i + 1][j] and matrix[i][j] >= 64:
                        reward += 10000  # Additional reward for merging tiles of 64 or higher

        # Penalize moves that don't change the board state
        if np.array_equal(matrix, previous_matrix):
            reward -= 5000  # Heavier penalty for no movement

        if low_tiles_count > 8:  # If more than half the board is filled with 2s and 4s
            reward -= 10000  # Higher penalty for low-value tiles crowding the board
  

        # Reward for keeping empty spaces while maximizing the highest tile
        empty_spaces_reward = self.check_empty_spaces(matrix)
        reward += empty_spaces_reward  # Reward based on empty spaces

        return reward


    def check_empty_spaces(self, matrix):
        empty_spaces = np.sum(matrix == 0)
        return empty_spaces * 1000  # Increased reward multiplier for keeping empty spaces

    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def save_model(self, filepath='model.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate,
            'episode_rewards': self.episode_rewards,
        }, filepath)

    def load_model(self, filepath='model.pth'):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.episode_rewards = checkpoint['episode_rewards']

    def check_for_threshold(self, state):
        matrix = state.view(4, 4) * 2048  # Denormalize
        for threshold in self.save_thresholds:
            if threshold in matrix:
                return True
        return False

    def save_current_state(self, state):
        matrix = state.view(4, 4) * 2048  # Denormalize
        self.saved_states.append(matrix.tolist())
        with open('saved_states.json', 'w') as f:
            json.dump(self.saved_states, f)
