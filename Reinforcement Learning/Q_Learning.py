import random
import numpy as np
import matplotlib.pyplot as plt

class EnvironmentSimulator:
    def __init__(self):
        self.map = [
            ['W', 'W', 'W', 'W', 'W', 'W', 'W'],
            ['W', 'S', 'M', 'M', 'M', 'M', 'W'],
            ['W', 'B', 'W', 'B', 'W', 'G', 'W'],
            ['W', 'W', 'W', 'W', 'W', 'W', 'W']
        ]
        
        self.agent_position = (1, 1)
        self.forward_prob = 0.6
        self.sideways_prob = 0.2

    def take_action(self, action):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        outcome = random.choices(['forward', 'sideways'], weights=[self.forward_prob, self.sideways_prob * 2])[0]
        
        if outcome == 'forward':
            next_position = (self.agent_position[0] + directions[action][0], 
                             self.agent_position[1] + directions[action][1])
        else:
            sideways_direction = random.choice([(directions[action][1], directions[action][0]),
                                                 (-directions[action][1], -directions[action][0])])
            next_position = (self.agent_position[0] + sideways_direction[0], 
                             self.agent_position[1] + sideways_direction[1])

        if self.is_valid_position(next_position):
            self.agent_position = next_position

        reward, terminal = self.get_reward_and_terminal()

        return reward, self.agent_position, terminal

    def is_valid_position(self, position):
        row, col = position
        return 0 < row < len(self.map) - 1 and 0 < col < len(self.map[0]) - 1 and self.map[row][col] != 'W'

    def get_reward_and_terminal(self):
        current_cell = self.map[self.agent_position[0]][self.agent_position[1]]

        if current_cell == 'B':
            return -1, True  
        elif current_cell == 'G':
            return 3, True  
        else:
            return 0, False


class QLearningAgent:
    def __init__(self, n_actions, epsilon, gamma, Q, learning_rate_strategy='constant'):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate_strategy = learning_rate_strategy
        self.Q = Q  
        self.episode = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    

    def update_q_values(self, state, action, next_state, reward):
        alpha = self.calculate_learning_rate()
        next_max_q = np.max(self.Q[next_state])
        self.Q[state][action] += alpha * (reward + self.gamma * next_max_q - self.Q[state][action])

    def calculate_learning_rate(self):
        if self.learning_rate_strategy == 'constant':
            return 0.0005
        elif self.learning_rate_strategy == 'variable':
            return np.log(self.episode + 1) / (self.episode + 1)

    def train(self, num_episodes, environment_simulator):
        v_values_per_episode = []
        for episode in range(1, num_episodes + 1):
            state = (1, 1)  
            total_reward = 0

            while True:
                action = self.select_action(state)
                reward, next_state, terminal = environment_simulator.take_action(action)
                total_reward += reward
                self.agent_position = next_state
                self.update_q_values(state, action, next_state, reward)

                state = next_state

                if terminal:
                    break

            self.episode += 1
            
            v_value_start_state = np.max(self.Q[1,1])
            v_values_per_episode.append(v_value_start_state)
            print(f"Ep {episode}, V-val: {np.max(self.Q[1,1])}")

        return total_reward / num_episodes, v_values_per_episode

    def test_agent(self, num_episodes, environment_simulator):
        total_rewards = []

        for _ in range(num_episodes):
            state = (1, 1)  
            episode_reward = 0
            
            while True:
                action = agent.select_action(state)
                reward, next_state, terminal = environment_simulator.take_action(action)
                episode_reward += reward

                state = next_state

                if terminal:
                    break

            total_rewards.append(episode_reward)
        
        return total_rewards


num_episodes = 10000
gamma_values = [0.9, 0.999]
v_vals_gamma = []

for gamma in gamma_values:
    print(f"\nTraining with gamma={gamma}:\n")
    Q = np.zeros((4, 7, 4))
    agent = QLearningAgent(n_actions=4, epsilon=0.9, gamma=gamma, Q = Q, learning_rate_strategy='variable')
    avg_reward, v_vals = agent.train(num_episodes, EnvironmentSimulator())
    v_vals_gamma.append(v_vals)
    
plt.figure(figsize=(8,5))
for i in range(len(gamma_values)):
    plt.plot(v_vals_gamma[i], label = f"{gamma_values[i]}")
plt.legend()
plt.title('V vrednosti')
plt.show()
#%%
print(agent.Q[1,2])

num_test_episodes = 10
reward_test = agent.test_agent(num_test_episodes, EnvironmentSimulator())
avg_reward_test = np.mean(reward_test)
print(f"Rewards for {num_test_episodes} test episodes: {reward_test}")
print(f"Avg: {avg_reward_test}")




