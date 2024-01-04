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
        
        
class REINFORCEAgent:
    def __init__(self, n_actions, gamma, learning_rate):
        self.n_actions = n_actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        #self.policy_parameters = np.random.rand(4, 7, self.n_actions) * 0.001
        self.policy_parameters = np.zeros((4, 7, n_actions))  

    def select_action(self, state):
        # Softmax policy
        probabilities = self.softmax(self.policy_parameters[state])
        return np.random.choice(self.n_actions, p=probabilities)

    def update_policy_parameters(self, episode_states, episode_actions, episode_rewards):
        returns = self.calculate_returns(episode_rewards)
        for t in range(len(episode_states)):
            state = episode_states[t]
            action = episode_actions[t]
            gradient = self.calculate_gradient(state, action)
            self.policy_parameters[state] += self.learning_rate * gradient * returns[t]

    def calculate_gradient(self, state, action):
        probabilities = self.softmax(self.policy_parameters[state])
        indicator = [0, 0, 0, 0];
        indicator[action] = 1
        return indicator - probabilities

    def calculate_returns(self, episode_rewards):
        returns = []
        return_so_far = 0
        for r in reversed(episode_rewards):
            return_so_far = r + self.gamma * return_so_far
            returns.append(return_so_far)
        returns.reverse()
        return returns

    def softmax(self, x):
        exp_values = np.exp(x - np.mean(x))
        return exp_values / np.sum(exp_values)

    def train(self, num_episodes, environment_simulator, monitor_episodes=10):
        episode_rewards_history = []
        policy_parameters_history = []
        episode_10_rewards = []
        for episode in range(1, num_episodes + 1):
            episode_states, episode_actions, episode_rewards = self.run_episode(environment_simulator)
            self.update_policy_parameters(episode_states, episode_actions, episode_rewards)
            episode_10_rewards.append(episode_rewards[-1])
            if episode % monitor_episodes == 0:
                avg_reward = np.mean(episode_10_rewards)
                episode_rewards_history.append(avg_reward)
                policy_parameters_history.append(np.copy(self.policy_parameters))
                episode_10_rewards = []
                
        print(f"Training completed for {num_episodes} episodes.")
        return episode_rewards_history, policy_parameters_history

    def run_episode(self, environment_simulator):
        state = (1, 1)  
        episode_states = []
        episode_actions = []
        episode_rewards = []

        while True:
            action = self.select_action(state)
            reward, next_state, terminal = environment_simulator.take_action(action)

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state

            if terminal:
                break

        return episode_states, episode_actions, episode_rewards

num_episodes = 10000
learning_rate = 0.25  

reinforce_agent = REINFORCEAgent(n_actions=4, gamma=0.9, learning_rate=learning_rate)
episode_rewards_history, policy_parameters_history = reinforce_agent.train(num_episodes, EnvironmentSimulator())

plt.figure(figsize=(10, 5))
plt.plot(range(10, num_episodes + 1, 10), episode_rewards_history, marker='o')
plt.title('Prosek na 10 epizoda')
plt.xlabel('Broj epizoda')
plt.ylabel('Prosecna nagrada')
plt.show()

plt.figure(figsize=(10,5))
policy_parameters_history = np.array(policy_parameters_history)
for action in range(4):
    plt.plot(range(10, num_episodes + 1, 10), policy_parameters_history[:, 1, 1, action], label=f'Akcija {action}')
plt.title('Parametri politike u (1,1)')
plt.xlabel('Broj epizoda')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
policy_parameters_history = np.array(policy_parameters_history)
for action in range(4):
    plt.plot(range(10, num_episodes + 1, 10), policy_parameters_history[:, 1, 5, action], label=f'Akcija {action}')
plt.title('Parametri politike u (1,5)')
plt.xlabel('Broj epizoda')
plt.legend()
plt.show()