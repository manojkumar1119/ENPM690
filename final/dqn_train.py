#Importing the libraries
#pygame to display
import pygame
import random
import numpy as np
import math
#using pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
#to plot using matplotlib
import matplotlib.pyplot as plt

# Defining map parameters height and width
WIDTH, HEIGHT = 500, 500
CELL_SIZE     = 25
NUM_CELLS_X   = WIDTH // CELL_SIZE
NUM_CELLS_Y   = HEIGHT // CELL_SIZE

#defining the color for display
#Agent            ---> Blue color
#Goal             ---> Green color
#Static obstacle  ---> Grey color
#Dynamic obstacle ---> Orange color
#Ray sensor line  ---> Light Grey color

COLOR_AGENT            = (0, 0, 255)
COLOR_GOAL             = (0, 255, 0)
COLOR_STATIC_OBSTACLE  = (64, 64, 64)
COLOR_DYNAMIC_OBSTACLE = (255, 165, 0)
COLOR_SENSOR_RAY       = (128, 128, 128)

#Pygame init call
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Map")
clock = pygame.time.Clock()

# DQN Parameters setting up
REPLAY_BUFFER_SIZE = 10000
EXPLORATION_RATE   = 1.0
EXPLORATION_MIN    = 0.1

LEARNING_RATE      = 0.001
DISCOUNT_FACTOR    = 0.99
EXPLORATION_DECAY  = 0.995
BATCH_SIZE         = 32

#defining the model
class QNetwork(nn.Module):
    #init function
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    #architecture of nn
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#setting up the agent
class DeepQLearningAgent:
    def __init__(self, x, y, learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR, exploration_rate=EXPLORATION_RATE, sensor_range=2, num_sensors=8):
        self.x, self.y = x, y
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.sensor_range = sensor_range
        self.num_sensors = num_sensors
        self.sensor_angles = [i * (360 // num_sensors) for i in range(num_sensors)]
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # [Down, right,up, left] -- >Actions

        #set input and output sizes
        input_dim = num_sensors + 2  
        output_dim = len(self.actions)

        #creatingnetworking and setting up the optimz
        self.q_network = QNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.criterion = nn.MSELoss()

    #Function to get the states from map
    def get_state(self, environment):
        state = [self.sensor_read(environment, math.radians(angle)) for angle in self.sensor_angles]
        rel_pos_to_goal = (environment.goal[0] - self.x, environment.goal[1] - self.y)
        return tuple(state + list(rel_pos_to_goal))

    #Reading ray sensor values
    def sensor_read(self, environment, angle):
        dx, dy = math.cos(angle), math.sin(angle)
        for step in range(1, self.sensor_range + 1):
            check_x, check_y = int(self.x + dx * step), int(self.y + dy * step)
            if (check_x, check_y) in environment.obstacles or (check_x, check_y) in environment.dynamic_obstacles:
                return step
        return self.sensor_range

    #selecting 4 action sets
    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return random.choice(range(len(self.actions)))
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # Get Q-Vaues
        q_values = self.q_network(state_tensor)
        # Return action with max Qvalue
        return torch.argmax(q_values).item()

    #q value updating
    def update_q_value(self, prev_state, action_index, reward, next_state):
        # Add the experience to Replay buffer
        self.replay_buffer.append((prev_state, action_index, reward, next_state))
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)

        # changing to tensors
        state_tensor      = torch.tensor(states, dtype=torch.float32)
        action_tensor     = torch.tensor(actions, dtype=torch.long)
        reward_tensor     = torch.tensor(rewards, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_states, dtype=torch.float32)

        # Get the current QValues
        current_q_values = self.q_network(state_tensor)
        current_q_values = current_q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        # Geting target Q-values
        next_q_values = self.q_network(next_state_tensor).detach()
        max_next_q_values = next_q_values.max(1)[0]
        target_q_values = reward_tensor + (self.discount_factor * max_next_q_values)
        #Loss
        loss = self.criterion(current_q_values, target_q_values)
        # Updating the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    #function to move the agent w.r.t action sets
    def move(self, action_index):
        action = self.actions[action_index]
        next_x, next_y = self.x + action[0], self.y + action[1]
        if (next_x, next_y) not in environment.obstacles and (next_x, next_y) not in environment.dynamic_obstacles:
            self.x, self.y = max(0, min(NUM_CELLS_X - 1, next_x)), max(0, min(NUM_CELLS_Y - 1, next_y))

    #pygame for display
    def draw(self, screen):
        pygame.draw.rect(screen, COLOR_AGENT, (self.x * CELL_SIZE, self.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        for angle in self.sensor_angles:
            radians = math.radians(angle)
            end_x = self.x * CELL_SIZE + CELL_SIZE // 2 + math.cos(radians) * self.sensor_range * CELL_SIZE
            end_y = self.y * CELL_SIZE + CELL_SIZE // 2 + math.sin(radians) * self.sensor_range * CELL_SIZE
            pygame.draw.line(screen, COLOR_SENSOR_RAY, (self.x * CELL_SIZE + CELL_SIZE // 2, self.y * CELL_SIZE + CELL_SIZE // 2), (end_x, end_y), 1)

#defining the environment params
class Environment:
    def __init__(self):
        self.agent = DeepQLearningAgent(10, 10)
        #static obstacle setting
        self.obstacles = [
            (1, 1),
            (10, 1),
            (18, 18),
            (18, 1),
            (1, 5),
            (5, 5),
            (18, 5),
            (9, 3),
            (15, 7 ),
            (17, 15),
            (1, 5  ),
            (1, 20 ),
            (3, 10 ),
            (16, 10),
            (8, 8  ),
            (1, 17 ),
            (5, 15 ),
            (5, 17 ),
            (11, 19),
            (17, 15)]
        self.dynamic_obstacles = [
            (6,  6),
            (19,19),
            (3, 10),
            (5, 15),
            (15, 12)]
        self.goal = (15, 15)
        self.reset_goal()

    #to chngee goal
    def reset_goal(self):
        while True:
            new_goal = (random.randint(0, NUM_CELLS_X-1), random.randint(0, NUM_CELLS_Y-1))
            if new_goal not in self.obstacles and new_goal not in self.dynamic_obstacles and new_goal != (self.agent.x, self.agent.y):
                self.goal = new_goal
                break

    #function to move the dynamic dynamic obstacles not collidg with static
    def move_dynamic_obstacles(self):
        for i, (x, y) in enumerate(self.dynamic_obstacles):
            if random.random() > 0.2:
                continue
            dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
            new_x, new_y = (x + dx) % NUM_CELLS_X, (y + dy) % NUM_CELLS_Y
            if (new_x, new_y) not in self.obstacles and (new_x, new_y) != self.goal:
                self.dynamic_obstacles[i] = (new_x, new_y)

    #reward function setting up
    def calculate_reward(self, next_state):
        goal_x, goal_y = self.goal
        agent_x, agent_y = next_state
        distance_to_goal = math.sqrt((goal_x - agent_x)**2 + (goal_y - agent_y)**2)
        #100 reward if reasched goal
        if (agent_x, agent_y) == self.goal:
            self.reset_goal()
            return 100, True
        #-100 reward if hit obstacle
        elif (agent_x, agent_y) in self.obstacles or (agent_x, agent_y) in self.dynamic_obstacles:
            return -100, True
        #reward to move in shortest distance
        else:
            return (-1 * distance_to_goal) , False

    def step(self):
        prev_state = self.agent.get_state(self)
        action_index = self.agent.choose_action(prev_state)
        self.agent.move(action_index)
        next_state = self.agent.get_state(self)
        reward,done = self.calculate_reward((self.agent.x, self.agent.y))
        self.agent.update_q_value(prev_state, action_index, reward, next_state)
        return reward,done

    #draw the environment- map
    def draw_environment(self):
        screen.fill((255, 255, 255))
        for x, y in self.obstacles:
            pygame.draw.rect(screen, COLOR_STATIC_OBSTACLE, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        for x, y in self.dynamic_obstacles:
            pygame.draw.rect(screen, COLOR_DYNAMIC_OBSTACLE, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, COLOR_GOAL, (self.goal[0] * CELL_SIZE, self.goal[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        self.agent.draw(screen)
        pygame.display.update()


environment = Environment()

#Episodes defining  as 10000
episodes = 10000

#setting plot list
rewards = []
exploration_rate_lst = []

#running till episodes
for i in range(0,episodes):

    print("Episode:",i+1)
    total_reward = 0
    done = False

    while not(done):
        # Move dynamic obstacles
        environment.move_dynamic_obstacles()
        reward,done = environment.step()
        total_reward += reward
        environment.draw_environment()

    rewards.append(total_reward)
    #changing exploitation or exploration
    if environment.agent.exploration_rate > EXPLORATION_MIN:
        environment.agent.exploration_rate *= EXPLORATION_DECAY
    else:
        environment.agent.exploration_rate = EXPLORATION_MIN
    exploration_rate_lst.append(environment.agent.exploration_rate)

pygame.quit()

#plotting
plt.figure(figsize=(10, 5))
plt.plot(rewards)
plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

# Plot losses during training
plt.figure(figsize=(10, 5))
plt.plot(exploration_rate_lst)
plt.title('Exploration Rate')
plt.xlabel('Episode')
plt.ylabel('Explortion rate')
plt.show()

# Saving the model as dqn_model.pth
model_path = 'dqn_model_train.pth'
torch.save(environment.agent.q_network.state_dict(), model_path)
