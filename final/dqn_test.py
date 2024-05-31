#importing the libraries
import pygame
import random
import numpy as np
import math
#pytorch
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2

# Defining map 
WIDTH, HEIGHT = 500, 500
CELL_SIZE     = 25
NUM_CELLS_X   = WIDTH // CELL_SIZE
NUM_CELLS_Y   = HEIGHT // CELL_SIZE

#color setting
COLOR_AGENT            = (0, 0, 255)
COLOR_GOAL             = (0, 255, 0)
COLOR_STATIC_OBSTACLE  = (64, 64, 64)
COLOR_DYNAMIC_OBSTACLE = (255, 165, 0)
COLOR_SENSOR_RAY       = (128, 128, 128)

#Pygame init
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Environment")
clock = pygame.time.Clock()

# Q-Net defining
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Deep Q-Learning Agent defining
class DeepQLearningAgent:
    def __init__(self, x, y, sensor_range=2, num_sensors=8):
        self.x, self.y = x, y
        self.sensor_range = sensor_range
        self.num_sensors = num_sensors
        self.sensor_angles = [i * (360 // num_sensors) for i in range(num_sensors)]
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  

        #set input and output dims
        input_dim = num_sensors + 2  
        output_dim = len(self.actions)

        self.q_network = QNetwork(input_dim, output_dim)
        self.load_pretrained_model()

    #loading the trined model file
    def load_pretrained_model(self):
        model_path = 'dqn_model_train.pth'
        self.q_network.load_state_dict(torch.load(model_path))

    #get the states
    def get_state(self, environment):
        state = [self.sensor_read(environment, math.radians(angle)) for angle in self.sensor_angles]
        rel_pos_to_goal = (environment.goal[0] - self.x, environment.goal[1] - self.y)
        return tuple(state + list(rel_pos_to_goal))

    #read the sensor values from ray sensor
    def sensor_read(self, environment, angle):
        dx, dy = math.cos(angle), math.sin(angle)
        for step in range(1, self.sensor_range + 1):
            check_x, check_y = int(self.x + dx * step), int(self.y + dy * step)
            if (check_x, check_y) in environment.obstacles or (check_x, check_y) in environment.dynamic_obstacles:
                return step
        return self.sensor_range

    #choosingthe actions
    def choose_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # Get Q-values from the neural network
        q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    #function to move the agent
    def move(self, action_index):
        action = self.actions[action_index]
        next_x, next_y = self.x + action[0], self.y + action[1]
        if (next_x, next_y) not in environment.obstacles and (next_x, next_y) not in environment.dynamic_obstacles:
            self.x, self.y = max(0, min(NUM_CELLS_X - 1, next_x)), max(0, min(NUM_CELLS_Y - 1, next_y))

    def draw(self, screen):
        pygame.draw.rect(screen, COLOR_AGENT, (self.x * CELL_SIZE, self.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        for angle in self.sensor_angles:
            radians = math.radians(angle)
            end_x = self.x * CELL_SIZE + CELL_SIZE // 2 + math.cos(radians) * self.sensor_range * CELL_SIZE
            end_y = self.y * CELL_SIZE + CELL_SIZE // 2 + math.sin(radians) * self.sensor_range * CELL_SIZE
            pygame.draw.line(screen, COLOR_SENSOR_RAY, (self.x * CELL_SIZE + CELL_SIZE // 2, self.y * CELL_SIZE + CELL_SIZE // 2), (end_x, end_y), 1)

#set  strt goal lst to compare
class Environment:
    def __init__(self):
        self.agent = DeepQLearningAgent(10, 10)
        self.obstacles = [(1, 1), (10, 1), (18, 18), (18, 1), (1, 5), (5, 5), (18, 5), (9, 3), (15, 7), (17, 15), (1, 5), (1, 20), (3, 10), (16, 10), (8, 8), (1, 17), (5, 15), (5, 17), (11, 19), (17, 15)]
        self.dynamic_obstacles = [(6, 6), (19, 19), (3, 10), (5, 15), (15, 12)]
        self.goal = (15, 15)
        #defining the goal list to compare with q-Learning
        self.goal_lst = [(5, 7), (7, 17), (4, 12), (0, 17), (11, 12), (12, 3), (8, 4), (1, 14), (3, 7), (8, 14), (15, 5), (18, 2), (10, 7), (8, 12), (15, 19), (2, 2), (15, 14), (17, 5), (0, 15), (1, 7), (2, 1), (16, 19), (19, 3), (4, 2), (4, 1)]
        self.index = 0
        
    def reset_goal(self):
        self.goal = self.goal_lst[self.index]
        self.index = self.index + 1
        
    def move_dynamic_obstacles(self):
        for i, (x, y) in enumerate(self.dynamic_obstacles):
            if random.random() > 0.2:
                continue
            dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
            new_x, new_y = (x + dx) % NUM_CELLS_X, (y + dy) % NUM_CELLS_Y
            if (new_x, new_y) not in self.obstacles and (new_x, new_y) != self.goal:
                self.dynamic_obstacles[i] = (new_x, new_y)
    #reward function
    def calculate_reward(self, next_state):
        goal_x, goal_y = self.goal
        agent_x, agent_y = next_state
        distance_to_goal = math.sqrt((goal_x - agent_x)**2 + (goal_y - agent_y)**2)
        if (agent_x, agent_y) == self.goal:
            self.reset_goal()
            return 100, True, False
        elif (agent_x, agent_y) in self.obstacles or (agent_x, agent_y) in self.dynamic_obstacles:
            return -100, True, True
        else:
            return (-1 * distance_to_goal), False, False

    #setting the step
    def step(self):
        prev_state = self.agent.get_state(self)
        action_index = self.agent.choose_action(prev_state)
        self.agent.move(action_index)
        next_state = self.agent.get_state(self)
        reward, done,obs = self.calculate_reward((self.agent.x, self.agent.y))
        return reward, done,obs

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
episodes = 25  # Number of episodes to run for goal_lst

rewards = []
steps = []
frames = []

#saving as video
video_name = 'output.mp4'
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (WIDTH, HEIGHT))

# Run the episodes
for i in range(episodes):
    print("Episode:", i + 1)
    total_reward = 0
    step = 0
    done = False
    while not done:
        # Move dynamic obstacles
        environment.move_dynamic_obstacles()
        reward, done,obs = environment.step()
        total_reward += reward

        environment.draw_environment()
        step+=1

        #saving opencv frames to save as video
        pygame_image = pygame.surfarray.array3d(screen)
        finalImage = cv2.cvtColor(pygame_image.swapaxes(0, 1), cv2.COLOR_RGB2BGR)
        out.write(finalImage)
        pygame.display.flip()

        clock.tick(10)
        if(obs == True):
            step=-1

    rewards.append(total_reward)
    steps.append(step)

# Quit Pygame
out.release()
pygame.quit()

#plotting
plt.figure(figsize=(10, 5))
plt.plot(rewards)
plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(steps)
plt.title('Total steps')
plt.xlabel('Episode')
plt.ylabel('steps')
plt.show()
plt.figure(figsize=(10, 5))
plt.bar(range(len(environment.goal_lst)), steps, tick_label=[str(goal) for goal in environment.goal_lst])
plt.xticks(rotation=90)
plt.ylabel('Number of Steps')
plt.title('Steps to Goal')
plt.show()
