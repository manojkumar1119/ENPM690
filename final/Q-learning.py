import pygame
import random
import numpy as np
import math
import matplotlib.pyplot as plt

# Configuration settings for the game display
DISPLAY_WIDTH, DISPLAY_HEIGHT = 500, 500
GRID_SIZE = 25
GRID_WIDTH = DISPLAY_WIDTH // GRID_SIZE
GRID_HEIGHT = DISPLAY_HEIGHT // GRID_SIZE
AGENT_COLOR = (0, 0, 255)  # Blue color for the agent
TARGET_COLOR = (0, 255, 0)  # Green color for the target
OBSTACLE_COLOR = (64, 64, 64)  # Dark gray color for static obstacles
MOVING_OBSTACLE_COLOR = (255, 165, 0)  # Orange color for moving obstacles
RAY_COLOR = (128, 128, 128)  # Gray color for sensor rays

# Initialize Pygame
pygame.init()
display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption("QLearning Path Finder")
timer = pygame.time.Clock()

class QLearningAgent:
    """ A Q-learning agent that navigates a grid environment. """
    def __init__(self, init_x, init_y, environment, alpha=0.1, gamma=0.9, epsilon=1.0, sensor_dist=2, sensor_count=8):
        self.environment = environment  # Environment in which the agent operates
        self.pos_x, self.pos_y = init_x, init_y  # Initial position of the agent
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.sensor_dist = sensor_dist  # How far each sensor can read
        self.sensor_count = sensor_count  # Number of sensors around the agent
        self.sensor_directions = [i * (360 // sensor_count) for i in range(sensor_count)]
        self.moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.q_values = {}

    def sense_environment(self, env):
        """ Returns the agent's perception of the environment around it, including obstacles and target relative position. """
        readings = [self.detect(env, math.radians(dir)) for dir in self.sensor_directions]
        goal_direction = (env.target[0] - self.pos_x, env.target[1] - self.pos_y)
        return tuple(readings + list(goal_direction))

    def detect(self, env, radian):
        """ Detect obstacles using sensors at given radian angle. """
        dx, dy = math.cos(radian), math.sin(radian)
        for dist in range(1, self.sensor_dist + 1):
            test_x, test_y = int(self.pos_x + dx * dist), int(self.pos_y + dy * dist)
            if (test_x, test_y) in env.blocked or (test_x, test_y) in env.moving_blocks:
                return dist
        return self.sensor_dist

    def decide(self, state):
        """ Choose an action based on the current state and learned Q-values. """
        if state not in self.q_values:
            self.q_values[state] = np.zeros(len(self.moves))
        if random.random() < self.epsilon:
            return random.choice(range(len(self.moves)))
        return np.argmax(self.q_values[state])

    def update_q(self, prev_state, action, reward, next_state):
        """ Update the Q-values for the given state and action. """
        if next_state not in self.q_values:
            self.q_values[next_state] = np.zeros(len(self.moves))
        old_val = self.q_values[prev_state][action]
        future_reward = np.max(self.q_values[next_state])
        new_val = old_val + self.alpha * (reward + self.gamma * future_reward - old_val)
        self.q_values[prev_state][action] = new_val

    def navigate(self, move_idx):
        """ Move the agent in the environment based on the chosen action. """
        move_x, move_y = self.moves[move_idx]
        new_x, new_y = self.pos_x + move_x, self.pos_y + move_y
        if (new_x, new_y) not in self.environment.blocked and (new_x, new_y) not in self.environment.moving_blocks:
            self.pos_x, self.pos_y = max(0, min(GRID_WIDTH - 1, new_x)), max(0, min(GRID_HEIGHT - 1, new_y))

    def render(self, canvas):
        """ Draw the agent and its sensors on the Pygame display. """
        pygame.draw.rect(canvas, AGENT_COLOR, (self.pos_x * GRID_SIZE, self.pos_y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        for dir in self.sensor_directions:
            rad = math.radians(dir)
            end_x = self.pos_x * GRID_SIZE + GRID_SIZE // 2 + math.cos(rad) * self.sensor_dist * GRID_SIZE
            end_y = self.pos_y * GRID_SIZE + GRID_SIZE // 2 + math.sin(rad) * self.sensor_dist * GRID_SIZE
            pygame.draw.line(canvas, RAY_COLOR, (self.pos_x * GRID_SIZE + GRID_SIZE // 2, self.pos_y * GRID_SIZE + GRID_SIZE // 2), (end_x, end_y), 1)

class GameEnvironment:
    """ The main environment class that holds the game state, including the agent, obstacles, and the target. """
    def __init__(self):
        self.explorer = QLearningAgent(GRID_WIDTH // 2, GRID_HEIGHT // 2, self)  
        self.blocked = [(1, 1), (10, 1), (18, 18), (18, 1), (1, 10), (5, 5), (18, 5), (9, 3), (15, 7), (17, 15), (3, 18), (6, 12)]
        self.moving_blocks = [(6, 6), (19, 19)]
        self.target = (15, 15)
        self.reposition_target()

    def reposition_target(self, specific_target=None):
        """ Randomly reposition the target in the grid unless a specific target is provided. """
        if specific_target:
            self.target = specific_target
        else:
            while True:
                new_target = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
                if new_target not in self.blocked and new_target not in self.moving_blocks and new_target != (self.explorer.pos_x, self.explorer.pos_y):
                    self.target = new_target
                    break

    def move_blocks(self):
        """ Move the dynamic obstacles randomly around the grid. """
        for index, (x, y) in enumerate(self.moving_blocks):
            if random.random() > 0.2:
                continue
            new_x, new_y = (x + random.choice([0, 1, -1])) % GRID_WIDTH, (y + random.choice([0, 1, -1])) % GRID_HEIGHT
            if (new_x, new_y) not in self.blocked and (new_x, new_y) != self.target:
                self.moving_blocks[index] = (new_x, new_y)

    def reward(self, new_position):
        """ Calculate the reward based on the agent's position relative to the target and obstacles. """
        target_x, target_y = self.target
        explorer_x, explorer_y = new_position
        dist_to_target = math.sqrt((target_x - explorer_x)**2 + (target_y - explorer_y)**2)
        if new_position == self.target:
            self.reposition_target()
            return 100  # High reward for reaching the target
        elif new_position in self.blocked or new_position in self.moving_blocks:
            return -100  # High penalty for hitting an obstacle
        else:
            return -1 * dist_to_target  # Negative reward proportional to the distance to the target

    def process_step(self):
        """ Process a single step of the environment, including decision making, navigation, and reward assessment. """
        current_state = self.explorer.sense_environment(self)
        action = self.explorer.decide(current_state)
        self.explorer.navigate(action)
        new_state = self.explorer.sense_environment(self)
        step_reward = self.reward((self.explorer.pos_x, self.explorer.pos_y))
        self.explorer.update_q(current_state, action, step_reward, new_state)

    def visualise_scene(self):
        """ Visualise the current state of the game environment, including all entities. """
        display.fill((255, 255, 255))
        for x, y in self.blocked:
            pygame.draw.rect(display, OBSTACLE_COLOR, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        for x, y in self.moving_blocks:
            pygame.draw.rect(display, MOVING_OBSTACLE_COLOR, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(display, TARGET_COLOR, (self.target[0] * GRID_SIZE, self.target[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        self.explorer.render(display)
        pygame.display.update()

# List of targets and results
targets = [(5, 7), (7, 17), (4, 12), (0, 17), (11, 12), (12, 3), (8, 4), (1, 14), (3, 7), (8, 14), (15, 5), (18, 2), (10, 7), (8, 12), (15, 19), (2, 2), (15, 14), (17, 5), (0, 15), (1, 7), (2, 1), (16, 19), (19, 3), (4, 2), (4, 1)]
result_steps = []

sim_env = GameEnvironment()
active = True
target_count = 0

while active and target_count < len(targets):
    sim_env.reposition_target(targets[target_count])
    step_count = 0
    while (sim_env.explorer.pos_x, sim_env.explorer.pos_y) != targets[target_count]:
        for event in pygame.event.get():
            if event.type is pygame.QUIT:
                active = False
        sim_env.move_blocks()
        sim_env.process_step()
        sim_env.visualise_scene()
        timer.tick(10)
        step_count += 1
    result_steps.append(step_count)
    target_count += 1

    if not active:
        break

pygame.quit()

# Visualization of the steps required to reach each target
plt.figure(figsize=(10, 5))
plt.bar(range(len(targets)), result_steps, tick_label=[str(target) for target in targets])
plt.xticks(rotation=90)
plt.ylabel('Steps Required')
plt.title('Efficiency of Navigation per Target')
plt.show()

