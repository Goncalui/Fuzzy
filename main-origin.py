import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

import pygame
import sys
import time
import pandas as pd
import threading
import tkinter as tk
from tkinter import Scale, Button
import time
import os
import warnings
warnings.filterwarnings("ignore")

matplotlib.use('TkAgg')

random = 0 # 0 to not random

class CustomEnv:
    def __init__(self, rows=10, columns=10, start_position=(0, 0), goal_position=(7, 2), reward=1.0,
                 alpha=0.1, gamma=0.9, epsilon=0.5):
        self.rows = rows
        self.columns = columns
        self.start_position = start_position
        self.goal_position = goal_position
        self.current_position = start_position
        self.reward_position = goal_position
        self.reward = reward
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.recStep = 0

        # Action space: 0 = down, 1 = up, 2 = left, 3 = right, 4 = stay
        self.action_space = [0, 1, 2, 3, 4]

        # Initialize Q-table
        self.q_table = np.zeros((self.rows, self.columns, len(self.action_space)))
    
    def restart(self):
        self.rows = 10
        self.columns = 10
        self.start_position=(0, 0)
        self.goal_position=(7, 2)
        self.reward=1.0
        self.alpha=0.4
        self.gamma=0.9
        self.epsilon=0.5
        self.recStep = 0

        # Action space: 0 = down, 1 = up, 2 = left, 3 = right, 4 = stay
        self.action_space = [0, 1, 2, 3, 4]

        # Initialize Q-table
        self.q_table = np.zeros((self.rows, self.columns, len(self.action_space)))


    def reset(self):
        global random
        if( random):
            self.current_position = [np.random.randint(0,10),np.random.randint(0,10)]
        else:
            self.current_position = [0,0] # [np.random.randint(0,10),np.random.randint(0,10)]
        self.recStep = 0
        return self.current_position

    def step(self, action):
        row, col = self.current_position

        done = 0
        
        # Update position based on action
        if action == 0:  # down
            row = min(row + 1, self.rows - 1)
        elif action == 1:  # up
            row = max(row - 1, 0)
        elif action == 3:  # left
            col = max(col - 1, 0)
        elif action == 2:  # right
            col = min(col + 1, self.columns - 1) 
        elif action == 4: #stay
            pass  # Do nothing, stay in the current position

        # Update position
        self.current_position = (row, col)
    
        if(self.current_position == self.reward_position):
            reward = self.reward +100
            done = (self.current_position == self.goal_position)
            return self.current_position, reward, done

        self.recStep -= 1 + self.reward

        # Calculate reward
        reward = -1.0
        if self.current_position == self.reward_position:
            reward = self.reward +5
        

        return self.current_position, reward, done

    def update_q_table(self, state, action, reward, next_state):
        row, col = state
        next_row, next_col = next_state

        current_q_value = self.q_table[row, col, action]
        max_next_q_value = np.max(self.q_table[next_row, next_col, :])

        new_q_value = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * max_next_q_value)
        self.q_table[row, col, action] = new_q_value

    def get_valid_actions(self, state):
            row, col = state
            valid_actions = [4]
            if col > 0:
                valid_actions.append(3)  # left
            if col < self.columns - 1:
                valid_actions.append(2)  # right
            if row > 0:
                valid_actions.append(1)  # up
            if row < self.rows - 1:
                valid_actions.append(0)  # down
            return valid_actions

    def choose_action(self, state):
        # Explore or exploit while considering border restrictions
        valid_actions = self.get_valid_actions(state)
        
        if np.random.rand() < self.epsilon:
            action = np.random.choice(valid_actions)
        else:
            q_values = self.q_table[state[0], state[1], :]
            valid_q_values = [q_values[a] for a in valid_actions]
            action = valid_actions[np.argmax(valid_q_values)]

        return action
    


def update_plot3D_thread():
    global env
    global rewards, stop_thread
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the current figure manager for the first plot
    fig_manager1 = plt.get_current_fig_manager()

    # Set the position of the first window (adjust these values as needed)
    fig_manager1.window.geometry("540x540+1380+0")

    # Create a figure and axis
    fig2, ax2 = plt.subplots()
    
    # Get the current figure manager for the first plot
    fig_manager2 = plt.get_current_fig_manager()

    # Set the position of the first window (adjust these values as needed)
    fig_manager2.window.geometry("540x540+1380+540")
    # Create a figure and axis
    fig3, ax3 = plt.subplots()
        
    # Get the current figure manager for the first plot
    fig_manager3 = plt.get_current_fig_manager()

    # Set the position of the first window (adjust these values as needed)
    fig_manager3.window.geometry("1388x282+-7+730")

    while 1:
        Q = env.q_table
        ax.cla()
        ax2.cla()
        ax3.cla()
        # Your data
        # Q = env.q_table  # Replace [...] with the actual data

        # Convert the list of lists into a NumPy array
        Q_array = np.array(Q)

        # Replace all occurrences of 0 with -10
        Q_array[Q_array == 0] = -10

        # Add 5 to all elements
        Q_array += 5

        # Create a meshgrid for the indices
        x, y, z = np.meshgrid(range(Q_array.shape[0]), range(Q_array.shape[1]), range(Q_array.shape[2]))

        # Flatten the arrays
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()

        # Flatten the Q_array
        Q_flat = Q_array.flatten()

        # Create a 3D scatter plot
        ax.scatter(x_flat, y_flat, z_flat, c=Q_flat, cmap='viridis', s=50, alpha=0.8)
                
        window_size = 10  # Adjust the window size as needed
        moving_average = pd.Series(rewards).rolling(window=window_size).mean()

        # Plot original rewards
        ax3.plot(np.arange(len(rewards)), rewards, label='Original Rewards')

        # Plot moving average
        ax3.plot(np.arange(len(moving_average)), moving_average, label=f'Moving Average ({window_size} periods)')
                
        ax3.set_xlabel('Index')
        ax3.set_ylabel('Reward')
        ax3.legend()


        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Create a new 10x10 matrix filled with zeros
        new_matrix = np.zeros((10, 10), dtype=int)
        for i in range(10):
            for j in range(10):
                # Find the index of the maximum value in the corresponding row and column of the original table
                max_index = np.argmax(Q_array[i, j, :])
                
                # Update the new matrix with the found index
                new_matrix[i, j] = max_index

        arrow_directions = np.empty_like(new_matrix, dtype='U1')
        arrow_directions[new_matrix == 0] = '↓'
        arrow_directions[new_matrix == 1] = '↑'
        arrow_directions[new_matrix == 3] = '←'
        arrow_directions[new_matrix == 2] = '→'
        arrow_directions[new_matrix == 4] = '0'


        # Iterate over each element in the matrix
        for i in range(10):
            for j in range(10):
                direction = new_matrix[i][j]
                if direction == 0:  # Down
                    arrow = 'v'
                elif direction == 1:  # Up
                    arrow = '^'
                elif direction == 2:  # Left
                    arrow = '>'
                elif direction == 3:  # Right
                    arrow = '<'
                elif direction == 4:  # Down-right
                    arrow = 's'
                else:  # No arrow for 0
                    continue
                ax2.scatter( j,-i+10, marker=arrow, s=100, c='black')

        ax2.set_xticks(np.arange(0, 10, 1))
        ax2.set_yticks(np.arange(0, 10, 1))
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.grid(False)


        # Update the display for matplotlib
        plt.pause(0.0001)

# Create the custom environment
env = CustomEnv()

stop_thread = False
start = False

# Initialize Pygame
pygame.init()

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Set up the display
cell_size = 70
rows = 10
columns = 10
screen_width = columns * cell_size
screen_height = rows * cell_size
window_position = [0,30]
# Set the environment variable for window position
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_position[0]},{window_position[1]}"

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Custom Environment Visualization')


# Define a function to draw the grid
def draw_grid():
    for i in range(0, screen_width, cell_size):
        pygame.draw.line(screen, BLACK, (i, 0), (i, screen_height))
    for j in range(0, screen_height, cell_size):
        pygame.draw.line(screen, BLACK, (0, j), (screen_width, j))
    
    # Draw the goal
    pygame.draw.rect(screen, GREEN, (env.goal_position[1] * cell_size, env.goal_position[0] * cell_size, cell_size, cell_size))


# Q-learning algorithm
num_episodes = 100

delay = 1
# Create a figure for the 3D scatter plot
plot_thread = threading.Thread(target=update_plot3D_thread)

# Start the plot update thread
plot_thread.start()



stop_visua = False
pause_visua = False
# Function to update epsilon based on slider value
def update_epsilon(value):
    env.epsilon = float(value)

# Function to update delay based on slider value
def update_delay(value):
    global delay
    delay = float(value)

# Function to start the visualization
def start_visualization():
    global stop_visua
    stop_visua = False

# Function to stop the visualization
def stop_visualization():
    global stop_visua,env
    stop_visua = True
    env.q_table = np.zeros((10, 10, len(env.action_space)))

# Function to pause the visualization
def pause_visualization():
    global pause_visua
    pause_visua = not pause_visua


# Tkinter thread function
def tk_thread():


    # Create Tkinter window
    root = tk.Tk()
    root.title("Reinforcement Learning Visualization Controls")

    # Epsilon slider
    epsilon_label = tk.Label(root, text="Epsilon:", font=("Arial", 14))
    epsilon_slider = Scale(root, from_=0, to=1, orient="horizontal", resolution=0.0001, command=update_epsilon,  length=300, sliderlength=50) 
    epsilon_slider.set(0.75)

    # Delay slider
    delay_label = tk.Label(root, text="Delay:", font=("Arial", 14))
    delay_slider = Scale(root, from_=0.000001, to=1, orient="horizontal", resolution=0.0001, command=update_delay, length=300, sliderlength=50)
    delay_slider.set(0.000001)

    # Start, Stop, and Pause buttons
    
    button_font = ("Arial", 14)
    start_button = Button(root, text="Start", command=start_visualization ,font = button_font,width = 10, height = 2)
    stop_button = Button(root, text="Stop", command=stop_visualization    ,font = button_font,width = 10, height = 2)
    pause_button = Button(root, text="Pause", command=pause_visualization ,font = button_font,width = 10, height = 2)

    # Pack elements into the Tkinter window
    epsilon_label.pack()
    epsilon_slider.pack()
    delay_label.pack()
    delay_slider.pack()
    start_button.pack()
    stop_button.pack()
    pause_button.pack()
    
    # Set the initial position of the Tkinter window
    root.geometry("700x700+700+0")  # Change the values based on your preferred position

    # Run Tkinter main loop
    while True:
        root.update()  # Update Tkinter window
        time.sleep(0.01)

# Create Tkinter thread
tk_thread = threading.Thread(target=tk_thread, daemon=True)
tk_thread.start()



# Create the custom environment
env = CustomEnv()

stop_thread = False
start = False

# Initialize Pygame
pygame.init()

from datetime import datetime
# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Set up the display
cell_size = 70
rows = 10
columns = 10
screen_width = columns * cell_size
screen_height = rows * cell_size
window_position = [0, 30]
# Set the environment variable for window position
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_position[0]},{window_position[1]}"

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Custom Environment Visualization')

# Q-learning algorithm
num_episodes = 100
delay = 1/1000000

# Create a figure for the 3D scatter plot
plot_thread = threading.Thread(target=update_plot3D_thread)
plot_thread.start()

stop_visua = False
pause_visua = False

# List to store rewards from multiple runs
all_rewards = []


rewards = []
episode = 0
stop_visua = False

import random as rddd
import csv

# Run the algorithm 50 times
for run in range(5):
    env = CustomEnv()
    rewards = []
    episode = 0
    stop_visua = False

    while episode < num_episodes:
        if stop_visua:
            episode = 0
            env.restart()
            Q = env.q_table
            rewards = []

        state = env.reset()
        done = False

        while not done and not stop_thread and not stop_visua:
            if not pause_visua:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        stop_thread = True
                        plot_thread.join()
                        plt.close('all')
                        pygame.quit()
                        sys.exit()

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_w:
                            env.epsilon += 0.1
                        elif event.key == pygame.K_s:
                            env.epsilon -= 0.1
                        elif event.key == pygame.K_a:
                            delay *= 10
                        elif event.key == pygame.K_d:
                            delay /= 10

                screen.fill(WHITE)
                draw_grid()

                pygame.draw.rect(screen, RED, (state[1] * cell_size, state[0] * cell_size, cell_size, cell_size))
                pygame.display.flip()
                time.sleep(delay)

                action = env.choose_action(state)
                next_state, reward, done = env.step(action)
                env.update_q_table(state, action, reward, next_state)
                state = next_state

        rewards.append(env.recStep)
        episode += 1
    # Save rewards from this run
    all_rewards.append(rewards)
    
    # Create a unique filename
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    random_number = rddd.randint(10000, 99999)
    filename = f'runs4/{timestamp}_{random_number}.csv'
    
    # Write rewards to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Reward'])
        for i, reward in enumerate(rewards):
            writer.writerow([i+1, reward])

# Plot the results
plt.figure(figsize=(10, 6))
for i, rewards in enumerate(all_rewards):
    plt.plot(rewards, label=f'Run {i+1}')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards per Episode for 50 Runs')
plt.legend()
plt.show()

        
        
# Quit Pygame
pygame.quit()

