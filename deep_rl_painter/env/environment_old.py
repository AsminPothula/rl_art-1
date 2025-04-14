import gym
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from gym import spaces
import os

class StringArtEnv(gym.Env):
    def __init__(self, image_path, num_points=100):
        super(StringArtEnv, self).__init__()

        self.max_steps = 50  # Limit total actions per episode
        self.current_step = 0  # Step counter

        # Load the target image
        self.target_image = self.preprocess_image(image_path)
        self.canvas = np.zeros_like(self.target_image)  # Empty canvas
        self.border_points = self.get_border_points(self.target_image, num_points) #border_points = indices of points 

        self.ind_init = random.randint(0, len(self.border_points) - 1) # picking a random border point index 

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(self.border_points))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.target_image.size,), dtype=np.uint8
        )

    """Detects areas with sharp intensity changes and identifies them as potential edges.
    Filters edges by discarding weak ones below 50, keeping strong ones above 150, and retaining intermediate ones only if connected to strong edges.
    Generates the final edge map with only the most significant edges preserved.
    Converts the result into a binary image where edges appear white and the background is black."""
    
    def preprocess_image(self, image_path):
        """Loads an image, converts to grayscale, and extracts edges."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.Canny(img, 50, 150)  # Edge detection 
        return img # edges are white, rest is black 

    def get_border_points(self, image, num_points):
        """Detects border points and selects a subset."""
        edges = np.argwhere(image > 0)  #  finds all non-zero (white) pixels, returning their row and column indices
        selected_points = random.sample(list(edges), num_points) # out of all edges, it picks num_points number of points 
        return selected_points

    def step(self, action):
        """Executes an action (draws a string) and updates state."""
        self.current_step += 1  # Increment step count

        ind_dest = action  # Selected point index
        start = self.border_points[self.ind_init]
        end = self.border_points[ind_dest]

        # Draw a line on the canvas - converting (y,x) to (x,y)
        cv2.line(self.canvas, tuple(start[::-1]), tuple(end[::-1]), color=255, thickness=1)

        # Update state
        self.ind_init = ind_dest  

        # Compute reward based on image similarity - assigning the (negative of) distance as the reward itself for now 
        reward = -np.linalg.norm(self.target_image.flatten() - self.canvas.flatten())

        # Stop condition: end episode after max_steps
        done = self.current_step >= self.max_steps

        return self.canvas.flatten(), reward, done, {}

    def reset(self):
        """Resets the environment for a new episode."""
        self.canvas = np.zeros_like(self.target_image)
        self.ind_init = random.randint(0, len(self.border_points) - 1) #out of all the border points, one is being picked as the initial point
        self.current_step = 0  # Reset step count
        return self.canvas.flatten()

    def render(self, mode="human"):
        """Saves the current canvas with border points after each action to the 'pics' folder in Codespaces."""
        
        # Define the save path inside the Codespaces directory
        save_path = os.path.join(os.getcwd(), "pics")  # Saves to a "pics" folder in the current working directory

        # Ensure the directory exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)  # Create folder if it doesn't exist

        # Convert canvas to a 3-channel image (so we can add color points)
        display_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_GRAY2BGR)
        
        # Draw border points as red dots
        for point in self.border_points:
            cv2.circle(display_canvas, tuple(point[::-1]), radius=5, color=(0, 0, 255), thickness=-1)

        # Save image to the "pics" folder
        frame_path = os.path.join(save_path, f"frame_{self.current_step}.png")

        # Print the path before saving
        print(f"Saving image to: {frame_path}")

        cv2.imwrite(frame_path, display_canvas)

        print(f"Saved: {frame_path}")






