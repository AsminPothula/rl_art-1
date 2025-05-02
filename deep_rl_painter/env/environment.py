"""
Painting Environment for Reinforcement Learning
This environment simulates a painting task where an agent learns to paint a target image
using strokes. The agent receives a reward based on how closely its painting matches the target image.

TODO:
- Add more detailed docstrings and comments.
- Implement the reward function in a separate module.
- Add more error handling and validation for inputs.
- Optimize the canvas update and rendering process.
- Add more visualization options for the canvas.
- Make this compatible with the rest of the code
- Add more tests and examples for usage.
- Make this compatible for rgb and grayscale images
"""

import os
import gym
import numpy as np
import cv2
import torch
from gym import spaces
from .canvas import init_canvas, update_canvas
from .reward import calculate_reward
import pdb
from typing import Tuple


class PaintingEnv(gym.Env):
    def __init__(self, target_image_path: str, canvas_size: Tuple[int, int], canvas_channels: int, max_strokes: int, device: str):
        """
        Initializes the Painting Environment.
        Args:
            target_image_path (str): Path to the target image.
            canvas_size (tuple): Size of the canvas (height, width).
            canvas_channels (int): Number of channels in the canvas (1 for grayscale, 3 for RGB).
            max_strokes (int): Maximum number of strokes allowed.
            device (torch.device): Device to run the computations on (CPU or GPU).

        # If channels is 1, use grayscale canvas (h, w) else (h, w, c)
        """
        super(PaintingEnv, self).__init__()

        self.device = device
        self.canvas_size = canvas_size
        self.max_strokes = max_strokes
        self.target_image_path = target_image_path
        self.channels = canvas_channels
        # target image is a numpy array of shape (H, W, C)
        self.target_image = self.load_image()
        self.center = np.array([self.canvas_size[0], self.canvas_size[1]]) // 2
        self.radius = min(self.canvas_size[0], self.canvas_size[1]) // 2

        # removed action space and observation space
        self._initialize()

    def _initialize(self):
        """
        Initializes the environment.
        Sets up the canvas, current point (starting point) and used_strokes (number of strokes used).
        """

        if self.channels == 1:  # Grayscale canvas
            self.canvas = init_canvas(
                (self.canvas_size[0], self.canvas_size[1], 1))
        else:  # RGB canvas
            self.canvas = init_canvas(
                (self.canvas_size[0], self.canvas_size[1], self.channels))
        self.current_point = self.random_circle_point()
        self.used_strokes = 0

    def load_image(self):
        """
        Loads an image from the given path and resizes it to the canvas size.
        Returns:
            img (numpy.ndarray): Loaded and resized image. (H, W, C)
        Raises:
            FileNotFoundError: If the image file is not found.
            ValueError: If the image format is not supported.
        """
        if not os.path.exists(self.target_image_path):
            raise FileNotFoundError(
                f"Image file {self.target_image_path} not found.")
        if not self.target_image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise ValueError(
                f"Unsupported image format: {self.target_image_path}. Please use PNG or JPG.")

        if self.channels == 1:
            img = cv2.imread(self.target_image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(
                img, (self.canvas_size[1], self.canvas_size[0]), interpolation=cv2.INTER_AREA)
            img = np.expand_dims(img, axis=-1)
        else:
            img = cv2.imread(self.target_image_path, cv2.IMREAD_COLOR)
            img = cv2.resize(
                img, (self.canvas_size[1], self.canvas_size[0]), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        return img

    def random_circle_point(self):
        """
        Generates a random point on the circumference of a circle. As initial point.
        Returns:
            (int, int): Random point on the circumference of the circle.
        """
        theta = np.random.uniform(0, 2 * np.pi)
        out = (self.center + self.radius *
               np.array([np.cos(theta), np.sin(theta)])).astype(int)
        return out

    def to_tensor(self, img):
        """Converts [H,W] numpy array to [1,C,H,W] normalized tensor with position encodings."""
        return torch.tensor(img).unsqueeze(0).to(self.device)  # [1, C, H, W]

    def step(self, action):
        """
        Takes a step in the environment using the given action.
        The action is a 2D vector representing the direction of the stroke.
        The reward is calculated based on the difference between the current canvas and the target image.
        """
        prev_canvas = self.canvas.copy()

        # Calculate direction and next point
        unit_vector = action / (np.linalg.norm(action) + 1e-8)
        next_point = (
            int(self.center[0] + unit_vector[0] * self.radius),
            int(self.center[1] + unit_vector[1] * self.radius)
        )

        self.canvas = update_canvas(self.canvas, tuple(
            self.current_point), tuple(next_point))
        self.used_strokes += 1
        self.current_point = next_point

        # Compute reward
        prev_tensor = self.to_tensor(prev_canvas)
        current_tensor = self.to_tensor(self.canvas)
        target_tensor = self.to_tensor(self.target_image)

        # Not sure if this is correct - Keshav
        reward = calculate_reward(
            prev_tensor, current_tensor, target_tensor, device=self.device)

        # Not sure if this is correct - Keshav
        done = self.used_strokes >= self.max_strokes
        next_state = self.to_tensor(self.canvas).squeeze(
            0).cpu().numpy().flatten()

        # No more flattening - Keshav
        return self.canvas, reward.item(), done

    def reset(self):
        """
        Resets the environment to its initial state.
        Returns:
            canvas (numpy.ndarray): The initial canvas state.
        """
        self._initialize()
        return self.canvas

    """def render(self, mode='human'):
        cv2.imshow('Canvas', self.canvas)
        cv2.waitKey(1)"""

    def render(self, episode_num=None, output_dir=None):
        """
        Saves the current canvas as an image in the specified output_dir with episode number.
        """
        if episode_num is not None and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"canvas_{episode_num}.png"
            path = os.path.join(output_dir, filename)
            cv2.imwrite(path, self.canvas)

    def close(self):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Example usage with the debugger
    target_path = '../target.jpg'
    canvas_size = (64, 64)
    max_strokes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        env = PaintingEnv(target_path, canvas_size, 1, max_strokes, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit()
    except ValueError as e:
        print(f"Error: {e}")
        exit()

    obs = env.reset()
    print("Initial observation shape:", obs.shape)

    for i in range(5):
        # Generate a random action
        action = env.action_space.sample()
        print(f"Step {i+1}: Action = {action}")

        # Set a breakpoint here to inspect variables
        # pdb.set_trace()

        next_obs, reward, done, _ = env.step(action)
        print("Next observation shape:", next_obs.shape)
        print("Reward:", reward)
        print("Done:", done)

        if done:
            print("Episode finished.")
            break

    env.close()
