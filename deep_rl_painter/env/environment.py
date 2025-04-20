# needs to be reviewed - add proper comments
import os
import gym
import numpy as np
import cv2
import torch
from gym import spaces
from .canvas import init_canvas, update_canvas
from .reward import calculate_reward


class PaintingEnv(gym.Env):
    def __init__(self, target_image_path, canvas_size, max_strokes, device):
        super(PaintingEnv, self).__init__()

        self.device = device
        self.canvas_size = canvas_size
        self.max_strokes = max_strokes

        self.target_image = self.load_image(target_image_path)
        self.canvas = init_canvas(self.canvas_size)
        self.center = np.array(
            [self.canvas.shape[1] // 2, self.canvas.shape[0] // 2])
        self.radius = min(self.canvas.shape[0], self.canvas.shape[1]) // 2
        # Initialize the starting point on the circumference of the circle
        self.current_point = self.random_circle_point()

        self.used_strokes = 0

        # I dont understand what these are - Keshav
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(3, *self.canvas_size), dtype=np.float32
        )

    def load_image(self, path):
        # Load the image in grayscale and resize it to the canvas size
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file {path} not found.")
        if not path.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise ValueError(
                f"Unsupported image format: {path}. Please use PNG or JPG.")
        # Read the image in grayscale, we can add support for color images later
        # img -> [H, W, C] -> [H, W] (grayscale), numpy array, dtype=np.uint8
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img

    def random_circle_point(self):
        """Generates a random point on the circumference of a circle. As initial point."""
        theta = np.random.uniform(0, 2 * np.pi)
        return (self.center + self.radius * np.array([np.cos(theta), np.sin(theta)])).astype(int)

    def add_position_channels(self, canvas):
        """Adds a single position map channel to the grayscale canvas."""
        # canvas -> [H, W] (grayscale), numpy array, dtype=np.uint8
        # position_map -> [H, W] (position map), numpy array, dtype=np.float32
        h, w = canvas.shape
        position_map = np.zeros((h, w), dtype=np.float32)
        position_map[self.current_point[1], self.current_point[0]
                     ] = 100.0  # Set current point to 100
        stacked = np.stack([canvas, position_map],
                           axis=0).astype(np.float32)  # shape [2, H, W]
        return stacked

    def to_tensor(self, img):
        """Converts [H,W] numpy array to [1,C,H,W] normalized tensor with position encodings."""
        return torch.tensor(img).unsqueeze(0).to(self.device) # [1, C, H, W]

    def step(self, action, reward_function, lpips_fn=None):
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
            prev_tensor, current_tensor, target_tensor, reward_function, lpips_fn)

        # Not sure if this is correct - Keshav
        done = self.used_strokes >= self.max_strokes
        next_state = self.to_tensor(self.canvas).squeeze(
            0).cpu().numpy().flatten()

        # Not sure if this is correct - Keshav
        # Return flattened canvas with positional channels — shape (3, H, W) → flattened (196608,)
        flattened_state = self.add_position_channels(
            self.canvas).astype(np.float32).flatten()
        return flattened_state, reward.item(), done

    # Why do we need this? - Keshav
    def reset(self):
        self.canvas = init_canvas(self.canvas_size)
        self.current_point = self.random_circle_point()
        self.used_strokes = 0
        return self.add_position_channels(self.canvas).astype(np.float32).flatten()

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
