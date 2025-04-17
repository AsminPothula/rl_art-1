# needs to be reviewed - add proper comments 
import os
import gym
import numpy as np
import cv2
import torch
from gym import spaces
from .canvas import init_canvas, update_canvas
from .renderer import render_stroke
from .reward import calculate_reward

class PaintingEnv(gym.Env):
    def __init__(self, target_image_path, canvas_size, max_strokes, device):
        super(PaintingEnv, self).__init__()
        
        self.device = device
        self.canvas_size = canvas_size
        self.max_strokes = max_strokes

        self.target_image = self.load_image(target_image_path)
        self.canvas = init_canvas(self.canvas_size)
        self.center = np.array([self.canvas.shape[1] // 2, self.canvas.shape[0] // 2])
        self.radius = min(self.canvas.shape[0], self.canvas.shape[1]) // 2
        self.current_point = self.random_circle_point()

        self.used_strokes = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(3, *self.canvas_size), dtype=np.float32
        )

    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, self.canvas_size)

    def random_circle_point(self):
        theta = np.random.uniform(0, 2 * np.pi)
        return (self.center + self.radius * np.array([np.cos(theta), np.sin(theta)])).astype(int)

    def add_position_channels(self, canvas):
        """Adds normalized x and y coordinate channels to the grayscale canvas."""
        h, w = canvas.shape
        x_map = np.tile(np.linspace(0, 1, w), (h, 1))         # shape [H, W]
        y_map = np.tile(np.linspace(0, 1, h), (w, 1)).T       # shape [H, W]
        stacked = np.stack([canvas / 255.0, x_map, y_map], axis=0)  # shape [3, H, W]
        return stacked.astype(np.float32)

    def to_tensor(self, img):
        """Converts [H,W] numpy array to [1,3,H,W] normalized tensor with position encodings."""
        stacked = self.add_position_channels(img)              # [3, H, W]
        return torch.tensor(stacked).unsqueeze(0).to(self.device)  # [1, 3, H, W]

    def step(self, action, reward_function, lpips_fn=None):
        prev_canvas = self.canvas.copy()

        # Calculate direction and next point
        direction = action / (np.linalg.norm(action) + 1e-8)
        next_point = (
            int(self.current_point[0] + direction[0] * self.radius),
            int(self.current_point[1] + direction[1] * self.radius)
        )

        self.canvas = update_canvas(self.canvas, tuple(self.current_point), tuple(next_point))
        self.used_strokes += 1
        self.current_point = next_point

        # Compute reward
        prev_tensor = self.to_tensor(prev_canvas)
        current_tensor = self.to_tensor(self.canvas)
        target_tensor = self.to_tensor(self.target_image)
        reward = calculate_reward(prev_tensor, current_tensor, target_tensor, reward_function, lpips_fn)

        done = self.used_strokes >= self.max_strokes
        next_state = self.to_tensor(self.canvas).squeeze(0).cpu().numpy().flatten()

        # Return flattened canvas with positional channels — shape (3, H, W) → flattened (196608,)
        flattened_state = self.add_position_channels(self.canvas).astype(np.float32).flatten()
        return flattened_state, reward.item(), done

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
