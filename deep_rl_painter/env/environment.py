# needs to be reviewed - add proper comments 
import gym
import numpy as np
import cv2
from gym import spaces
from env.canvas import init_canvas, update_canvas
from env.renderer import render_stroke
from reward import compute_reward

class PaintingEnv(gym.Env):
    def __init__(self, image_path, error_threshold, max_total_length):
        super(PaintingEnv, self).__init__()
        
        self.target_image = self.load_image(image_path)
        self.canvas = init_canvas(self.target_image.shape)
        self.center = np.array([self.canvas.shape[1] // 2, self.canvas.shape[0] // 2])
        self.radius = min(self.canvas.shape[0], self.canvas.shape[1]) // 2
        self.current_point = self.random_circle_point()
        
        self.error_threshold = error_threshold
        self.max_total_length = max_total_length
        self.used_length = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.canvas.shape, dtype=np.uint8
        )

    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, (256, 256))

    def random_circle_point(self):
        theta = np.random.uniform(0, 2 * np.pi)
        return (self.center + self.radius * np.array([np.cos(theta), np.sin(theta)])).astype(int)

    def step(self, action):
        direction = action / (np.linalg.norm(action) + 1e-8)
        next_point = (
            int(self.current_point[0] + direction[0] * self.radius),
            int(self.current_point[1] + direction[1] * self.radius)
        )

        self.canvas = update_canvas(self.canvas, tuple(self.current_point), tuple(next_point))
        self.used_length += self.radius
        self.current_point = next_point

        reward = compute_reward(self.target_image, self.canvas)
        done = reward >= -self.error_threshold or self.used_length >= self.max_total_length
        return self.canvas.flatten(), reward, done, {}

    def reset(self):
        self.canvas = init_canvas(self.target_image.shape)
        self.current_point = self.random_circle_point()
        self.used_length = 0
        return self.canvas.flatten()

    def render(self, mode='human'):
        cv2.imshow('Canvas', self.canvas)
        cv2.waitKey(1)
