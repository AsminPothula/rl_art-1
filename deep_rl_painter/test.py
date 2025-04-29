import torch
import unittest
import numpy as np
from config import config
from env.environment import PaintingEnv
from models.actor import Actor
from models.critic import Critic
from models.ddpg import DDPGAgent
from utils.replay_buffer import ReplayBuffer
from utils.noise import OUNoise
import lpips
import os

class TestDeepRLPipeline(unittest.TestCase):
    def setUp(self):
        self.device = config["device"]
        self.image_size = config["image_size"]
        self.in_channels = 3
        self.action_dim = 6

        # Minimal config override
        config["batch_size"] = 4
        config["replay_buffer_size"] = 10

        self.env = PaintingEnv(
            target_image_path=config["target_image_path"],
            canvas_size=self.image_size,
            max_strokes=5,
            device=self.device
        )

        self.actor = Actor(
            image_encoder_model=config["model_name"],
            image_encoder_model_2=config["model_name"],
            pretrained=False,  # for test speed
            out_neurons=self.action_dim,
            in_channels=self.in_channels
        )

        self.critic = Critic(
            image_encoder_model=config["model_name"],
            image_encoder_model_2=config["model_name"],
            pretrained=False,
            out_neurons=1,
            in_channels=self.in_channels
        )

        self.agent = DDPGAgent(
            actor=self.actor,
            critic=self.critic,
            actor_target=self.actor,
            critic_target=self.critic,
            actor_optimizer=torch.optim.Adam(self.actor.parameters(), lr=1e-4),
            critic_optimizer=torch.optim.Adam(self.critic.parameters(), lr=1e-3),
            replay_buffer=ReplayBuffer(config["replay_buffer_size"]),
            noise=OUNoise(self.action_dim),
            config=config,
            channels=self.in_channels
        )

    def test_env_reset_and_step(self):
        state = self.env.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (2 * self.in_channels * self.image_size[0] * self.image_size[1],))

        action = self.env.action_space.sample()
        next_state, reward, done = self.env.step(action, lambda *a, **kw: 1.0, None)
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_agent_action_shape(self):
        state = self.env.reset()
        action = self.agent.select_action(state)
        self.assertEqual(action.shape, (self.action_dim,))
        self.assertTrue(np.all(np.abs(action) <= 1))

    def test_agent_training_step(self):
        state = self.env.reset()
        for _ in range(config["batch_size"] + 1):
            action = self.agent.act(state)
            next_state, reward, done = self.env.step(action, lambda *a, **kw: 1.0, None)

            self.agent.replay_buffer.store(
                state.flatten(), 
                action, 
                float(reward), 
                next_state.flatten(), 
                float(done)
            )
            state = next_state

        try:
            self.agent.train()
        except Exception as e:
            self.fail(f"Agent training raised an exception: {e}")

    def test_reward_shapes(self):
        from env.reward import calculate_ssim_reward, calculate_mse_reward, calculate_lpips_reward

        B, C, H, W = 2, self.in_channels, *self.image_size
        prev_canvas = torch.rand(B, C, H, W).to(self.device)
        curr_canvas = torch.rand(B, C, H, W).to(self.device)
        target_image = torch.rand(B, C, H, W).to(self.device)

        self.assertEqual(calculate_ssim_reward(prev_canvas, curr_canvas, target_image).shape, (B, 1))
        self.assertEqual(calculate_mse_reward(prev_canvas, curr_canvas, target_image).shape, (B, 1))

        lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
        self.assertEqual(calculate_lpips_reward(prev_canvas, curr_canvas, target_image, lpips_fn).shape, (B, 1))

if __name__ == "__main__":
    unittest.main()
