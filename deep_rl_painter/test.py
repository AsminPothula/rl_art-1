import torch
import unittest
from deep_rl_painter.env.environment import PaintingEnvironment
from deep_rl_painter.models.ddpg import DDPG  # Or your chosen RL algorithm
from deep_rl_painter.config import Config  # Assuming you have a config.py
import lpips
import os

class TestDeepRLPainter(unittest.TestCase):
    """
    Unit tests for the Deep RL Painter project.
    """

    def setUp(self):
        """
        Set up common test objects.  This is run before each test.
        """
        self.config = Config()  # Or create a simplified config for testing
        self.config.target_image_path = 'deep_rl_painter/target.jpg'  # Make sure this exists
        self.config.image_size = (256, 256)
        self.config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config.model_name = "resnet" # Or set to a default
        self.config.actor_lr = 1e-4
        self.config.critic_lr = 1e-3

        # Create a dummy target image if it doesn't exist
        if not os.path.exists(self.config.target_image_path):
            dummy_target = torch.zeros(1, 3, self.config.image_size[0], self.config.image_size[1])
            torch.save(dummy_target, 'deep_rl_painter/target.jpg')  #changed from .pt to .jpg

        self.env = PaintingEnvironment(
            target_image_path=self.config.target_image_path,
            canvas_size=self.config.image_size,
            max_strokes=10,  # Keep this small for testing
            device=self.config.device
        )

        self.agent = DDPG(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
            actor_lr=self.config.actor_lr,
            critic_lr=self.config.critic_lr,
            gamma=0.99,
            tau=0.005,
            device=self.config.device,
            model_name=self.config.model_name,
            height=self.config.image_size[0],
            width=self.config.image_size[1],
            pretrained=False # Don't use pretrained for tests.
        )

    def tearDown(self):
        """
        Clean up after each test.  For example, delete any files created.
        """
        pass # No need to delete dummy target.pt


    def test_env_reset(self):
        """
        Test that the environment can be reset.
        """
        state = self.env.reset()
        self.assertIsInstance(state, torch.Tensor)
        self.assertEqual(state.shape, self.env.observation_space.shape)

    def test_env_step(self):
        """
        Test that the environment can take a step.
        """
        state = self.env.reset()
        action = self.env.action_space.sample()  # Get a random action
        reward_function = self.env.calculate_ssim_reward # Use a valid reward function.
        lpips_fn = None # Only needed for LPIPS
        next_state, reward, done, _ = self.env.step(action, reward_function, lpips_fn)

        self.assertIsInstance(next_state, torch.Tensor)
        self.assertEqual(next_state.shape, self.env.observation_space.shape)
        self.assertIsInstance(reward, torch.Tensor)
        self.assertEqual(reward.shape, (1,))  # Reward should be a scalar tensor
        self.assertIsInstance(done, bool)

    def test_agent_select_action(self):
        """
        Test that the agent can select an action.
        """
        state = self.env.reset()
        action = self.agent.select_action(state)
        self.assertIsInstance(action, torch.Tensor)
        self.assertEqual(action.shape, (self.env.action_space.shape[0],))

    def test_agent_update(self):
        """
        Test that the agent can update its parameters.
        """
        state = self.env.reset()
        action = self.env.action_space.sample()
        next_state, reward, done, _ = self.env.step(action, self.env.calculate_ssim_reward, None)
        #Populate the replay buffer
        self.agent.replay_buffer.push(state, action, next_state, reward, done)
        for _ in range(self.config.batch_size):
            self.agent.replay_buffer.push(next_state, action, next_state, reward, done) # fill up buffer

        # Check that update does not raise an exception
        try:
            self.agent.update(self.agent.replay_buffer, self.config.batch_size)
        except Exception as e:
            self.fail(f"agent.update() raised an exception: {e}")

    def test_reward_functions(self):
        """
        Test the reward functions.
        """
        prev_canvas = torch.randn(2, 3, self.config.image_size[0], self.config.image_size[1]).to(self.config.device)
        current_canvas = torch.randn(2, 3, self.config.image_size[0], self.config.image_size[1]).to(self.config.device)
        target_canvas = torch.randn(2, 3, self.config.image_size[0], self.config.image_size[1]).to(self.config.device)

        # Ensure data is in the range [0, 1] for LPIPS
        prev_canvas = (prev_canvas - prev_canvas.min()) / (prev_canvas.max() - prev_canvas.min())
        current_canvas = (current_canvas - current_canvas.min()) / (current_canvas.max() - current_canvas.min())
        target_canvas = (target_canvas - target_canvas.min()) / (target_canvas.max() - target_canvas.min())


        ssim_reward = self.env.calculate_ssim_reward(prev_canvas, current_canvas, target_canvas)
        self.assertIsInstance(ssim_reward, torch.Tensor)
        self.assertEqual(ssim_reward.shape, (2, 1))

        mse_reward = self.env.calculate_mse_reward(prev_canvas, current_canvas, target_canvas)
        self.assertIsInstance(mse_reward, torch.Tensor)
        self.assertEqual(mse_reward.shape, (2, 1))

        lpips_fn = lpips.LPIPS(net='vgg').to(self.config.device)
        lpips_reward = self.env.calculate_lpips_reward(prev_canvas, current_canvas, target_canvas, lpips_fn)
        self.assertIsInstance(lpips_reward, torch.Tensor)
        self.assertEqual(lpips_reward.shape, (2, 1))
    
    def test_action_space_sample(self):
        """
        Test that the action space can be sampled.
        """
        action = self.env.action_space.sample()
        self.assertIsInstance(action, torch.Tensor)
        self.assertEqual(action.shape, (self.env.action_space.shape[0],)) # Shape must be correct

if __name__ == '__main__':
    unittest.main()
