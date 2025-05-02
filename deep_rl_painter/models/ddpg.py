"""
DDPG Agent Module

Defines the DDPGAgent class for training and action inference in a goal-conditioned painting task.

The agent:
    - Uses dual CNN encoders to process canvas and target images.
    - Takes previous actions into account to maintain temporal coherence.
    - Trains Actor and Critic networks using transitions from a replay buffer.

Key Inputs:
    - canvas: current image
    - target_image: goal image to paint
    - prev_action: the last action applied to canvas
    - action: the action to evaluate or apply

Outputs:
    - Actor: next action to apply (x, y, r, g, b, width)
    - Critic: Q-value estimating expected return from a (state, action) pair
"""

import torch
import torch.nn.functional as F
import numpy as np

class DDPGAgent:
    def __init__(self, actor, critic, actor_target, critic_target,
                 actor_optimizer, critic_optimizer, replay_buffer, noise, config):
        self.actor = actor
        self.critic = critic
        self.actor_target = actor_target
        self.critic_target = critic_target
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.replay_buffer = replay_buffer
        self.noise = noise
        self.config = config
        self.channels = config["canvas_channels"]
        self.device = config["device"]

        # Initialize target networks with the same weights as the main networks
        self.actor_target.load_state_dict(actor.state_dict())
        self.critic_target.load_state_dict(critic.state_dict())

    def select_action(self, canvas, target_image, prev_action):
        """
        Select an action from the actor network given the current state.
        Used in test.py

        Args:
            canvas (np.ndarray): Current canvas.
            target_image (np.ndarray): Target image.
            prev_action (np.ndarray): Action previously applied to canvas.

        Returns:
            action (np.ndarray): Next action predicted by actor network.
        """
        device = self.config["device"]
        canvas = torch.FloatTensor(canvas).to(device).unsqueeze(0)
        target_image = torch.FloatTensor(target_image).to(device)
        prev_action = torch.FloatTensor(prev_action).to(device).unsqueeze(0)

        self.actor.eval()
        with torch.no_grad():
            out = self.actor(canvas, target_image, prev_action).cpu().numpy()
            action = out[0]  # Remove batch dimension
        self.actor.train()
        return action

    def act(self, canvas, target_image, prev_action, noise_scale=0.01):
        """
        Select an action and apply Ornstein-Uhlenbeck exploration noise.
        Used in train.py
        Args:
            canvas (np.ndarray): Current canvas. Dimensions: (H, W, C)
            target_image (np.ndarray): Target image. Dimensions: (H, W, C)
            prev_action (np.ndarray): Action previously applied to canvas. Dimensions: (batch,action_dim)
            noise_scale (float): Scale of the noise to be added. 
        Used to control exploration.


        Returns:
            action (np.ndarray): Noisy action for exploration.
        """
        action = self.select_action(canvas, target_image, prev_action)
        action += self.noise.sample() * noise_scale
        return action

    def update_actor_critic(self, target_image):
        """
        Perform one training update step for both the actor and critic networks using
        a mini-batch sampled from the replay buffer.

        The update includes:
            - Computing target Q-values using the target networks
            - Calculating critic loss (MSE between current Q and target Q)
            - Updating the critic network via backprop
            - Generating predicted actions from the actor
            - Calculating actor loss (negative mean Q-value)
            - Updating the actor network via backprop
            - Soft-updating the target networks (Polyak averaging)

        Args:
            target_image (np.ndarray): The fixed goal image for the episode.
                                    Used in both actor and critic inputs.
        
        Notes:
            - The state is represented as (canvas, prev_action)
            - target_image is passed separately (assumed fixed per episode)
            - next state's prev_action = action from the current transition
        """
        if len(self.replay_buffer) < self.config["batch_size"]:
            return

        B = self.config["batch_size"]
        device = self.config["device"]

        canvas, prev_actions, actions, next_canvas, rewards, dones = self.replay_buffer.sample(B)

        canvas = torch.tensor(canvas, dtype=torch.float32).to(device)
        target = torch.tensor(target_image, dtype=torch.float32).to(device).repeat(canvas.shape[0], 1, 1, 1)
        prev_actions = torch.tensor(prev_actions, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        next_canvas = torch.tensor(next_canvas, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        with torch.no_grad():
            next_prev_actions = actions
            next_actions = self.actor_target(next_canvas, target, next_prev_actions)
            critic_actions = torch.cat((next_prev_actions, next_actions), dim=1)
            # target_Q = self.critic_target(next_canvas, target, next_prev_actions, next_actions) - not passing next_prev_action
            target_Q = self.critic_target(next_canvas, target, critic_actions)
            target_Q = rewards + self.config["gamma"] * target_Q * (1 - dones)

        # current_Q = self.critic(canvas, target, prev_actions, actions) - not passing prev_action
        current_Q = self.critic(canvas, target, critic_actions)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        predicted_actions = self.actor(canvas, target, prev_actions)
        predicted_actions = torch.cat((prev_actions, predicted_actions), dim=1)
        # actor_loss = -self.critic(canvas, target, prev_actions, predicted_actions).mean()
        actor_loss = -self.critic(canvas, target, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic, self.critic_target, self.config["tau"])
        self.soft_update(self.actor, self.actor_target, self.config["tau"])

    def train(self, target_image):
        """Wrapper to update actor and critic networks."""
        self.update_actor_critic(target_image)

    def soft_update(self, local_model, target_model, tau):
        """
        Perform soft target network update.

        Args:
            local_model: Actor or Critic main network.
            target_model: Corresponding target network.
            tau (float): Soft update coefficient (0 < tau << 1).
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

if __name__ == "__main__":
    """
    Quick diagnostic test for DDPGAgent setup and API:
    - Builds dummy Actor/Critic
    - Runs .act() and .train() with fake canvas data
    - Verifies integration works without crashing
    """
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    import numpy as np
    from .actor import Actor
    from .critic import Critic
    from ..utils.replay_buffer import ReplayBuffer
    from ..utils.noise import OUNoise

    print("Running DDPGAgent standalone integration test...")

    # Config setup
    config = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "image_size": (224, 224),
        "batch_size": 4,
        "gamma": 0.99,
        "tau": 0.005
    }

    B, C, H, W = config["batch_size"], 3, *config["image_size"]
    action_dim = 6

    # Models
    actor = Actor("resnet18", "resnet18", pretrained=False, out_neurons=action_dim, in_channels=C)
    critic = Critic("resnet18", "resnet18", pretrained=False, out_neurons=1, in_channels=C)
    actor_target = Actor("resnet18", "resnet18", pretrained=False, out_neurons=action_dim, in_channels=C)
    critic_target = Critic("resnet18", "resnet18", pretrained=False, out_neurons=1, in_channels=C)

    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # Agent and utils
    replay_buffer = ReplayBuffer(capacity=100)
    noise = OUNoise(action_dim)

    agent = DDPGAgent(
        actor=actor,
        critic=critic,
        actor_target=actor_target,
        critic_target=critic_target,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        replay_buffer=replay_buffer,
        noise=noise,
        config=config,
        channels=C
    )

    # Dummy tensors
    canvas = np.random.rand(C, H, W).astype(np.float32)
    target = np.random.rand(C, H, W).astype(np.float32)
    prev_action = np.zeros(action_dim, dtype=np.float32)

    # Test .act()
    try:
        action = agent.act(canvas, target, prev_action)
        print("DDPGAgent.act() output:", action)
    except Exception as e:
        print("DDPGAgent.act() failed:", e)

    # Populate buffer
    for _ in range(B):
        next_canvas = np.random.rand(C, H, W).astype(np.float32)
        reward = 1.0
        done = 0.0
        replay_buffer.store(canvas, prev_action, action, next_canvas, reward, done)

    # Test .train()
    try:
        agent.train(target)
        print("DDPGAgent.train() ran without error.")
    except Exception as e:
        print("DDPGAgent.train() failed:", e)
