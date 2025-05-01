"""
Training Script

Handles the training loop for the DDPG painting agent. It:
- Initializes the environment and models
- Runs episodes where the agent paints using predicted actions
- Stores experiences (canvas, prev_action, action, next_canvas, reward, done)
- Periodically updates the Actor and Critic using replay buffer samples
- Applies soft target updates
- Logs training progress and saves checkpoints

Inputs:
    - Canvas (current canvas image)
    - Target image (fixed goal)
    - Previous action (used to generate the next one)

Outputs:
    - Trained Actor and Critic models saved to disk
"""

import torch
import numpy as np
import os
from collections import deque
from env.environment import PaintingEnv
from models.actor import Actor
from models.critic import Critic
from models.ddpg import DDPGAgent
from utils.replay_buffer import ReplayBuffer
from utils.noise import OUNoise


def train(config):
    """
    Full training pipeline for the DDPG agent using canvas drawing environment.
    Initializes all components and trains the agent over multiple episodes.

    Args:
        config (dict): Configuration dictionary containing hyperparameters and paths.
    """

    # Initialize environment and load target image
    env = PaintingEnv(
        target_image_path=config["target_image_path"],
        canvas_size=config["image_size"],
        canvas_channels=config["canvas_channels"],
        max_strokes=config["max_steps"],
        device=config["device"]
    )

    # don't know why this is needed
    target_image = env.get_target_tensor().to(config["device"])

    # Initialize Actor & Critic networks (main and target)
    actor = Actor(
        image_encoder_model=config["model_name"],
        actor_network_input=config["action_dim"],
        in_channels=config["canvas_channels"],
        out_neurons=config["action_dim"]
    )
    critic = Critic(
        image_encoder_model=config["model_name"],
        actor_network_input=config["action_dim"]*2,
        in_channels=config["canvas_channels"],
        out_neurons=1
    )
    actor_target = Actor(
        image_encoder_model=config["model_name"],
        actor_network_input=config["action_dim"],
        out_neurons=config["action_dim"],
        in_channels=config["canvas_channels"]
    )
    critic_target = Critic(
        image_encoder_model=config["model_name"],
        actor_network_input=config["action_dim"]*2,
        in_channels=config["canvas_channels"],
        out_neurons=1
    )

    # Sync target networks
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    # Set optimizers
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=config["actor_lr"])
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config["critic_lr"])

    # Initialize replay buffer and noise process
    replay_buffer = ReplayBuffer(config["buffer_size"])
    noise = OUNoise(config["action_dim"])

    # THE DDPG AGENT IS FREAKING WRONG!!
    # Build the DDPG agent
    agent = DDPGAgent(
        actor=actor,
        critic=critic,
        actor_target=actor_target,
        critic_target=critic_target,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        replay_buffer=replay_buffer,
        noise=noise,
        config=config,
        channels=config["canvas_channels"]
    )

    # Setup logging
    os.makedirs("logs", exist_ok=True)
    scores_window = deque(maxlen=100)
    scores = []

    # Exploration noise control
    noise_scale = config["initial_noise_scale"]
    noise_decay = config["noise_decay"]

    # Main training loop
    for episode in range(config["episodes"]):
        canvas = env.reset()
        prev_action = np.zeros(config["action_dim"], dtype=np.float32)
        # Initialize previous action with current point
        prev_action[0] = env.current_point[0]
        prev_action[1] = env.current_point[1]
        episode_reward = 0
        done = False

        # Episode step loop
        while not done:
            # Choose action using current state + exploration
            action = agent.act(canvas, target_image, prev_action, noise_scale)

            # Apply action in the environment
            next_canvas, reward, done = env.step(action)

            # Store experience in replay buffer
            replay_buffer.store(canvas, prev_action, action,
                                next_canvas, reward, done)

            # Train the agent using sampled experiences
            agent.train(target_image)

            # Move to next state
            canvas = next_canvas
            prev_action = action
            episode_reward += reward

        # Decay exploration noise
        noise_scale *= noise_decay
        scores.append(episode_reward)
        scores_window.append(episode_reward)

        # Progress log
        print(
            f"Episode {episode + 1} | Reward: {episode_reward:.2f} | Running Avg(100): {np.mean(scores_window):.2f}")

        # Periodically save model checkpoints
        if (episode + 1) % config["save_every"] == 0:
            torch.save(actor.state_dict(), f"trained_models/actor_{episode + 1}.pth")
            torch.save(critic.state_dict(), f"trained_models/critic_{episode + 1}.pth")

    print("Training complete.")

