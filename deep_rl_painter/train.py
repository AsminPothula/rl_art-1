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

TODO:
    - FIX THE DDPG AGENT
    - Add logging for training progress
    - Add hyperparameter tuning options
    - Consider using a learning rate scheduler for optimizers
"""
import os
import torch
import numpy as np
from collections import deque
import pandas as pd
import csv
from env.environment import PaintingEnv
from models.actor import Actor
from models.critic import Critic
from models.ddpg import DDPGAgent
from utils.noise import OUNoise
from utils.replay_buffer import ReplayBuffer

from env.canvas import save_canvas



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
        canvas_size=config["canvas_size"],
        canvas_channels=config["canvas_channels"],
        max_strokes=config["max_strokes"],
        device=config["device"]
    )

    # Load target image
    # target_image needs to be of dimensions (batch=1, channels, height, width)
    target_image = torch.tensor(env.target_image).unsqueeze(0).to(config["device"])

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
        config=config
    )

    # Setup logging
    os.makedirs("logs", exist_ok=True)

    scores_window = deque(maxlen=100) # keep track of last 100 episodes, more stable than 3 

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
            # Save step frame every 50th stroke for select episodes
            if (episode + 1) in [1, 1000, 10000, 25000, 50000] and env.current_step % config["save_every_step"] == 0:
                step_dir = f"step_outputs/"
                os.makedirs(step_dir, exist_ok=True)
                step_canvas = (canvas.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
                if config["canvas_channels"] == 1:
                    step_canvas = step_canvas[0]
                elif config["canvas_channels"] == 3:
                    step_canvas = np.transpose(step_canvas, (1, 2, 0))  # (C, H, W) → (H, W, C)
                # sample path: step_outputs/episode_25000/episode_25000_step_00150.png
                save_path = os.path.join(step_dir, f"episode_{episode + 1}_step_{env.current_step}.png")
                save_canvas(step_canvas, save_path)
            prev_action = action
            episode_reward += reward

            # Log step reward
            with open("logs/step_rewards.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                if episode == 0 and env.current_step == 1:  # write header only once
                    writer.writerow(["episode", "step", "reward"])
                writer.writerow([episode + 1, env.current_step, round(float(reward), 4)])
                print(f"Episode {episode + 1} | Step {env.current_step} | Step Reward: {reward}")


        # Decay exploration noise
        noise_scale *= noise_decay
        scores.append(episode_reward)
        # Log episode reward
        with open("logs/episode_rewards.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            if episode == 0:  # Write header only once
                writer.writerow(["episode", "total_reward"])
            writer.writerow([episode + 1, round(float(episode_reward), 4)])
        scores_window.append(episode_reward)

        # Progress log
        print(
            f"Episode {episode + 1} | Reward: {episode_reward} | Running Avg(100): {np.mean(scores_window)}")




        # Save model checkpoints every 100 episodes
        if (episode + 1) % config["save_every_episode"] == 0:
            os.makedirs("trained_models", exist_ok=True)
            torch.save(actor.state_dict(),
                    f"trained_models/actor_{episode + 1}.pth")
            torch.save(critic.state_dict(),
                    f"trained_models/critic_{episode + 1}.pth")
            print(f"Saved model at episode {episode + 1}")

    print("Training complete.")
