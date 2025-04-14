import torch
import torch.optim as optim
from deep_rl_painter.config import Config  # Assuming you have a config.py
from deep_rl_painter.env.environment import PaintingEnvironment
from deep_rl_painter.models.ddpg import DDPG  # Or your chosen RL algorithm
from deep_rl_painter.utils.replay_buffer import ReplayBuffer
import lpips
import argparse
import os

def main(config):
    """
    Main function to run the Deep RL Painter training process.
    """

    # Initialize the environment
    env = PaintingEnvironment(
        target_image_path=config.target_image_path,
        canvas_size=config.image_size,
        max_strokes=config.max_strokes,
        device=config.device
    )

    # Initialize the agent
    agent = DDPG(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        tau=config.tau,
        device=config.device,
        model_name=config.model_name,  # Pass the model name.
        height=config.image_size[0],
        width=config.image_size[1],
        pretrained=True
    )

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(config.replay_buffer_size)

    # Initialize reward function
    if config.reward_function_name == 'ssim':
        reward_function = env.calculate_ssim_reward
        lpips_fn = None
    elif config.reward_function_name == 'mse':
        reward_function = env.calculate_mse_reward
        lpips_fn = None
    elif config.reward_function_name == 'lpips':
        reward_function = env.calculate_lpips_reward
        lpips_fn = lpips.LPIPS(net='vgg').to(config.device)  # Initialize LPIPS *once* and move to device
    else:
        raise ValueError(f"Invalid reward function: {config.reward_function_name}")

    # Main training loop
    for episode in range(config.num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done:
            # Select action
            action = agent.select_action(state)
            action = agent.add_noise(action, std=config.noise_std)  # Add noise.
            next_state, reward, done, _ = env.step(action, reward_function, lpips_fn)  # Pass lpips_fn
            replay_buffer.push(state, action, next_state, reward, done)
            state = next_state
            episode_reward += reward
            step += 1

            # Update agent
            if len(replay_buffer) > config.batch_size:
                agent.update(replay_buffer, config.batch_size)

        print(f"Episode {episode}, Reward: {episode_reward:.3f}, Steps: {step}")

        # Save model periodically
        if (episode + 1) % 100 == 0:
            agent.save_model(f"checkpoint_{episode + 1}.pth")
    env.close()



if __name__ == "__main__":
    # Use argparse to get configuration
    parser = argparse.ArgumentParser(description="Deep RL Painter")
    parser.add_argument('--model', type=str, default='resnet', help='Model architecture (resnet, efficientnet, cae)')
    parser.add_argument('--reward', type=str, default='ssim', help='Reward function (ssim, mse, lpips)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--target_image', type=str, default='deep_rl_painter/target.jpg', help='Path to the target image')
    args = parser.parse_args()

    # Create a config object
    config = Config()
    config.model_name = args.model
    config.reward_function_name = args.reward
    config.batch_size = args.batch_size
    config.num_episodes = args.episodes
    config.target_image_path = args.target_image # Get target image path from args

    # Set other config parameters.  These could also come from argparse if needed.
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.image_size = (256, 256)
    config.replay_buffer_size = 100000
    config.actor_lr = 1e-4
    config.critic_lr = 1e-3
    config.gamma = 0.99
    config.tau = 0.005
    config.noise_std = 0.1
    #config.target_image_path = 'deep_rl_painter/target.jpg'  # Or make this a command line arg


    print(f"Running with configuration: {config.__dict__}")  # Print configuration
    main(config)