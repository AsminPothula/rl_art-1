import os
import torch
import numpy as np
import argparse
import lpips


from config import config
from env.environment import PaintingEnv
from models.actor import Actor
from models.critic import Critic
from models.ddpg import DDPGAgent
from utils.replay_buffer import ReplayBuffer
from utils.noise import OUNoise  


def main(config):
    print("Initializing environment...")
    env = PaintingEnv(
        target_image_path=config["target_image_path"],
        canvas_size=config["image_size"],
        max_strokes=config["max_steps"],
        device=config["device"]
    )
    # since we are currently using target_image_1 (config.py) - change accordingly
    output_dir = os.path.join("target_image_outputs", "target_output_1")

    print("Building actor & critic networks...")
    height, width = config["image_size"]
    in_channels = 2  # grayscale canvas input
    action_dim = env.action_space.shape[0]
    #state_dim = env.observation_space.shape[0]
    state_dim = in_channels * height * width  # = 2 * 256 * 256

    # Actor and Target Actor
    actor = Actor(config["model_name"], height, width, in_channels=in_channels,
                  out_neurons=action_dim, pretrained=True).to(config["device"])
    actor_target = Actor(config["model_name"], height, width, in_channels=in_channels,
                         out_neurons=action_dim, pretrained=True).to(config["device"])

    # Critic and Target Critic
    # Critic(model_name, height, width, in_channels, action_dim)
    critic = Critic(config["model_name"],height, width,state_dim, action_dim).to(config["device"])
    critic_target = Critic(config["model_name"],height, width,state_dim, action_dim).to(config["device"])
    print(f"[INFO] state_dim = {state_dim}, action_dim = {action_dim}")

    # Optimizers
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config["actor_lr"])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config["critic_lr"])

    # Noise and Buffer
    noise = OUNoise(action_dim)
    replay_buffer = ReplayBuffer(config["replay_buffer_size"])

    print("Initializing agent...")
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
        channels=in_channels
    )

    from env.reward import (
    calculate_ssim_reward,
    calculate_mse_reward,
    calculate_lpips_reward
    )

    # Reward function setup
    if config["reward_function_name"] == 'ssim':
        reward_function = calculate_ssim_reward
        lpips_fn = None
    elif config["reward_function_name"] == 'mse':
        reward_function = calculate_mse_reward
        lpips_fn = None
    elif config["reward_function_name"] == 'lpips':
        reward_function = calculate_lpips_reward
        lpips_fn = lpips.LPIPS(net='vgg').to(config["device"])
    else:
        raise ValueError(f"Invalid reward function: {config['reward_function_name']}")

    print("Starting training loop...")
    for episode in range(config["num_episodes"]):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done:
            action = agent.act(state, noise_scale=config["noise_std"])
            next_state, reward, done = env.step(action, reward_function, lpips_fn)
            # Render every 5th episode - change accordingly
            if (episode + 1) % 5 == 0:
                env.render(episode_num=episode + 1, output_dir=output_dir)
            # Flatten and convert everything before storing
            state_np = state.flatten() if isinstance(state, np.ndarray) else state.cpu().numpy().flatten()
            next_state_np = next_state.flatten() if isinstance(next_state, np.ndarray) else next_state.cpu().numpy().flatten()
            action_np = np.array(action, dtype=np.float32).flatten()
            reward_scalar = float(reward)
            done_scalar = float(done)

            replay_buffer.store(state_np, action_np, reward_scalar, next_state_np, done_scalar)

            state = next_state
            episode_reward += reward
            step += 1

            if len(replay_buffer) > config["batch_size"]:
                agent.train()

        print(f"Episode {episode + 1}, Reward: {episode_reward:.3f}, Steps: {step}")

        if (episode + 1) % config["save_every"] == 0:
            agent.save_model(f"checkpoint_{episode + 1}.pth")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep RL Painter")
    parser.add_argument('--model', type=str, default='resnet', help='Model: resnet, efficientnet, cae')
    parser.add_argument('--reward', type=str, default='ssim', help='Reward: ssim, mse, lpips')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--target_image', type=str, default='target_images/target_image_1.jpg')
    args = parser.parse_args()

    # Update config from CLI
    config["model_name"] = args.model
    config["reward_function_name"] = args.reward
    config["batch_size"] = args.batch_size
    config["num_episodes"] = args.episodes
    config["target_image_path"] = args.target_image

    # Static config settings
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["image_size"] = (256, 256)
    config["replay_buffer_size"] = 100000
    config["actor_lr"] = 1e-4
    config["critic_lr"] = 1e-3
    config["gamma"] = 0.99
    config["tau"] = 0.005
    config["noise_std"] = 0.1

    print("Running with configuration:")
    for k, v in config.items():
        print(f"{k}: {v}")

    main(config)

