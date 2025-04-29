import os
import torch
import numpy as np
import lpips

from env.environment import PaintingEnv
from models import Actor
from models.critic import Critic
from models.ddpg import DDPGAgent
from utils.replay_buffer import ReplayBuffer
from utils.noise import OUNoise
from env.reward import calculate_ssim_reward, calculate_mse_reward, calculate_lpips_reward

def train(config):
    print("Initializing environment...")
    env = PaintingEnv(
        target_image_path=config["target_image_path"],
        canvas_size=config["image_size"],
        max_strokes=config["max_steps"],
        device=config["device"]
    )

    output_dir = os.path.join("target_image_outputs", "target_output_1")
    os.makedirs(output_dir, exist_ok=True)

    in_channels = 3
    action_dim = 6

    actor = Actor(
        image_encoder_model=config["model_name"],
        image_encoder_model_2=config["model_name"],
        pretrained=True,
        out_neurons=action_dim,
        in_channels=in_channels
    )

    actor_target = Actor(
        image_encoder_model=config["model_name"],
        image_encoder_model_2=config["model_name"],
        pretrained=True,
        out_neurons=action_dim,
        in_channels=in_channels
    )

    critic = Critic(
        image_encoder_model=config["model_name"],
        image_encoder_model_2=config["model_name"],
        pretrained=True,
        out_neurons=1,
        in_channels=in_channels
    )

    critic_target = Critic(
        image_encoder_model=config["model_name"],
        image_encoder_model_2=config["model_name"],
        pretrained=True,
        out_neurons=1,
        in_channels=in_channels
    )

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config["actor_lr"])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config["critic_lr"])
    noise = OUNoise(action_dim)
    replay_buffer = ReplayBuffer(config["replay_buffer_size"])

    agent = DDPGAgent(
        actor, critic,
        actor_target, critic_target,
        actor_optimizer, critic_optimizer,
        replay_buffer, noise,
        config, in_channels
    )

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

            state_np = state.flatten()
            next_state_np = next_state.flatten()
            action_np = np.array(action, dtype=np.float32).flatten()
            replay_buffer.store(state_np, action_np, float(reward), next_state_np, float(done))

            state = next_state
            episode_reward += reward
            step += 1

            if len(replay_buffer) > config["batch_size"]:
                agent.train()

            if (episode + 1) % 5 == 0:
                env.render(episode_num=episode + 1, output_dir=output_dir)

        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}, Steps: {step}")

        if (episode + 1) % config["save_every"] == 0:
            torch.save(actor.state_dict(), os.path.join(output_dir, f"actor_{episode+1}.pth"))
            torch.save(critic.state_dict(), os.path.join(output_dir, f"critic_{episode+1}.pth"))

    env.close()
