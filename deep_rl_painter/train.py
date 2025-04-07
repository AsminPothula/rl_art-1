import torch
import numpy as np
from collections import deque
from env.environment import PaintingEnv
from models.actor import Actor
from models.critic import Critic
from models.ddpg import DDPGAgent
from utils.replay_buffer import ReplayBuffer
from utils.noise import OUNoise
from config import config
import os

# set seeds
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

# initialize environment
env = PaintingEnv(
    image_path=config["target_image"],
    stroke_length=config["stroke_length"],                   # fixed stroke length 
    error_threshold=config["error_threshold"],               # for minimum error for episode to end
    max_total_length=config["max_total_length"]              # total string length (for episode to end)
)

state_dim = env.observation_space.shape[0]                   # flattened canvas image
action_dim = 2                                               # continuous x and y direction

# initialize actor and critic networks
actor = Actor(state_dim, action_dim)                         # main actor and critic do the learning
critic = Critic(state_dim, action_dim)
actor_target = Actor(state_dim, action_dim)                  # target networks are slow-moving copies used to stabilize training
critic_target = Critic(state_dim, action_dim)
actor_target.load_state_dict(actor.state_dict())             # sync the targets with the main networks initially
critic_target.load_state_dict(critic.state_dict())

# optimizers - these update the network weights during training
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config["actor_lr"])
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config["critic_lr"])

# replay buffer and noise process
replay_buffer = ReplayBuffer(config["buffer_size"])          # stores (state, action, reward, next_state)
noise = OUNoise(action_dim)                                  # adds random noise to encourage exploration

# create agent
agent = DDPGAgent(
    actor, critic,
    actor_target, critic_target,
    actor_optimizer, critic_optimizer,
    replay_buffer, noise,
    config
)

# logging
os.makedirs("logs", exist_ok=True)                           # trained models are saved
scores_window = deque(maxlen=100)                            # keeps the last 100 scores for moving average - np.mean(scores_window)
scores = []

# exploration noise scale - start with a larger noise, decay over time
noise_scale = config["initial_noise_scale"]
noise_decay = config["noise_decay"]

# training loop
for episode in range(config["episodes"]):
    state = env.reset()                                      # reset to blank canvas
    episode_reward = 0                                       # reset reward to 0 

    for step in range(config["max_steps"]):
        action = agent.act(state, noise_scale)               # use agent.act to get (x,y) with exploration - select_action(state) = no noise - use for testing
        next_state, reward, done, _ = env.step(action)       # call step function and get other values

        agent.replay_buffer.store(state, action, reward, next_state, done)           # store values
        agent.train()                                                                # train agent based on the values

        state = next_state                                   # make next state the current state 
        episode_reward += reward                             # add up each action's reward to get total episode reward

        if done:
            break

    # decay noise
    noise_scale *= noise_decay

    # total episode reward 
    scores.append(episode_reward)
    scores_window.append(episode_reward)
    print(f"Episode {episode+1} | Reward: {episode_reward:.2f} | Avg(100): {np.mean(scores_window):.2f}")

    # save model periodically
    if (episode + 1) % config["save_every"] == 0:
        torch.save(actor.state_dict(), f"logs/actor_{episode+1}.pth")
        torch.save(critic.state_dict(), f"logs/critic_{episode+1}.pth")

print("Training complete.")