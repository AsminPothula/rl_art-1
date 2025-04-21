import torch
import torch.nn.functional as F
import numpy as np

class DDPGAgent:
    def __init__(self, actor, critic, actor_target, critic_target,
                 actor_optimizer, critic_optimizer, replay_buffer, noise, config, channels):
        self.actor = actor
        self.critic = critic
        self.actor_target = actor_target
        self.critic_target = critic_target
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.replay_buffer = replay_buffer
        self.noise = noise
        self.config = config
        self.channels = channels

        self.actor_target.load_state_dict(actor.state_dict())
        self.critic_target.load_state_dict(critic.state_dict())

    def select_action(self, state):
        height, width = self.config["image_size"]
        channels = self.channels # 2  # grayscale
        state = torch.FloatTensor(state).to(self.config["device"]).view(1, channels, height, width)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()
        return np.clip(action, -1, 1)

    def act(self, state, noise_scale=0.0):
        action = self.select_action(state)
        action += self.noise.sample() * noise_scale
        return np.clip(action, -1, 1)

    def update_actor_critic(self):
        if len(self.replay_buffer) < self.config["batch_size"]:
            return

        # Load training parameters
        batch_size = self.config["batch_size"]
        height, width = self.config["image_size"]
        channels = 3  # canvas + x + y positional channels
        flat_state_dim = channels * height * width

        # Sample experience batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.config["device"])
        actions = torch.tensor(actions, dtype=torch.float32).to(self.config["device"])
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.config["device"])
        # print(f"[DEBUG] next_states shape: {next_states.shape}")
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.config["device"])
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.config["device"])

        # Reshape for CNN-based actor input
        if states.shape[1] != flat_state_dim:
            raise ValueError(f"Expected states shape to be ({batch_size}, {flat_state_dim}), got {states.shape}")
        if next_states.shape[1] != flat_state_dim:
            raise ValueError(f"Expected next_states shape to be ({batch_size}, {flat_state_dim}), got {next_states.shape}")

        states_reshaped = states.view(batch_size, channels, height, width)
        next_states_reshaped = next_states.view(batch_size, channels, height, width)

        # --- Critic update ---
        with torch.no_grad():
            next_actions = self.actor_target(next_states_reshaped)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + self.config["gamma"] * target_Q * (1 - dones)

        current_Q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        predicted_actions = self.actor(states_reshaped)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Target Network Soft Update ---
        self.soft_update(self.critic, self.critic_target, self.config["tau"])
        self.soft_update(self.actor, self.actor_target, self.config["tau"])


    def train(self):
        self.update_actor_critic()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
