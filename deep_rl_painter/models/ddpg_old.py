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
        self.config = config                                         # initialize 

        self.actor_target.load_state_dict(actor.state_dict())        # sync target networks with main networks to begin with
        self.critic_target.load_state_dict(critic.state_dict())

    # get an action from the main actor network (for testing)
    def select_action(self, state):
        height, width = self.config["image_size"]
        channels = 3  # = 3 assuming grayscale canvas + x + y

        state = torch.FloatTensor(state).to(self.config["device"])
        state = state.view(1, channels, height, width)  # Reshape to 4D for CNN
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()
        return np.clip(action, -1, 1)

    # calls select_action, receives the action, adds exploration noise to it (for training)
    def act(self, state, noise_scale=0.0):
        action = self.select_action(state)
        action += self.noise.sample() * noise_scale
        return np.clip(action, -1, 1)

    # updates actor and critic using a batch of experiences
    """def update_actor_critic(self):
        if len(self.replay_buffer) < self.config["batch_size"]:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config["batch_size"])

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # critic network update 
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + self.config["gamma"] * target_Q * (1 - dones)     # target_Q (s, a) = r + γ * Q_target(s', a')

        current_Q = self.critic(states, actions)                                   # get current_Q value from the critic network                             
        critic_loss = F.mse_loss(current_Q, target_Q)                              # calculate the loss (difference)

        self.critic_optimizer.zero_grad()                                          # clear previous gradients
        critic_loss.backward()                                                     # calculate gradients of the loss wrt each of the weights
        self.critic_optimizer.step()                                               # update the weights using the gradients
 
        # actor update 
        
            #actor LOSS calculation - 
            #passes the batch of states to the actor network
            #    self.actor(states) --> predicted_actions 
            #self.critic(states, predicted_actions) --> outputs q values for each (s,a) pair
            #actor_loss = -q_values.mean() 
            #    want the actor to maximize the q values, but pytorch optimizers minimize loss functions by default, so take the negative mean 
        
        actor_loss = -self.critic(states, self.actor(states)).mean()            

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # calls soft_update for target networks 
        self.soft_update(self.critic, self.critic_target, self.config["tau"])"""
    def update_actor_critic(self):
        if len(self.replay_buffer) < self.config["batch_size"]:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config["batch_size"])

        batch_size = self.config["batch_size"]
        height, width = self.config["image_size"]
        channels = 3  # 1 grayscale + 2 position channels

        # Reshape state tensors if needed
        #states = torch.tensor(states, dtype=torch.float32).view(batch_size, channels, height, width).to(self.config["device"])
        #next_states = torch.tensor(next_states, dtype=torch.float32).view(batch_size, channels, height, width).to(self.config["device"])
        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.config["device"])
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(self.config["device"])
        
        actions = torch.tensor(actions, dtype=torch.float32).to(self.config["device"])
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.config["device"])
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.config["device"])

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + self.config["gamma"] * target_Q * (1 - dones)

        current_Q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        self.soft_update(self.critic, self.critic_target, self.config["tau"])
        self.soft_update(self.actor, self.actor_target, self.config["tau"])

    def train(self):
        # wrapper method for training — calls one update step
        self.update_actor_critic()

    def soft_update(self, local_model, target_model, tau):
        # soft update target network : θ_target ← τ * θ_local + (1 - τ) * θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)