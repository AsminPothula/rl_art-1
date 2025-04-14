# needs to be reviewed - add proper comments 
import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        # First layer takes state and action as input (concatenated)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Outputs a single Q-value
        )

    def forward(self, state, action):
        """
        Forward pass through the critic network.
        Concatenate state and action, then predict Q-value.
        """
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        x = torch.cat([state, action], dim=1)  # Combine state and action
        return self.net(x)  # Output: Q-value
