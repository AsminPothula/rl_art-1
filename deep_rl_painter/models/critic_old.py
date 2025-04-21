# needs to be reviewed - add proper comments 
import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),  # this needs to be 196610
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
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

        print(f"State shape: {state.shape}")
        print(f"Action shape: {action.shape}")

        x = torch.cat([state, action], dim=1)  # Combine state and action
        print(f"Concatenated input shape: {x.shape}")

        return self.net(x)  # Output: Q-value

if __name__ == "__main__":
    # Test the Critic class
    state_dim = 4  # Example state dimension
    action_dim = 6  # Example action dimension
    hidden_dim = 256  # Example hidden dimension

    # Create a Critic instance
    critic = Critic(state_dim, action_dim, hidden_dim)

    # Example inputs
    state = torch.randn(state_dim)  # Random state tensor
    action = torch.randn(action_dim)  # Random action tensor

    # Forward pass
    q_value = critic(state, action)
    print(f"Predicted Q-value: {q_value.item()}")
