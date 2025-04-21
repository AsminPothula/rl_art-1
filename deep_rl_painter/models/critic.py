import torch
import torch.nn as nn
import torchvision.models as models
import warnings

# Turn off all warnings
warnings.filterwarnings("ignore")

class Critic(nn.Module):
    def __init__(self, model_name, height, width, in_channels, action_dim, pretrained=True):
        """
        Initialize the Critic model.
        Args:
            model_name (str): Name of the base model architecture ('resnet', 'efficientnet', 'cae' : Custom).
            height (int): Height of the input image.
            width (int): Width of the input image.
            in_channels (int): Number of input channels (if 1: grayscale canvas).
            Or I guess we can keep it just one channel, and use the action dims for other inputs.
            action_dim (int): Dimension of the action space. (included in the in_channels). 
            if 8 in this case: 4 for x1,y1,x2,y2,r,g,b, width.
            if 4 for x1,y1,x2,y2
            pretrained (bool): Whether to use a pretrained model (default: True) for resnet/efficientnet.
        """
        super(Critic, self).__init__()
        self.model_name = model_name
        self.in_channels = in_channels
        self.action_dim = action_dim
        self.height = height
        self.width = width
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        if model_name == "resnet":
            self.feature_extractor = models.resnet18(pretrained=pretrained)
            self.feature_extractor.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Remove the final fully connected layer
            num_features = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Identity()
            self.fc = nn.Sequential(
                nn.Linear(num_features + action_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 1)  # Output a single Q-value
            )
        elif model_name == "efficientnet":
            self.feature_extractor = models.efficientnet_b0(pretrained=pretrained)
            self.feature_extractor.features[0][0] = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
            # Remove the final classifier layer
            num_features = self.feature_extractor.classifier[1].in_features
            self.feature_extractor.classifier = nn.Identity()
            self.fc = nn.Sequential(
                nn.Linear(num_features + action_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 1)  # Output a single Q-value
            )
        elif model_name == "cae":
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
            # Calculate the output size of the convolutional layers
            with torch.no_grad():
                dummy_input = torch.randn(1, in_channels, height, width)
                features_dim = self.feature_extractor(dummy_input).shape[1]
            self.fc = nn.Sequential(
                nn.Linear(features_dim + action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)  # Output a single Q-value
            )
        else:
            raise ValueError(f"Invalid model name: {model_name}. Choose from 'resnet', 'efficientnet', or 'cae'.")

    def forward(self, state, action):
        if self.model_name in ["resnet", "efficientnet"]:
            features = self.feature_extractor(state)
            features = torch.flatten(features, 1)
        elif self.model_name == "cae":
            features = self.feature_extractor(state)
        else:
            raise NotImplementedError

        # Concatenate the features and the action
        combined_input = torch.cat([features, action], dim=1)
        q_value = self.fc(combined_input)
        return q_value

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))


if __name__ == '__main__':
    model_name = "resnet"
    height = 64
    width = 64
    in_channels = 3
    in_channels = 1
    action_dim = 8
    batch_size = 2

    critic = Critic(model_name, height, width, in_channels, action_dim)
    print(f"Critic ({model_name}): {critic}")

    state = torch.randn(batch_size, in_channels, height, width)
    action = torch.randn(batch_size, action_dim)
    q_value = critic(state, action)
    print(f"Q-value output shape: {q_value.shape}\nExample: {q_value}")

    save_path = "critic_test.pth"
    critic.save_model(save_path)
    loaded_critic = Critic(model_name, height, width, in_channels, action_dim, pretrained=False)
    loaded_critic.load_model(save_path)
    loaded_q_value = loaded_critic(state, action)
    assert torch.allclose(q_value, loaded_q_value, atol=1e-6)
    print(f"Model saved and loaded successfully from {save_path}.")