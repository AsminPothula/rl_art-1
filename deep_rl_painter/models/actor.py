# Actor = Actor(state_dim, action_dims)   - in train.py

""" Input: [state_dim]
→ Hidden layers
→ Output: [action_dim] = 2 (x and y direction) """

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary

class Actor(nn.Module):
    def __init__(self, model_name, height, width, in_channels=1, out_channels=2, pretrained=True):
        super(Actor, self).__init__()
        self.model_name = model_name
        self.in_channels = in_channels + 2  # 2 for x and y direction
        self.out_channels = out_channels
        self.height = height
        self.width = width

        if model_name == "resnet":
            self.model = models.resnet18(pretrained=pretrained)
            self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.out_channels)  #
        elif model_name == "efficientnet":
            self.model = models.efficientnet_b0(pretrained=pretrained)
            self.model.features[0][0] = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.out_channels)  #
        elif model_name == "cae":
            self.model = nn.Sequential(
                nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Invalid model name: {model_name}. Choose from 'resnet', 'efficientnet', or 'cae'.")

    def forward(self, x):
        if self.model_name in ["resnet", "efficientnet"]:
            #out = self.model(x).view(x.size(0), self.out_channels, self.height, self.width)
            out = self.model(x)
        elif self.model_name == "cae":
            out = self.model(x)
        return out

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


# ✅ Example/test code (runs only when script is executed directly)
if __name__ == "__main__":
    batch_size = 4
    input_channels = 3
    height = 256
    width = 256

    input_image = torch.randn(batch_size, input_channels, height, width)
    start_x = torch.randint(0, width, (batch_size, 1, 1, 1)).float() / width
    start_y = torch.randint(0, height, (batch_size, 1, 1, 1)).float() / height
    start_x = start_x.expand(-1, 1, height, width)
    start_y = start_y.expand(-1, 1, height, width)
    input_tensor = torch.cat((input_image, start_x, start_y), dim=1)

    # Test each model
    for model_name in ["resnet", "efficientnet", "cae"]:
        print(f"\n--- Testing {model_name} ---")
        model = Actor(model_name, height, width, input_channels, 2, pretrained=False)
        summary(model, input_tensor.shape[1:])
        output = model(input_tensor)
        print(f"{model_name} output shape:", output.shape)
