# Actor = Actor(state_dim, action_dims)   - in train.py

""" Input: [state_dim]
→ Hidden layers
→ Output: [action_dim] = 2 (x and y direction) 
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
import warnings
from image_encoder import get_image_encoder
from typing import Dict, Optional

# Turn off all warnings
warnings.filterwarnings("ignore")



class Actor(nn.Module):
    def __init__(self, model_name, height, width, in_channels, out_neurons, action_dim, pretrained=True):
        """
        Initialize the Actor model.
        Args:
            model_name (str): Name of the model architecture to use ('resnet', 'efficient
            net', 'cae' : Custom).
            height (int): Height of the input image.
            width (int): Width of the input image.
            # in_channels (int): Number of input channels (if 2: grayscale canvas + positional map).
            action_dim (int): Dimension of the action space (for previous action) (if 8: 4 for x1,y1,x2,y2,r,g,b, width).
            
            in_channels (int): Number of input channels, modifying it to include grayscale and rgb images.
            
            out_neurons (int): Number of output values (if 6: x,y,r,g,b,width).
            pretrained (bool): Whether to use a pretrained model (default: True).
        """
        super(Actor, self).__init__()
        self.model_name = model_name
        self.in_channels = in_channels
        self.out_neurons = out_neurons
        self.height = height
        self.width = width
        self.image_type = "RGB" if in_channels == 3 else "Grayscale"
        self.action_dim = action_dim

        if model_name == "resnet":
            self.model = models.resnet18(pretrained=pretrained)
            self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.out_neurons)  #

        elif model_name == "efficientnet":
            self.model = models.efficientnet_b0(pretrained=pretrained)
            self.model.features[0][0] = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.out_neurons)  #
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
            out = self.model(x)
        elif self.model_name == "cae":
            out = self.model(x)
        return out

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


# Example/test code (runs only when script is executed directly)
if __name__ == "__main__":
    batch_size = 4
    input_channels = 1  # 1 for grayscale, 3 for RGB
    output_neurons = 6
    height = 256
    width = 256

    input_image = torch.randn(batch_size, input_channels, height, width)

    # Test each model
    for model_name in ["resnet", "efficientnet", "cae"]:
        print(f"\n--- Testing {model_name} ---")
        model = Actor(model_name, height, width, input_channels, output_neurons, pretrained=False)
        summary(model, input_image.shape[1:])
        output = model(input_image)
        print(f"{model_name} output shape:", output.shape)
