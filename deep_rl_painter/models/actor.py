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
        self.in_channels = in_channels + 2 # 2 for x and y direction
        self.out_channels = out_channels
        self.height = height
        self.width = width

        if model_name == "resnet":
            self.model = models.resnet18(pretrained=pretrained)
            self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False) #Modify the first conv layer.
            self.model.fc = nn.Linear(self.model.fc.in_features, self.output_channels * height * width)
        elif model_name == "efficientnet":
            self.model = models.efficientnet_b0(pretrained=pretrained)
            self.model.features[0][0] = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False) #Modify the first conv layer.
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.output_channels * height * width)
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
        if self.model_name == "resnet" or self.model_name == "efficientnet":
            out = self.model(x).view(x.size(0), self.output_channels, self.height, self.width)
        elif self.model_name == "cae":
            out = self.model(x)
        return out

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device)) # Load to the correct device


# This part would be data preprocessing and loading., not having any seperate class. I guess.

# Example Usage
batch_size = 4
input_channels = 3
height = 256
width = 256
input_image = torch.randn(batch_size, input_channels, height, width)
start_x = torch.randint(0, width, (batch_size, 1, 1, 1)).float() / width #normalized x
start_y = torch.randint(0, height, (batch_size, 1, 1, 1)).float() / height #normalized y
start_x = start_x.expand(-1, 1, height, width)
start_y = start_y.expand(-1, 1, height, width)
input_tensor = torch.cat((input_image, start_x, start_y), dim=1)


# ResNet
resnet_predictor = Actor("resnet", input_channels, height, width)
summary(resnet_predictor, input_tensor.shape[1:]) #input shape without the batch dimension

# EfficientNet
efficientnet_predictor = Actor("efficientnet", input_channels, height, width)
summary(efficientnet_predictor, input_tensor.shape[1:])

# CAE
cae_predictor = Actor("cae", input_channels, height, width)
summary(cae_predictor, input_tensor.shape[1:])


# ResNet
resnet_predictor = Actor("resnet", height, width)
resnet_output = resnet_predictor(input_tensor)
print("ResNet output shape:", resnet_output.shape)

# EfficientNet
efficientnet_predictor = Actor("efficientnet", height, width)
efficientnet_output = efficientnet_predictor(input_tensor)
print("EfficientNet output shape:", efficientnet_output.shape)

# CAE
cae_predictor = Actor("cae", height, width)
cae_output = cae_predictor(input_tensor)
print("CAE output shape:", cae_output.shape)

# Example Loss calculation. (Replace with appropriate brush stroke specific loss.)
loss_function = nn.MSELoss()
dummy_target = torch.randn(batch_size, 2, height,width) #replace with the actual target.

resnet_loss = loss_function(resnet_output, dummy_target)
efficientnet_loss = loss_function(efficientnet_output, dummy_target)
cae_loss = loss_function(cae_output, dummy_target)
print(f"ResNet Loss: {resnet_loss}")
print(f"EfficientNet Loss: {efficientnet_loss}")
print(f"CAE Loss: {cae_loss}")