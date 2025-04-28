# This file contains the ImageEncoder class, which is a neural network model for encoding images.
# It uses different architectures (ResNet, EfficientNet, etc) based on the specified model name.
# This is imported in the Critic and Actor classes.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import warnings
from typing import Dict, Optional
from torchsummary import summary

# Turn off all warnings
warnings.filterwarnings("ignore")

class ImageEncoder(nn.Module):
    """
    Encodes input images using various CNN architectures.
    """
    def __init__(self, model_name: str = 'resnet50', pretrained: bool = True, fine_tune: bool = True, custom_model: Optional[nn.Module] = None) -> None:
        """
        Args:
            model_name (str): Name of the CNN architecture to use (e.g., 'resnet50', 'efficientnet_b0', 'vgg16').
                           Ignored if custom_model is provided.
            pretrained (bool): Whether to load pre-trained weights from ImageNet.
                               Ignored if custom_model is provided.
            fine_tune (bool): Whether to fine-tune the pre-trained weights during training.
            custom_model (nn.Module, optional): A custom PyTorch model to use for image encoding.
                                             If provided, model_name and pretrained are ignored.
        """

        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.fine_tune = fine_tune
        self.custom_model = custom_model
        self.cnn = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Initializes the CNN model based on the specified architecture.
        """

        # Select CNN architecture
        if self.custom_model is not None:
            self.cnn = self.custom_model
        elif model_name.startswith('resnet'):
            self.cnn = self._get_resnet(model_name)
        elif model_name.startswith('efficientnet'):
            self.cnn = self._get_efficientnet(model_name)
        elif model_name.startswith('vgg'):
            self.cnn = self._get_vgg(model_name)
        elif model_name == 'inception_v3':
            self.cnn = self._get_inception_v3()
        elif model_name == 'convnext_tiny':
            self.cnn = self._get_convnext_tiny()
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        self._freeze_params()

    def _get_resnet(self, model_name: str) -> nn.Module:
        """
        Returns a ResNet model.
        """
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=self.pretrained)
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=self.pretrained)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=self.pretrained)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=self.pretrained)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=self.pretrained)
        else:
            raise ValueError(f"Invalid ResNet model name: {model_name}")
        # Remove the last fully connected layer
        model = nn.Sequential(*list(model.children())[:-2])
        return model

    def _get_efficientnet(self, model_name: str) -> nn.Module:
        """
        Returns an EfficientNet model.
        """
        if model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=self.pretrained)
        elif model_name == 'efficientnet_b1':
            model = models.efficientnet_b1(pretrained=self.pretrained)
        elif model_name == 'efficientnet_b2':
            model = models.efficientnet_b2(pretrained=self.pretrained)
        elif model_name == 'efficientnet_b3':
            model = models.efficientnet_b3(pretrained=self.pretrained)
        elif model_name == 'efficientnet_b4':
            model = models.efficientnet_b4(pretrained=self.pretrained)
        elif model_name == 'efficientnet_b5':
            model = models.efficientnet_b5(pretrained=self.pretrained)
        elif model_name == 'efficientnet_b6':
            model = models.efficientnet_b6(pretrained=self.pretrained)
        elif model_name == 'efficientnet_b7':
            model = models.efficientnet_b7(pretrained=self.pretrained)
        else:
            raise ValueError(f"Invalid EfficientNet model name: {model_name}")
        # Remove the last fully connected layer
        model = nn.Sequential(*list(model.children())[:-1])
        return model

    def _get_vgg(self, model_name: str) -> nn.Module:
        """
        Returns a VGG model.
        """
        if model_name == 'vgg11':
            model = models.vgg11(pretrained=self.pretrained)
        elif model_name == 'vgg13':
            model = models.vgg13(pretrained=self.pretrained)
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=self.pretrained)
        elif model_name == 'vgg19':
            model = models.vgg19(pretrained=self.pretrained)
        else:
            raise ValueError(f"Invalid VGG model name: {model_name}")
        # Remove the classifier part
        model = nn.Sequential(*list(model.children())[:-1])
        return model

    def _get_inception_v3(self) -> nn.Module:
        """
        Returns an Inception v3 model.
        """
        model = models.inception_v3(pretrained=self.pretrained, transform_input=False) #transform_input=False
        # Remove the auxiliary classifier.
        model.AuxLogits = None
        # Redefine the forward method to return only the primary output
        def forward(x):
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.conv2(x)
            x = model.bn2(x)
            x = model.relu(x)
            x = model.conv3(x)
            x = model.bn3(x)
            x = model.relu(x)
            x = model.maxpool1(x)
            x = model.conv4(x)
            x = model.bn4(x)
            x = model.relu(x)
            x = model.conv5(x)
            x = model.bn5(x)
            x = model.relu(x)
            x = model.maxpool2(x)
            x = model.conv6(x)
            x = model.bn6(x)
            x = model.relu(x)
            x = model.conv7(x)
            x = model.bn7(x)
            x = model.relu(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            x = model.dropout(x)
            x = model.fc(x)
            return x
        model.forward = forward
        return model

    def _get_convnext_tiny(self):
        """
        Returns a ConvNeXt-tiny model.
        """
        model = models.convnext_tiny(pretrained=self.pretrained)
        model = nn.Sequential(*list(model.children())[:-1])  # Remove the classification head
        return model

    def _freeze_params(self) -> None:
        """
        Freezes the parameters of the CNN if not fine-tuning.
        """
        if not self.fine_tune:
            for param in self.cnn.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input image.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Image features.
        """
        features = self.cnn(x)
        return features

def get_image_encoder(model_name: str = 'resnet50', pretrained: bool = True, fine_tune: bool = True, custom_model: Optional[nn.Module] = None) -> ImageEncoder:
    """
    Returns an ImageEncoder instance with the specified architecture.

    Args:
        model_name (str): Name of the CNN architecture to use.
        pretrained (bool): Whether to load pre-trained weights.
        fine_tune (bool): Whether to fine-tune the weights.
        custom_model (nn.Module, optional): A custom PyTorch model to use.
    Returns:
        ImageEncoder: An instance of the ImageEncoder class.
    """
    return ImageEncoder(model_name, pretrained, fine_tune, custom_model)

if __name__ == '__main__':
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    img_size = 224  # Or 299 for InceptionV3

    # Create a dummy input tensor
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)

    # Test with different models
    models_to_test = ['resnet18', 'resnet50', 'efficientnet_b0', 'vgg16', 'inception_v3', 'convnext_tiny']
    for model_name in models_to_test:
        encoder = get_image_encoder(model_name=model_name, pretrained=True, fine_tune=False).to(device)
        encoder.eval()  # Set to evaluation mode
        with torch.no_grad():
            features = encoder(dummy_input)
            print(f"{model_name} Output shape: {features.shape}")
        del encoder  # Clear memory
        torch.cuda.empty_cache()
