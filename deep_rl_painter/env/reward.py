import torch
from torchvision import transforms
from torch.nn.functional import cosine_similarity
import clip


def calculate_reward(prev_canvas, current_canvas, target_canvas, device):
    """
    Calculates the reward based on the chosen reward function.

    Args:
        prev_canvas (torch.Tensor): The previous canvas state (shape: [batch_size, channels, height, width]).
        current_canvas (torch.Tensor): The current canvas state (shape: [batch_size, channels, height, width]).
        target_canvas (torch.Tensor): The target canvas state (shape: [batch_size, channels, height, width]).
    Returns:
        torch.Tensor: The calculated reward (shape: [batch_size, 1]).
    """
    # Using CLIP for calculating cosine similarity
    latent1 = get_latent_representation(
        prev_canvas, device)
    latent2 = get_latent_representation(
        current_canvas, device)
    target_latent = get_latent_representation(
        target_canvas, device)

    # Calculate cosine similarity
    cosine_similarity_score_prev = calculate_cosine_similarity(
        latent1, target_latent)

    cosine_similarity_score_current = calculate_cosine_similarity(
        latent2, target_latent)

    return cosine_similarity_score_current - cosine_similarity_score_prev


def get_latent_representation(image, device):
    """
    Extracts the latent representation of an image using a pre-trained model.
    The model should be a feature extractor (e.g., ResNet) with the last layer removed.

    Args:
        image (torch.Tensor): The input image tensor (shape: [batch_size, channels, height, width]).
        device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: The latent representation of the image (a 1D tensor).
    """

    # Modify the model to output the features from the penultimate layer
    model, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(transforms.ToPILImage()(image)).unsqueeze(0).to(device)
    # Preprocess the image and remove the last layer (e.g., classification head)
    modules = list(model.children())[:-1]
    feature_extractor = torch.nn.Sequential(*modules)
    feature_extractor.eval()  # Set to evaluation mode

    with torch.no_grad():
        latent_representation = feature_extractor(image)
        latent_representation = torch.flatten(
            latent_representation, 1)  # Flatten to a 1D tensor

    return latent_representation


def calculate_cosine_similarity(latent1, latent2):
    """
    Calculates the cosine similarity between two latent vectors.

    Args:
        latent1 (torch.Tensor): The first latent vector.
        latent2 (torch.Tensor): The second latent vector.

    Returns:
        torch.Tensor: The cosine similarity score (a scalar tensor). Returns None if
                      either latent vector is None.
    """
    if latent1 is None or latent2 is None:
        return None
    return cosine_similarity(latent1, latent2)



if __name__ == "__main__":
    # Determine the device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Example usage: Create dummy canvas tensors
    batch_size = 2
    channels = 3
    height = 224
    width = 224

    prev_canvas = torch.randn(batch_size, channels, height, width).to(device)
    current_canvas = torch.randn(batch_size, channels, height, width).to(device)
    target_canvas = torch.randn(batch_size, channels, height, width).to(device)

    # Calculate the reward
    reward = calculate_reward(prev_canvas, current_canvas, target_canvas, device)

    # Print the reward
    print("Reward:")
    print(reward)

    # Example of getting a single latent representation
    single_image = torch.randn(1, channels, height, width).to(device)
    latent_vector = get_latent_representation(single_image, device)
    print("\nSingle Latent Representation:")
    print(latent_vector.cpu().numpy())

    # Example of calculating cosine similarity between two latent vectors
    latent1_example = torch.randn(1, 512).to(device)  # Assuming CLIP ViT-B/32 outputs a feature vector of size 512
    latent2_example = torch.randn(1, 512).to(device)
    similarity = calculate_cosine_similarity(latent1_example, latent2_example)
    print("\nCosine Similarity Example:")
    print(similarity.item())