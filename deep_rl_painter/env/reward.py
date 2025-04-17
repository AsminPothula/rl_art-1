import torch
import torch.nn.functional as F
import lpips
from typing import Callable


def calculate_ssim_reward(prev_canvas: torch.Tensor, current_canvas: torch.Tensor, target_canvas: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Structural Similarity Index Measure (SSIM) reward.  Higher is better (closer to 1).

    Args:
        prev_canvas (torch.Tensor): The previous canvas state (shape: [batch_size, 3, height, width]).
        current_canvas (torch.Tensor): The current canvas state (shape: [batch_size, 3, height, width]).
        target_canvas (torch.Tensor): The target canvas state (shape: [batch_size, 3, height, width]).

    Returns:
        torch.Tensor: The SSIM reward (shape: [batch_size, 1]).
    """
    # SSIM is often calculated per-image, then averaged.  We do that here.
    batch_size = prev_canvas.shape[0]
    ssim_values = []
    for i in range(batch_size):
        # Calculate SSIM between current and target
        ssim_ct = calculate_ssim(current_canvas[i].unsqueeze(0), target_canvas[i].unsqueeze(0))
        # Calculate SSIM between previous and target
        ssim_pt = calculate_ssim(prev_canvas[i].unsqueeze(0), target_canvas[i].unsqueeze(0))

        # Reward is the change in SSIM.  We want to maximize the change.
        ssim_values.append(ssim_ct - ssim_pt)

    return torch.tensor(ssim_values).unsqueeze(1).to(prev_canvas.device)

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, k1: float = 0.01, k2: float = 0.03) -> torch.Tensor:
    """
    Calculates the Structural Similarity Index Measure (SSIM) between two images.
    This is a lower-level function, used by calculate_ssim_reward.

    Args:
        img1 (torch.Tensor): The first image (shape: [1, 3, height, width]).
        img2 (torch.Tensor): The second image (shape: [1, 3, height, width]).
        window_size (int, optional): The size of the Gaussian window (default: 11).
        k1 (float, optional): Parameter to prevent division by zero (default: 0.01).
        k2 (float, optional): Parameter to prevent division by zero (default: 0.03).

    Returns:
        torch.Tensor: The SSIM value (shape: [1]).
    """
    C1 = (k1 * 255) ** 2
    C2 = (k2 * 255) ** 2
    device = img1.device

    # Ensure images are float
    img1 = img1.float()
    img2 = img2.float()

    # Create Gaussian Window
    def gaussian_filter(size, sigma):
        x = torch.arange(size).to(device)
        mean = (size - 1) / 2.0
        gauss = torch.exp(-((x - mean) / sigma) ** 2 / 2.0)
        return gauss / gauss.sum()

    window_1d = gaussian_filter(window_size, 1.5).unsqueeze(1)
    window_2d = torch.mm(window_1d, window_1d.t()).unsqueeze(0).unsqueeze(0)  # [1, 1, window_size, window_size]
    window = window_2d.expand(img1.size(1), 1, window_size, window_size).contiguous()  # [C, 1, window_size, window_size]

    def conv_gauss(img, window):
        padding = window_size // 2
        return F.conv2d(img, window, padding=padding, groups=img.size(1))

    mu1 = conv_gauss(img1, window)
    mu2 = conv_gauss(img2, window)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv_gauss(img1 * img1, window) - mu1_sq
    sigma2_sq = conv_gauss(img2 * img2, window) - mu2_sq
    sigma12 = conv_gauss(img1 * img2, window) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()



def calculate_mse_reward(prev_canvas: torch.Tensor, current_canvas: torch.Tensor, target_canvas: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Mean Squared Error (MSE) reward.  Lower is better, so we negate it.

    Args:
        prev_canvas (torch.Tensor): The previous canvas state (shape: [batch_size, 3, height, width]).
        current_canvas (torch.Tensor): The current canvas state (shape: [batch_size, 3, height, width]).
        target_canvas (torch.Tensor): The target canvas state (shape: [batch_size, 3, height, width]).

    Returns:
        torch.Tensor: The MSE reward (shape: [batch_size, 1]).
    """
    mse_prev = F.mse_loss(prev_canvas, target_canvas, reduction='none').mean(dim=[1, 2, 3])
    mse_current = F.mse_loss(current_canvas, target_canvas, reduction='none').mean(dim=[1, 2, 3])
    # Reward is the *negative* change in MSE (we want to *decrease* the MSE).
    return -(mse_current - mse_prev).unsqueeze(1)



def calculate_lpips_reward(prev_canvas: torch.Tensor, current_canvas: torch.Tensor, target_canvas: torch.Tensor, lpips_fn: lpips.LPIPS) -> torch.Tensor:
    """
    Calculates the LPIPS (Learned Perceptual Image Patch Similarity) reward. Lower is better, so we negate it.

    Args:
        prev_canvas (torch.Tensor): The previous canvas state (shape: [batch_size, 3, height, width]).  Assumes values in [0, 1].
        current_canvas (torch.Tensor): The current canvas state (shape: [batch_size, 3, height, width]). Assumes values in [0, 1].
        target_canvas (torch.Tensor): The target canvas state (shape: [batch_size, 3, height, width]). Assumes values in [0, 1].
        lpips_fn (lpips.LPIPS):  The LPIPS loss function.  Should be initialized outside this function for efficiency.

    Returns:
        torch.Tensor: The LPIPS reward (shape: [batch_size, 1]).
    """

    # LPIPS expects input in the range [0, 1]
    lpips_prev = lpips_fn(prev_canvas, target_canvas).squeeze() # shape [batch_size]
    lpips_current = lpips_fn(current_canvas, target_canvas).squeeze()
    return -(lpips_current - lpips_prev).unsqueeze(1) # shape [batch_size, 1]



def calculate_reward(
    prev_canvas: torch.Tensor,
    current_canvas: torch.Tensor,
    target_canvas: torch.Tensor,
    reward_function: Callable,
    lpips_fn: lpips.LPIPS = None  # Make lpips_fn optional
) -> torch.Tensor:
    """
    Calculates the reward based on the chosen reward function.

    Args:
        prev_canvas (torch.Tensor): The previous canvas state (shape: [batch_size, 3, height, width]).
        current_canvas (torch.Tensor): The current canvas state (shape: [batch_size, 3, height, width]).
        target_canvas (torch.Tensor): The target canvas state (shape: [batch_size, 3, height, width]).
        reward_function (Callable): The reward function to use (e.g., calculate_ssim_reward, calculate_mse_reward, calculate_lpips_reward).
        lpips_fn (lpips.LPIPS, optional):  The LPIPS loss function instance, if using LPIPS.  Should be created
            *outside* this function and passed in, to avoid re-creating it on every call.

    Returns:
        torch.Tensor: The calculated reward (shape: [batch_size, 1]).
    """
    if reward_function == calculate_lpips_reward and lpips_fn is None:
        raise ValueError("lpips_fn must be provided when using calculate_lpips_reward")

    #return reward_function(prev_canvas, current_canvas, target_canvas, lpips_fn) # Pass lpips_fn if needed.
    if reward_function == calculate_lpips_reward:
        return reward_function(prev_canvas, current_canvas, target_canvas, lpips_fn)
    else:
        return reward_function(prev_canvas, current_canvas, target_canvas)




if __name__ == "__main__":
    # Example Usage
    batch_size = 4
    height = 256
    width = 256
    # Create dummy data
    prev_canvas = torch.randn(batch_size, 3, height, width)
    current_canvas = torch.randn(batch_size, 3, height, width)
    target_canvas = torch.randn(batch_size, 3, height, width)

    # Ensure data is in the range [0, 1] for LPIPS
    prev_canvas = (prev_canvas - prev_canvas.min()) / (prev_canvas.max() - prev_canvas.min())
    current_canvas = (current_canvas - current_canvas.min()) / (current_canvas.max() - current_canvas.min())
    target_canvas = (target_canvas - target_canvas.min()) / (target_canvas.max() - target_canvas.min())


    # Initialize LPIPS (only do this once!)
    lpips_fn = lpips.LPIPS(net='vgg').to(prev_canvas.device) #  Move to the device

    # Calculate rewards
    ssim_reward = calculate_reward(prev_canvas, current_canvas, target_canvas, calculate_ssim_reward)
    mse_reward = calculate_reward(prev_canvas, current_canvas, target_canvas, calculate_mse_reward)
    lpips_reward = calculate_reward(prev_canvas, current_canvas, target_canvas, calculate_lpips_reward, lpips_fn) # Pass the lpips_fn


    print("SSIM Reward:", ssim_reward)
    print("MSE Reward:", mse_reward)
    print("LPIPS Reward:", lpips_reward)
