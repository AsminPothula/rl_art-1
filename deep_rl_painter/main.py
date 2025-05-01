"""
Main Entry Point for Deep RL Painter

This script:
- Parses command-line arguments
- Updates config dictionary
- Delegates full training to `train(config)` from train.py
"""

import argparse
import torch
from config import config
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deep RL Painter")
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--reward', type=str, default='ssim')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--target_image', type=str, default='target_images/target_image_1.jpg')
    args = parser.parse_args()

    # Override config from CLI
    config["model_name"] = args.model
    config["reward_function_name"] = args.reward
    config["batch_size"] = args.batch_size
    config["episodes"] = args.episodes
    config["target_image_path"] = args.target_image
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("✨ Configuration:")
    for k, v in config.items():
        print(f"{k}: {v}")

    train(config)

if __name__ == "__main__":
    main()
