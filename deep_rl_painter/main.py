import argparse
import torch
from config import config
from train import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--reward', type=str, default='ssim')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--target_image', type=str, default='target_images/target_image_1.jpg')
    args = parser.parse_args()

    # Update config
    config["model_name"] = args.model
    config["reward_function_name"] = args.reward
    config["batch_size"] = args.batch_size
    config["num_episodes"] = args.episodes
    config["target_image_path"] = args.target_image

    # Static settings
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["image_size"] = (256, 256)
    config["replay_buffer_size"] = 100000
    config["actor_lr"] = 1e-4
    config["critic_lr"] = 1e-3
    config["gamma"] = 0.99
    config["tau"] = 0.005
    config["noise_std"] = 0.1
    config["max_steps"] = 50
    config["save_every"] = 50

    print("Running with configuration:")
    for k, v in config.items():
        print(f"{k}: {v}")

    # Call train
    train(config)

if __name__ == "__main__":
    main()
