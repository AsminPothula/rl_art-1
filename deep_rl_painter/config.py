import torch
config = {
    # training setup
    "seed": 42,
    "episodes": 50,                          # change to 500 or more later
    "max_steps": 50,                         # is this the number of strokes?

    # painting environment
    "target_image_path": "target_images/target_image_1.jpg",
    "error_threshold": 10000.0,
    "max_total_length": 10000,
    "max_strokes": 10000,                      # max number of strokes per episode
    "max_strokes_per_step": 1,               # max number of strokes per step

    # model parameters
    "model_name": "resnet18",                # resnet18, resnet50, resnet101
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "buffer_size": 100000,
    "batch_size": 64,
    "gamma": 0.99,
    "tau": 0.005,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "action_dim": 6,                        # 2D action space (x, y, r, g, b, w)

    # exploration noise
    "initial_noise_scale": 0.2,
    "noise_decay": 0.995,

    # saving
    "save_every_step": 100,
    "save_every_episode": 5,
    "save_model_dir": "models",
    "save_model_name": "model.pth",
    "save_best_model": True,
    "save_best_model_dir": "best_models",
    "save_best_model_name": "best_model.pth",

    # Canvas parameters
    "canvas_size": (224, 224),  # (height, width)
    "canvas_channels": 3,  # 1 for grayscale, 3 for RGB
    "canvas_color": (0, 0, 0),  # black background
    "canvas_stroke_color": 255,  # white stroke

    # logging
    "log_dir": "logs",
    "log_every": 10,
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "log_file": "training.log",

}
