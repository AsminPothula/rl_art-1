config = {
    # training setup
    "seed": 42,
    "episodes": 500,
    "max_steps": 100,                        # not neccesary

    # painting environment
    "target_image": "target.jpg",
    "stroke_length": 10,                     # not using it anymore 
    "error_threshold": 10000.0,
    "max_total_length": 10000,

    # model parameters
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "buffer_size": 100000,
    "batch_size": 64,
    "gamma": 0.99,
    "tau": 0.005,

    # exploration noise
    "initial_noise_scale": 0.2,
    "noise_decay": 0.995,

    # saving
    "save_every": 50
}
