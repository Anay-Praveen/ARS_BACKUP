ENV_HYPERPARAMS = {
    "CartPole-v1": {
        "batch_size": 64,              # Smaller batch size for simpler dynamics
        "actor_lr": 1e-3,             # Faster learning for a simple task
        "critic_lr": 1e-3,
        "mask_lr": 5e-5,              # Slower updates for mask
        "noise_lr": 5e-5,             # Controlled noise updates
        "alpha": 0.2,                 # Standard entropy coefficient
        "gamma": 0.99,                # Discount factor for long-term rewards
        "tau": 0.005,                 # Soft target update
        "initial_noise_scale": 0.1,   # Less exploration for stable rewards
        "noise_decay_factor": 0.997,  # Gradual reduction in exploration
        "hidden_dims": [128, 64],     # Smaller network for simplicity
        "min_buffer_size": 500,
        "target_update_interval": 5,
        "reward_scale": 1.0,
        "warmup_episodes": 10,        # Random actions for initial exploration
        "lr_decay_factor": 0.995,
        "lr_decay_episodes": 100,
        "max_lr": 1e-3,
        "min_lr": 1e-5,
    },
    "MountainCarContinuous-v0": {
        "batch_size": 128,            # Larger batch size for sparse rewards
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "mask_lr": 1e-4,
        "noise_lr": 1e-4,
        "alpha": 0.1,                 # Lower temperature to encourage exploitation
        "gamma": 0.99,
        "tau": 0.001,
        "initial_noise_scale": 0.3,   # Higher noise for greater exploration
        "noise_decay_factor": 0.998,
        "hidden_dims": [256, 128],    # Moderate-sized network for complexity
        "min_buffer_size": 2000,
        "target_update_interval": 5,
        "reward_scale": 10.0,
        "warmup_episodes": 20,
        "lr_decay_factor": 0.99,
        "lr_decay_episodes": 50,
        "max_lr": 5e-3,
        "min_lr": 1e-4,
    },
    "LunarLanderContinuous-v3": {
        "batch_size": 256,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "mask_lr": 1e-4,
        "noise_lr": 1e-4,
        "alpha": 0.2,
        "gamma": 0.99,
        "tau": 0.005,
        "initial_noise_scale": 0.2,   # Moderate noise for steady exploration
        "noise_decay_factor": 0.997,
        "hidden_dims": [512, 256, 128],
        "min_buffer_size": 5000,
        "target_update_interval": 5,
        "reward_scale": 1.0,
        "warmup_episodes": 15,
        "lr_decay_factor": 0.995,
        "lr_decay_episodes": 50,
        "max_lr": 1e-3,
        "min_lr": 1e-6,
    },
    "BipedalWalker-v3": {
        "batch_size": 512,            # Large batch size for stability
        "actor_lr": 5e-4,
        "critic_lr": 5e-4,
        "mask_lr": 5e-5,
        "noise_lr": 5e-5,
        "alpha": 0.15,                # Lower entropy coefficient for stability
        "gamma": 0.99,
        "tau": 0.001,
        "initial_noise_scale": 0.3,   # Increased noise for exploration
        "noise_decay_factor": 0.998,
        "hidden_dims": [1024, 512, 256],
        "min_buffer_size": 10000,
        "target_update_interval": 5,
        "reward_scale": 5.0,
        "warmup_episodes": 50,
        "lr_decay_factor": 0.99,
        "lr_decay_episodes": 30,
        "max_lr": 2e-3,
        "min_lr": 1e-5,
    }
}

# Training-specific configuration for each environment
TRAINING_CONFIG = {
    "CartPole-v1": {
        "episodes_per_env": 5000,     # Shorter training duration
        "max_steps_per_episode": 500,
        "eval_frequency": 10,
        "stable_threshold": 195.0,
        "stable_episodes_required": 20,
        "early_stop_threshold": 200.0,
        "plateau_window": 50,
        "plateau_threshold": 0.5,     # Stricter plateau detection
        "save_frequency": 50,
    },
    "MountainCarContinuous-v0": {
        "episodes_per_env": 10000,    # Longer duration for sparse rewards
        "max_steps_per_episode": 999,
        "eval_frequency": 20,
        "stable_threshold": 90.0,
        "stable_episodes_required": 30,
        "early_stop_threshold": 95.0,
        "plateau_window": 100,
        "plateau_threshold": 1.5,     # Adjusted for slow improvement
        "save_frequency": 50,
    },
    "LunarLanderContinuous-v3": {
        "episodes_per_env": 10000,
        "max_steps_per_episode": 1000,
        "eval_frequency": 25,
        "stable_threshold": 200.0,
        "stable_episodes_required": 40,
        "early_stop_threshold": 250.0,
        "plateau_window": 100,
        "plateau_threshold": 1.0,
        "save_frequency": 50,
    },
    "BipedalWalker-v3": {
        "episodes_per_env": 20000,    # Longer training for complexity
        "max_steps_per_episode": 1600,
        "eval_frequency": 30,
        "stable_threshold": 300.0,
        "stable_episodes_required": 50,
        "early_stop_threshold": 310.0,
        "plateau_window": 100,
        "plateau_threshold": 2.5,
        "save_frequency": 50,
    }
}
