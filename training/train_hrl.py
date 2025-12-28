# training/train_hrl.py
import torch
import argparse
import torch.nn as nn
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environments.safety_wrappers import make_safety_env
from models.custom_feature_extractor import CustomMultimodalFeatureExtractor


def main(config_path):
    # Load configuration from YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup environment
    # Use multiple environments for faster training and disable rendering
    n_envs = config['training'].get('n_envs', 8)
    vec_env = make_vec_env(
        lambda: make_safety_env(env_name=config['environment']['name'], render_mode=None),
        n_envs=n_envs
    )

    # Get proprioceptive dimension from the wrapped environment's observation space
    proprio_dim = vec_env.observation_space['proprio'].shape[0]
    action_dim = vec_env.action_space.shape[0]

    model_config = config['model']
    policy_kwargs = dict(
        features_extractor_class=CustomMultimodalFeatureExtractor,
        features_extractor_kwargs=dict(
            embed_dim=model_config['embed_dim'],
            proprio_dim=proprio_dim,
            tiny_transformer_layers=model_config['tiny_transformer_layers'],
            action_dim=action_dim
        ),
        net_arch=[dict(pi=[64, 64], vf=[64, 64])]
    )

    # Determine device: Check for NVIDIA, Apple MPS, Intel XPU, otherwise let SB3 decide (auto)
    device = "auto"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Apple Silicon GPU detected")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = "xpu"
        print(f"Intel Arc GPU detected: {torch.xpu.get_device_name(0)}")
        torch.xpu.empty_cache()

    model = PPO(
        "MultiInputPolicy", # MultiInputPolicy for Dict observation spaces
        vec_env,
        verbose=1,
        learning_rate=float(config['training']['lr']),
        batch_size=config['training'].get('batch_size', 64), # Lower default batch size to prevent VRAM OOM
        n_steps=config['training'].get('n_steps', 512), # Lower default n_steps to prevent RAM OOM
        policy_kwargs=policy_kwargs,
        tensorboard_log="./ppo_multimodal_tensorboard/",
        device=device
    )

    print("--- Training Started ---")
    print(f"Environment: {config['environment']['name']}")
    print(f"Algorithm: {config['training']['algo']}")
    print("------------------------")

    # Save model checkpoints
    checkpoint_freq = config['training'].get('checkpoint_freq', 50000)
    checkpoint_freq = max(checkpoint_freq // n_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path='./checkpoints/models/',
        name_prefix='ppo_multimodal_agent'
    )

    total_timesteps = config['training'].get('total_timesteps', 1_000_000)
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        log_interval=1, # Log every 1 update
        progress_bar=True
    )
    
    # Save the trained model
    model.save("ppo_multimodal_agent")
    print("Training finished and model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Hierarchical Reinforcement Learning agent.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()
    main(args.config)
