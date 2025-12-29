# training/train_hrl.py
import torch
import argparse
import numpy as np
import torch.nn as nn
import cv2
import gymnasium as gym
import yaml
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environments.safety_wrappers import make_safety_env
from models.custom_feature_extractor import CustomMultimodalFeatureExtractor


class SafetyMetricsCallback(BaseCallback):
    """
    Callback for logging safety metrics (cost) to TensorBoard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_costs = None
        self.episode_costs = deque(maxlen=100)

    def _on_training_start(self):
        self.current_costs = np.zeros(self.training_env.num_envs)

    def _on_step(self) -> bool:
        infos = self.locals['infos']
        dones = self.locals['dones']
        
        for i, info in enumerate(infos):
            if 'cost' in info:
                self.current_costs[i] += info['cost']
            
            if dones[i]:
                self.episode_costs.append(self.current_costs[i])
                self.current_costs[i] = 0
        return True

    def _on_rollout_end(self):
        if len(self.episode_costs) > 0:
            self.logger.record("rollout/ep_cost_mean", np.mean(self.episode_costs))

class BaselineWrapper(gym.ObservationWrapper):
    """
    Prepares the environment for a standard PPO baseline:
    1. Removes text modalities (text, text_attention_mask) which require special handling.
    2. Resizes vision observation to 64x64 for efficient standard CNN processing.
    """
    def __init__(self, env):
        super().__init__(env)
        # Define new observation space: Vision (64x64) + Proprioception
        self.observation_space = gym.spaces.Dict({
            'vision': gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            'proprio': env.observation_space['proprio']
        })

    def observation(self, obs):
        # Resize vision image from 256x256 to 64x64
        vision_resized = cv2.resize(obs['vision'], (64, 64), interpolation=cv2.INTER_AREA)
        return {
            'vision': vision_resized,
            'proprio': obs['proprio']
        }

def main(config_path, model_type):
    # Load configuration from YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup environment
    # Use multiple environments for faster training and disable rendering
    n_envs = config['training'].get('n_envs', 8)

    def make_env():
        env = make_safety_env(env_name=config['environment']['name'], render_mode=None)
        if model_type == 'baseline':
            env = BaselineWrapper(env)
        return env

    vec_env = make_vec_env(
        make_env,
        n_envs=n_envs
    )

    if model_type == 'hrl':
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
    else:
        # Baseline configuration
        policy_kwargs = None

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
        tensorboard_log=f'tensorboard/ppo_{model_type}',
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
        save_path=f'./checkpoints/{model_type}',
        name_prefix=f'ppo_{model_type}_agent'
    )
    
    safety_callback = SafetyMetricsCallback()

    total_timesteps = config['training'].get('total_timesteps', 1_000_000)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, safety_callback],
        log_interval=1, # Log every 1 update
        progress_bar=True
    )
    
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Hierarchical Reinforcement Learning agent.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    parser.add_argument("--type", type=str, default="hrl", choices=["hrl", "baseline"], help="Type of model to train (hrl or baseline).")
    args = parser.parse_args()
    main(args.config, args.type)
