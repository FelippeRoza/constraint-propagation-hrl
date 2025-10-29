# training/train_hrl.py
import torch
import argparse
import torch.nn as nn
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from environments.safety_wrappers import make_safety_env
from models.custom_feature_extractor import CustomMultimodalFeatureExtractor


def main(config_path):
    # Load configuration from YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup environment
    env = make_safety_env(env_name=config['environment']['name'], render_mode="human")
    vec_env = make_vec_env(lambda: env, n_envs=1)

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

    model = PPO(
        "MultiInputPolicy", # MultiInputPolicy for Dict observation spaces
        vec_env,
        verbose=1,
        learning_rate=float(config['training']['lr']),
        batch_size=config['training']['batch_size'],
        n_steps=config['training']['n_steps'] if 'n_steps' in config['training'] else 2048, # Number of steps to run for each environment per update
        policy_kwargs=policy_kwargs,
        tensorboard_log="./ppo_multimodal_tensorboard/"
    )

    print("--- Training Started ---")
    print(f"Environment: {config['environment']['name']}")
    print(f"Algorithm: {config['training']['algo']}")
    print("------------------------")

    total_timesteps = config['training']['total_timesteps'] if 'total_timesteps' in config['training'] else 1_000_000
    model.learn(
        total_timesteps=total_timesteps,
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
