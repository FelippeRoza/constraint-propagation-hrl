import cv2
import numpy as np
import gymnasium as gym
import argparse
import yaml
import sys
import os
import torch
from stable_baselines3 import PPO
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from environments.safety_wrappers import make_safety_env
from models.custom_feature_extractor import CustomMultimodalFeatureExtractor

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

def main(config_path, model_path, is_baseline):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup environment
    # We use the same wrapper as training to ensure observation space consistency
    env_name = config['environment']['name']
    env = make_safety_env(env_name=env_name, render_mode=None)

    if is_baseline:
        print("Wrapping environment for Baseline Agent...")
        env = BaselineWrapper(env)

    print(f"Loading model from {model_path}...")
    
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
    
    try:
        model = PPO.load(model_path, device=device)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print("Make sure you have trained the model using 'python training/train_hrl.py' or provide the correct path.")
        return

    print("\n--- Agent Visualization Started ---")
    print(f"Environment: {env_name}")
    print("Press 'q' to quit.")
    print("-----------------------------------")

    obs, info = env.reset()
    
    while True:
        # Predict action using the trained model
        action, _states = model.predict(obs, deterministic=True)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Visualization: extract vision data
        if 'vision' in obs:
            vision_image_rgb = obs['vision']
            
            # Upscale the 64x64 image back to 256x256 for better visualization
            if vision_image_rgb.shape[0] < 256:
                vision_image_rgb = cv2.resize(vision_image_rgb, (256, 256), interpolation=cv2.INTER_NEAREST)

            # Convert RGB to BGR for OpenCV
            vision_image_bgr = cv2.cvtColor(vision_image_rgb, cv2.COLOR_RGB2BGR)
        else:
            # Create a black image if vision is missing
            vision_image_bgr = np.zeros((256, 256, 3), dtype=np.uint8)

        # Get text prompt
        text_prompt = info.get('text_prompt', 'No text available.')

        # Draw UI
        h, w, _ = vision_image_bgr.shape
        # Black rectangle for text background
        cv2.rectangle(vision_image_bgr, (0, h - 30), (w, h), (0, 0, 0), -1)
        # Text
        cv2.putText(
            vision_image_bgr,
            text_prompt,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # Show image
        cv2.imshow("Agent View", vision_image_bgr)

        # Wait 1ms for key press (allows window to update)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break

        # Reset if episode ends
        if terminated or truncated:
            print("Episode finished. Resetting environment.")
            obs, info = env.reset()

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a trained agent in the environment.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to the configuration YAML file.")
    parser.add_argument("--model", type=str, default="ppo_multimodal_agent", help="Path to the trained model file (e.g., ppo_multimodal_agent.zip).")
    parser.add_argument("--baseline", action="store_true", help="Set this flag if loading a baseline agent (resizes inputs to 64x64).")
    args = parser.parse_args()
    main(args.config, args.model, args.baseline)