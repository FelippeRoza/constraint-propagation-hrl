# environments/multimodal_wrapper.py
import gymnasium as gym
import numpy as np
from gymnasium.spaces import flatten, flatten_space
from transformers import AutoTokenizer
import cv2


class MultimodalWrapper(gym.Wrapper):
    """
    A wrapper to convert a standard Safety Gymnasium environment's observation
    into a multimodal dictionary format ({'vision', 'text', 'proprio'}).

    It generates placeholder data for vision and text modalities, as the base
    environment only provides proprioceptive data.
    """
    def __init__(self, env, cost_coefficient=1.0):
        super().__init__(env)
        # Initialize the tokenizer for the text modality
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.seq_len = 128 # Standard sequence length
        
        # Define a threshold for LIDAR "closeness"
        # A value of 1.0 means the object is at the agent's location
        # A value of 0.0 means the object is at the maximum lidar distance
        self.lidar_threshold = 0.7
        
        # Coefficient to penalize unsafe behavior in the reward function
        self.cost_coefficient = cost_coefficient

        self.is_vision_env = isinstance(self.env.observation_space, gym.spaces.Dict)

        if self.is_vision_env:
            # For vision envs, proprio is everything *except* the 'vision' key
            # Create a new space for the flattened proprioceptive data
            proprio_keys = [k for k in self.env.observation_space.keys() if k != 'vision']
            self.proprio_subspace = gym.spaces.Dict({k: self.env.observation_space[k] for k in proprio_keys if k in dict(self.env.observation_space)})
            self.proprio_space = flatten_space(self.proprio_subspace)
            # The vision space comes directly from the environment
            self.original_vision_space = self.env.observation_space['vision']
        else:
            # For non-vision envs, the whole observation is proprioception
            self.proprio_space = self.env.observation_space

        self.vision_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.text_space = gym.spaces.Box(low=0, high=self.tokenizer.vocab_size, shape=(self.seq_len,), dtype=np.int64)
        self.text_attention_mask_space = gym.spaces.Box(low=0, high=1, shape=(128,), dtype=np.int64)

        # The new observation space is a dictionary combining all modalities
        self.observation_space = gym.spaces.Dict({
            "vision": self.vision_space,
            "text": self.text_space,
            "text_attention_mask": self.text_attention_mask_space,
            "proprio": self.proprio_space
        })

    def _generate_text_from_obs(self, obs):
        """Generates a descriptive text string from the observation dictionary."""
        if not self.is_vision_env:
            return "No vision data available."

        descriptions = []
        lidar_sensors = {
            "goal": obs.get("goal_lidar"),
            "hazard": obs.get("hazards_lidar"),
            "vase": obs.get("vases_lidar"),
        }

        for name, lidar_data in lidar_sensors.items():
            if lidar_data is None:
                continue
            
            non_zero_lidar = lidar_data[lidar_data > 0.0]

            if non_zero_lidar.size > 0: # An object is detected
                # The largest value corresponds to the closest object
                max_closeness = np.max(non_zero_lidar)
                
                # Check if the closest detected object is "close enough" to report
                if max_closeness > self.lidar_threshold:
                    # Find the index of the closest object in the original array
                    # to determine its direction.
                    closest_obj_idx = np.where(lidar_data == max_closeness)[0][0]
                    
                    if closest_obj_idx in [11, 12]:
                        direction = "ahead"
                    elif closest_obj_idx in [9, 10]:
                        direction = "right"
                    elif closest_obj_idx in [13, 14]:
                        direction = "left"
                    else:
                        direction = "out of view"

                    descriptions.append(f"{name} {direction}")

        if not descriptions:
            return "Path is clear."
        
        return ". ".join(descriptions) + "."

    def _create_multimodal_obs(self, obs, info):
        """Creates a multimodal observation dictionary from a base environment observation."""
        
        if self.is_vision_env:
            # The observation is a dictionary. Extract vision and flatten the rest for proprio
            vision_obs = cv2.resize(obs['vision'], (64, 64), interpolation=cv2.INTER_AREA)
            proprio_obs_dict = {k: v for k, v in obs.items() if k != 'vision'}            
            proprio_flat = flatten(self.proprio_subspace, proprio_obs_dict)
        else:
            # The observation is a single array (proprio). Create a dummy vision image
            vision_obs = np.zeros(self.vision_space.shape, dtype=self.vision_space.dtype)
            proprio_flat = obs

        # Generate and tokenize text based on the observation
        text_prompt = self._generate_text_from_obs(obs)
        tokenized_text = self.tokenizer(
            text_prompt,
            padding='max_length',
            truncation=True,
            max_length=self.seq_len,
            return_tensors='np' # Return numpy arrays
        )

        multimodal_obs = {
            "vision": vision_obs,
            "text": tokenized_text['input_ids'].squeeze(),
            "text_attention_mask": tokenized_text['attention_mask'].squeeze(),
            "proprio": proprio_flat
        }
        # Add the raw text to the info dictionary for debugging/visualization
        info['text_prompt'] = text_prompt
        return multimodal_obs, info

    def reset(self, **kwargs):
        # The base env returns a dict for vision envs, or an array for non-vision
        obs, info = self.env.reset(**kwargs)
        return self._create_multimodal_obs(obs, info)

    def step(self, action):
        # Pack the 'cost' into the info dictionary to maintain compatibility with sb3
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        
        # Reward Shaping: Penalize the agent for unsafe behavior (costs)
        # This allows standard PPO to learn a balance between task performance and safety.
        reward -= self.cost_coefficient * cost
        
        multimodal_obs, info_with_text = self._create_multimodal_obs(obs, info)
        info_with_text['cost'] = cost
        
        return multimodal_obs, reward, terminated, truncated, info_with_text