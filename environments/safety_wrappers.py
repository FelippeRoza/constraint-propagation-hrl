# environments/safety_wrappers.py
import safety_gymnasium
from .multimodal_wrapper import MultimodalWrapper

def make_safety_env(env_name="SafetyPointGoal1", render_mode=None):
    # Vision environments in safety-gymnasium handle their own rendering via the
    # observation dictionary. Setting render_mode="human" will cause an error
    if "Vision" in env_name and render_mode == "human":
        render_mode = None

    env = safety_gymnasium.make(env_name, render_mode=render_mode)
    # Wrap the base environment to convert observations to a multimodal dictionary
    env = MultimodalWrapper(env)
    return env
