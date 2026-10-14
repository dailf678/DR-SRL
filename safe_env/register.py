import gymnasium as gym
from safe_env.safe_control_gym.config import cartpole_config

ENV_ID = 'DRSRL-CartPole-v0'


def register():
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID,
                     entry_point='safe_env.safe_control_gym.my_cartpole:MyCartPole',
                     kwargs=cartpole_config, order_enforce=False,
                     disable_env_checker=True)
