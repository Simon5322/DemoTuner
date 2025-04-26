from typing import Union, Type, TypeVar

from stable_baselines3 import DDPG
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.td3.policies import TD3Policy

SelfDDPG = TypeVar("SelfDDPG", bound="testDDPG")


class DDPGFD(DDPG):
    def __init__(self, policy: Union[str, Type[TD3Policy]], env: Union[GymEnv, str]):
        super().__init__(policy, env)

    def learn(
        self: SelfDDPG,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "testDDPG",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDDPG:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
