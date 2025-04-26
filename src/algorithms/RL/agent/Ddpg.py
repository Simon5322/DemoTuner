import gymnasium as gym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import PrioritizedReplayBuffer


class Ddpg():
    def __init__(self, env):
        self.env = env

    def train(self, total_timessteps):
        #env = gym.make(self.env, render_mode="rgb_array")
        # The noise objects for testDDPG
        n_actions = self.env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        prioritized_replay_buffer = PrioritizedReplayBuffer
        model = DDPG("MlpPolicy", self.env, replay_buffer_class=PrioritizedReplayBuffer, action_noise=action_noise, verbose=1)
        model.learn(total_timesteps=total_timessteps, log_interval=10)
        try:
            model.save("ddpg_pendulum")
            print("Model saved successfully.")
        except Exception as e:
            print("Error while saving the model:", str(e))

    def test(self, model, env):
        vec_env = model.get_env()
        del model  # remove to demonstrate saving and loading
        model = DDPG.load("ddpg_pendulum")
        obs = vec_env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
            self.env.render("human")
