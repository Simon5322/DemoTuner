"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from <http://incompleteideas.net/sutton/book/code/pole.c>
permalink: <https://perma.cc/C9ZM-652R>
"""
import math
from typing import Optional, Union

import numpy as np
import yaml

import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.utils import seeding

from pathlib import Path


class PgEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    """
    config : 配置参数范围
    dbms : 对应的数据库
    """

    def __init__(self, config, obs, dbms, benchmark, throughput_default, latency_default, episode_len,
                 render_mode=None):

        self.dbms = dbms
        self.benchmark = benchmark
        self.conf = {}  # 当前的conf
        self.config = {**config}
        self.conf_default = {}
        self.obs = obs  # state metrics 和 workload

        self.episode_len = episode_len
        self.current_step = 0
        self.step = 0

        self.throughput_default = throughput_default
        self.latency_default = latency_default
        self.throughput_best = throughput_default
        self.latency_best = latency_default

        space = {}
        # 获得配置的范围
        for k, v in self.config.items():
            self.conf_default[k] = v.get('default')
            v_range = v.get('range')
            if v_range:  # discrete ranged parameter
                space[k] = (0, len(v_range))  # note: right-close range
            else:
                space[k] = (float(v['min']), float(v['max']))

        value_list = list(space.values())
        low_configs = np.array([k[0] for k in value_list])
        print("low" + str(low_configs))
        high_configs = np.array([k[1] for k in value_list])
        print(high_configs)
        self.action_space = spaces.Box(low=low_configs, high=high_configs, dtype=np.float32)

        # state的范围
        ob_space = {}
        for k, v in self.obs.items():
            v_range_obs = v.get('range')
            if v_range_obs:  # discrete ranged parameter
                ob_space[k] = (0, len(v_range_obs))  # note: right-close range
            else:
                ob_space[k] = (float(v['min']), float(v['max']))
        low_ob = np.array([k[0] for k in ob_space.values()])
        high_ob = np.array([k[1] for k in ob_space.values()])
        self.observation_space = spaces.Box(low=low_ob, high=high_ob, dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def reset(
            self,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
            random=False
    ):
        super().reset(seed=seed)
        if not random:
            self.conf = self.conf_default
            self.dbms.reset_config()
        self.current_step = 0
        obs = self._get_obs()
        return obs

    def step(self, action):
        print("a step going")
        self.current_step = self.current_step + 1
        self.step = self.step + 1
        # 将获取到的配置应用到数据艰中并生效
        self.conf = action
        for c, val in self.conf.items():
            self.dbms.set_param(c, val)
        self.dbms.make_conf_effect()

        # 进行测试
        # self.benchmark.run_benchmark(self.step)
        # result_path = "/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/results/pgResult" + "/pg_ycsb" + ".txt"
        throughput, latency = self.benchmark.run_benchmark(-1)  # self.benchmark.get_result(self.step)

        if throughput > self.throughput_default:
            if throughput > self.throughput_best:
                reward = 1
                self.throughput_best = throughput
            else:
                reward = 0.1
        else:
            reward = 0

        terminated = True if self.current_step > self.episode_len else False
        truncated = False
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def _get_obs(self):
        metrics = self.dbms.get_metrics()
        workload = {}
        other = {}
        obs = metrics
        return obs

    def _get_info(self):
        return None


if __name__ == '__main__':
    pass
    # params_setting_path = "/tuning_params_states/pg_params.yml"
    # params_setting = yaml.load(Path(params_setting_path).read_text(), Loader=yaml.FullLoader)
    # configs = {
    #     **params_setting,
    # }
    # env = PgEnv(configs)
    # print()
