"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from <http://incompleteideas.net/sutton/book/code/pole.c>
permalink: <https://perma.cc/C9ZM-652R>
"""
import asyncio
import configparser
import math
import os
import time

import dbms
import torch
from typing import Optional, Union

import numpy as np
import spaces
import yaml

import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.utils import seeding

from pathlib import Path

from Global_Variables import globalCurrentAction
from IRL.Maximun_Entropy_IRL import load_reward_weights, reward_function, load_global_min_max
from benchmark.ForestRegressionEstimater import ForestEstimateBenchmark
from benchmark.YCSB import YCSB
from benchmark.simulateBenchmark import SimulateBenchmark
from dbms.myMySql import MySQLconfig
from dbms.postgres import PgConfig
from training_utils import load_conf
# from tuning_run import config
import dbms.factory
import Global_Variables
from utils.util import record_cpu_utilization

script_path = os.path.dirname(os.path.abspath(__file__))
class PgEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    """
    db_config : mysql.ini 或 pg.ini
    conf_path :s0,或s1的位置
    config : 数据库的配置参数范围
    dbms : 对应的数据库
    """

    def __init__(self, conf_path, config, db_config, metrics, workload, dbms, benchmark, throughput_default,
                 latency_default,
                 episode_len, goal, fieldcount, fake_test,
                 use_percent, repeat_times, render_mode=None):
        self.full_conf = load_conf(conf_path)
        self.conf = self.full_conf.train_config
        self.db_config = db_config
        self.fake_test = fake_test
        self.repeat_times = repeat_times

        self.dbms = dbms
        self.benchmark = benchmark
        self.config = {**config}  # 配置参数
        self.conf_default = {}
        self.use_percent = use_percent

        self.space = {}
        # 获得配置的范围
        for k, v in self.config.items():
            self.conf_default[k] = v.get('default')
            v_range = v.get('range')
            if v_range:  # discrete ranged parameter
                self.space[k] = (0, len(v_range))  # note: right-close range
            else:
                self.space[k] = (float(v['min']), float(v['max']))
        self.current_conf = self.conf_default  # 当前的conf
        self.last_conf = self.conf_default
        self.conf_names = [key for key in self.space.keys()]
        self.metrics_name = [m for m in metrics.keys()]
        self.metrics = metrics  # state metrics 和 workload
        self.obs_keys_list = [key for key in self.metrics.keys()] + [key for key in workload.keys()]
        self.conf_keys_list = [key for key in self.config.keys()]
        self.workload = workload
        self.fieldcount = fieldcount
        self.episode_len = episode_len
        self.current_step = 0
        self.total_step = 0
        self.reward_function_type = self.conf.reward_function_type

        self.throughput_default = throughput_default
        self.latency_default = latency_default
        self.throughput_best = throughput_default
        self.latency_best = latency_default
        self.temp_throughput = 0
        self.temp_latency = 0

        # value_list = list(space.values())
        self.low_configs = np.array([k[0] for k in self.space.values()])
        self.high_configs = np.array([k[1] for k in self.space.values()])
        self.config_dims = len(self.low_configs)
        if self.use_percent:
            self.action_space = spaces.Box(low=np.full(self.config_dims, -1), high=np.full(self.config_dims, 1),
                                           dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=np.array(self.low_configs), high=np.array(self.high_configs),
                                           dtype=np.float32)
            # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.config_dims,), dtype=np.float32)


        # state的范围（metrics，workload）
        # metrics范围
        self.ob_space = {}
        for k, v in metrics.items():
            v_range_obs = v.get('range')
            if v_range_obs:  # discrete ranged parameter
                self.ob_space[k] = (0, len(v_range_obs))  # note: right-close range
            else:
                self.ob_space[k] = (float(v['min']), float(v['max']))
        low_ob = np.array([k[0] for k in self.ob_space.values()])
        high_ob = np.array([k[1] for k in self.ob_space.values()])
        metrics_space = spaces.Box(low=low_ob, high=high_ob, dtype=np.float32)

        self.metrics_num = len(self.ob_space)
        # workload范围
        continues_workload_space, discrete_workload_space, self.continues_workload_num, self.discrete_workload_num = self.workload_to_float()

        self.observation_space = spaces.Tuple((metrics_space, continues_workload_space) + discrete_workload_space)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.goal = goal

        self.result_X = []
        self.result = []
        self.hint_meet = []

        self.result_X.append([v for v in self.conf_default.values()])
        self.result.append(throughput_default) if self.goal == "throughput" else self.result.append(latency_default)


    def workload_to_float(self):
        workload_range_Box = {}
        workload_range_discrete = {}
        continues_workload_num = 0
        discrete_workload_num = 0

        for k, v in self.workload.items():
            typ = v.get('type')
            if typ == 'int':
                workload_range_Box[k] = (int(v['min']), int(v['max']))
                continues_workload_num += 1
            elif typ == 'float':
                workload_range_Box[k] = (float(v['min']), float(v['max']))
                continues_workload_num += 1
            elif typ == 'bool':
                workload_range_discrete[k] = spaces.Discrete(2)
                discrete_workload_num += 1
            elif typ == 'enum':
                workload_range_discrete[k] = spaces.Discrete(int(v.get('enum')))
                discrete_workload_num += 1
            else:
                raise ValueError

        low = np.array([k[0] for k in workload_range_Box.values()])
        high = np.array([k[1] for k in workload_range_Box.values()])
        continues_space = spaces.Box(low=low, high=high, dtype=np.float32)

        discrete_space = tuple(num_value for num_value in workload_range_discrete.values())
        return continues_space, discrete_space, continues_workload_num, discrete_workload_num

    def reset(
            self,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
            random=False
    ):

        super().reset(seed=seed)
        self.benchmark.set_default_global_conf()  # 用作Simulate的时候用
        if not random:
            if not self.fake_test:
                self.current_conf = self.conf_default
                self.dbms.reset_config()

                # self.dbms.make_conf_effect()
                self.dbms.reconfigure()
                # self.dbms.restart_dbms()
                pg = dbms.factory.from_file(self.db_config)
                self.dbms = pg
        self.current_step = 0
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        Global_Variables.globalCurrentAction = action    # 用作simulate

        # self.result_X.append(action)
        reward = 0
        self.current_step = self.current_step + 1
        self.total_step = self.total_step + 1
        print(" step " + str(self.total_step) + " going ")

        # 设置新配置
        reward = self.action_to_value(action, reward)
        self.check_conf()
        self.result_X.append([value for value in self.current_conf.values()])
        if not self.fake_test:
            for config_name, config_value in self.current_conf.items():
                self.dbms.set_param(config_name, int(config_value))
            self.dbms.make_conf_effect()
            self.dbms.reconfigure()
            #self.dbms.restart_dbms()
            # self.dbms = dbms.factory.from_file(self.db_config)
            for config_name, config_value in self.current_conf.items():
                self.dbms.check_conf_set(config_name, config_value)

        conf_values = ' '.join([f'{conf_name} : {self.dbms.get_value(conf_name)}' for conf_name in self.conf_names])
        # print(conf_values)

        # 测试 获得state‘
        observation = self._get_obs()
        info = self._get_info()
        if self.temp_throughput * self.temp_latency != 0:
            throughput, latency = self.temp_throughput, self.temp_latency
            self.temp_throughput, self.temp_latency = 0, 0
        else:
            raise ValueError('temp_throughput, temp_latency出现了0,没有被正确赋值')
        if throughput > self.throughput_best:
            self.throughput_best = throughput
        if latency < self.latency_best:
            self.latency_best = latency
        reward = self.reward_function(reward, throughput, latency, self.reward_function_type, s=observation)

        # reward guide
        # if self.conf.reward_guide:
        #     for hint in self.hint_meet:
        #         conf_name = hint.get_conf_name()
        #         min_val, max_val = self.space[conf_name]
        #
        #         index = self.conf_names.index(conf_name)
        #         current_a = min_val + (max_val - min_val) * (action[index] + 1) / 2
        #         tuning_value = hint.get_tuningValue().get_value(current_a, min_val, max_val, conf_name)
        #         suggest_a = 2 * ((tuning_value - min_val) / (max_val - min_val)) - 1
        #         if (action[index] > 0 and suggest_a < 0) or (action[index] < 0 and suggest_a > 0):
        #             reward -= reward / 10
        #             break

        terminated = True if self.current_step > self.episode_len-1 else False
        truncated = False


        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def reward_function(self, reward, throughput, latency, reward_func="speed", s=None):
        if reward_func == 'basic':
            if self.goal == "throughput":
                print(str(self.goal) + " : " + str(throughput))
                self.result.append(throughput)
                if throughput > self.throughput_default:
                    if throughput > self.throughput_best:
                        reward = reward + 1
                    else:
                        reward = reward
                else:
                    reward = reward
            elif self.goal == "latency":
                print(str(self.goal) + str(latency))
                self.result.append(latency)
                if latency < self.latency_default:
                    if latency < self.latency_best:
                        reward = reward + 1
                        self.latency_best = latency
                    else:
                        reward = reward
                else:
                    reward = reward
            else:
                raise ValueError('Invalid optimization goal' + str(self.goal))
            return reward
        elif reward_func == 'speed':   # 默认使用加速比
            if self.goal == "throughput":
                print(str(self.goal) + " : " + str(throughput))
                self.result.append(throughput)
                if throughput > self.throughput_default:
                    reward = (throughput - self.throughput_default) / self.throughput_default
                else:
                    reward -= 0  # 0.001
            elif self.goal == "latency":
                print(str(self.goal) + str(latency))
                self.result.append(latency)
                if latency < self.latency_default:
                    reward = float(self.conf.reward_func_multiple_ratio)*(self.latency_default - latency) / self.latency_default
                else:
                    reward = float(self.conf.reward_func_multiple_ratio_negative)*(self.latency_default - latency) / self.latency_default
                # if latency < self.latency_default:
                #     reward = (self.latency_default - latency) / self.latency_default
                # else:
                #     reward -= 0  # 0.001
            else:
                raise ValueError('Invalid optimization goal' + str(self.goal))
            return reward
        elif reward_func == 'IRL':
            if self.goal == "throughput":
                raise KeyError(f"IRL is not available for {self.goal}")
            elif self.goal == 'latency':
                print(str(self.goal) + " : " + str(latency))
                self.result.append(latency)
                reward_weight_path = os.path.join(script_path, f"../../../../../../results/reward_weights/reward_weight.txt")
                reward_weights = load_reward_weights(reward_weight_path)
                global_min, global_max = load_global_min_max(os.path.join(script_path,
                                                                          f'../../../../../../results/reward_weights/global_min_max.json'))

                reward = reward_function(s, reward_weights, global_min, global_max)
                print(f"reward : {reward}")
                return reward
        raise ValueError('Invalid reward function' + str(reward_func))



    """
    _get_obs获取state
    """
    def _get_obs(self):
        # 测试前的metrics
        throughput_list, latency_list = [], []
        metricsD_before_task_list = []
        metricsD_list = []
        metricsD_before_task = {}
        metricsD = {}
        for i in range(self.repeat_times):
            metricsD_before_task_list.append(self.dbms.get_metrics())
            if not self.fake_test:
                self.dbms.create_new_usertable(fieldcount=int(self.fieldcount))
                self.benchmark.load_data()
                throughput, latency = self.benchmark.run_benchmark(-1)
                # self.dbms.query_one('select pg_stat_clear_snapshot()')
                self.temp_throughput, self.temp_latency = throughput, latency
                throughput_list.append(throughput)
                latency_list.append(latency)
            else:
                self.temp_throughput, self.temp_latency = self.benchmark.run_benchmark(-1)
                throughput_list.append(self.temp_throughput)
                latency_list.append(self.temp_latency)
                #print('!!!!!!!!!!!!!!!!!!!fake test!!!!!!!!!!!!!!!!!!!')
                # time.sleep(0.1)  # 等待收集metrics
                # 测试过后的metrics
            metricsD_list.append(self.dbms.get_metrics())

        self.temp_throughput, self.temp_latency = sum(throughput_list) / len(throughput_list), sum(
            latency_list) / len(latency_list)

        # state加入性能指标

        for key in metricsD_before_task_list[0]:  # 假设所有字典具有相同的键
            sum_values = sum(d[key] for d in metricsD_before_task_list)  # 计算同一键的值的总和
            avg_value = sum_values / len(metricsD_before_task_list)  # 计算平均值
            metricsD_before_task[key] = avg_value

        for key in metricsD_list[0]:  # 假设所有字典具有相同的键
            sum_values = sum(d[key] for d in metricsD_list)  # 计算同一键的值的总和
            avg_value = sum_values / len(metricsD_list)  # 计算平均值
            metricsD[key] = avg_value

        # 加入性能指标
        metricsD['latency'] = self.temp_latency
        metricsD['throughput'] = self.temp_throughput

        collected_metrics = [key for key, v in self.metrics.items() if
                             bool(v['collect_metric'])]

        for k, v in zip(metricsD, metricsD_before_task):
            if k in collected_metrics:
                metricsD[k] = metricsD[k] - metricsD_before_task[k]

        # 负载的指标
        continues_workload, discrete_workload = self.get_workload()

        # 保证metrics的顺序
        metrics = []
        metricsD_keys = [key for key in self.metrics.keys()]
        for k in metricsD_keys:
            if k in metricsD.keys():
                metrics.append(metricsD[k])
        metrics_array = np.array(metrics)
        metrics_array = metrics_array.astype(np.float32)

        cworkload = [m for m in continues_workload.values()]
        cworkload_array = np.array(cworkload)
        cworkload_array = cworkload_array.astype(np.float32)
        print('State : ', end='')
        for name, metric in zip(self.metrics_name, metrics_array):
            print(name + ':' + str(metric), end=' ')
        print('')
        dworkload = tuple(m for m in discrete_workload.values())

        return metrics_array, cworkload_array, *dworkload

    def _get_info(self):
        return {'latency': self.temp_latency, 'throughput': self.temp_throughput}

    def action_to_value(self, action, reward):
        if self.use_percent:
            for current_conf_name, current_conf_value, percent in zip(self.current_conf.keys(),
                                                                      self.current_conf.values(), action):
                if percent < -1:
                    percent = -1
                if percent > 1:
                    percent = 1
                real_config = current_conf_value + (percent / 2) * current_conf_value
                low_bound = self.space[current_conf_name][0]
                high_bound = self.space[current_conf_name][1]
                if real_config > high_bound:
                    reward = reward
                    self.current_conf[current_conf_name] = high_bound
                elif real_config < low_bound:
                    reward = reward
                    self.current_conf[current_conf_name] = low_bound
                else:
                    self.current_conf[current_conf_name] = real_config
        else:
            for current_conf_name, current_conf_value, value, v in zip(self.current_conf.keys(),
                                                                       self.current_conf.values(), action,
                                                                       self.config.values()):
                if value < -1:
                    value = -1
                if value > 1:
                    value = 1
                low_bound = self.space[current_conf_name][0]
                high_bound = self.space[current_conf_name][1]
                real_config = low_bound + (high_bound - low_bound) * (value + 1) / 2
                if v.get('float') == 'yes':
                    self.last_conf[current_conf_name] = self.current_conf[current_conf_name]
                    self.current_conf[current_conf_name] = round(float(real_config), 1)
                else:
                    self.last_conf[current_conf_name] = self.current_conf[current_conf_name]
                    self.current_conf[current_conf_name] = int(real_config)

        return reward

    def update_hints(self, new_hints_meet):
        self.hint_meet = new_hints_meet

    # 16gb内存 不能超了
    def check_conf(self):
        memory_max = 16777216  # kb
        if isinstance(self.dbms, PgConfig):  # MySQLconfig):

            conf_using_mem = ['work_mem', 'wal_buffers', 'temp_buffers', 'shared_buffers', 'effective_cache_size',
                              'maintenance_work_mem']
            conf_8kb = ['wal_buffers', 'temp_buffers', 'shared_buffers', 'effective_cache_size']
            mem_used = 0
            for config_name, config_value in self.current_conf.items():
                if config_name in conf_using_mem:
                    mem_used += config_value * 8 if config_name in conf_8kb else config_value

            if mem_used > memory_max:
                ratio = memory_max / mem_used
                for config_name, config_value in self.current_conf.items():
                    val = config_value * ratio
                    min_val, max_val = self.space[config_name]
                    if val < min_val:
                        val = min_val
                    elif val > max_val:
                        val = max_val
                    self.current_conf[config_name] = val

        elif isinstance(self.dbms, MySQLconfig):
            conf_using_mem = ['innodb_buffer_pool_size']
            conf_8kb = []
        else:
            raise KeyError(f'{self.dbms.__class__} is not a mysql or postgres instance')


    def get_workload(self):
        continues_settings = {}
        discrete_settings = {}
        if isinstance(self.benchmark, YCSB) or isinstance(self.benchmark, SimulateBenchmark) or isinstance(self.benchmark, ForestEstimateBenchmark) :
            config = configparser.ConfigParser()
            dbms_name = None
            if isinstance(self.dbms, PgConfig):  # MySQLconfig):
                dbms_name = 'pg'
            elif isinstance(self.dbms, MySQLconfig):
                dbms_name = 'mysql'
            else:
                raise KeyError(f'{self.dbms.__class__} is not a mysql or postgres instance')

            config.read(os.path.join(script_path, f'../../../../../../config/{dbms_name}.ini'))
            workload_file_path = os.path.join(script_path, f'../../../../../../src/tuning_params_states/{dbms_name}/{dbms_name}_workload.yml')
            with open(workload_file_path, 'r') as f:
                workload_settings = yaml.load(f, Loader=yaml.FullLoader)

            for k, v in workload_settings.items():
                if v['type'] == 'enum' or v['type'] == 'bool':
                    discrete_settings[k] = config.getint('WORKLOAD', str(k))
                elif v['type'] == 'float':
                    continues_settings[k] = config.getfloat('WORKLOAD', str(k))
                elif v['type'] == 'int':
                    continues_settings[k] = config.getint('WORKLOAD', str(k))
                else:
                    raise KeyError('type : {} is not a valid type'.format(k))
            return continues_settings, discrete_settings



            # for k, v in workload_settings.items():
            #     if v['type'] == 'enum' or v['type'] == 'bool':
            #         discrete_setting_names.append(k)
            #     elif v['type'] == 'int' or v['type'] == 'float':
            #         continues_setting_names.append(k)
            #     else:
            #         raise KeyError('{} is not a valid type'.format(k))

            # workload['fieldcount'] = config.getint('WORKLOAD', 'fieldcount')
            # workload['fieldlength'] = config.getint('WORKLOAD', 'fieldlength')
            # workload['minfieldlength'] = config.getint('WORKLOAD', 'minfieldlength')
            # workload['readallfields'] = config.getint('WORKLOAD', 'readallfields')
            # workload['writeallfields'] = config.getint('WORKLOAD', 'writeallfields')
            # workload['readproportion'] = config.getfloat('WORKLOAD', 'readproportion')
            # workload['updateproportion'] = config.getfloat('WORKLOAD', 'updateproportion')
            # workload['insertproportion'] = config.getfloat('WORKLOAD', 'insertproportion')
            # workload['scanproportion'] = config.getfloat('WORKLOAD', 'scanproportion')
            # workload['readmodifywriteproportion'] = config.getfloat('WORKLOAD', 'readmodifywriteproportion')
            # workload['minscanlength'] = config.getint('WORKLOAD', 'minscanlength')
            # workload['maxscanlength'] = config.getint('WORKLOAD', 'maxscanlength')
            # workload['scanlengthdistribution'] = config.get('WORKLOAD', 'scanlengthdistribution')
            # workload['insertstart'] = config.getint('WORKLOAD', 'insertstart')
            # workload['insertcount'] = config.get('WORKLOAD', 'insertcount')
            # workload['zeropadding'] = config.getint('WORKLOAD', 'zeropadding')
            # workload['insertorder'] = config.get('WORKLOAD', 'insertorder')
            # workload['requestdistribution'] = config.get('WORKLOAD', 'requestdistribution')


if __name__ == '__main__':
    pass
    # params_setting_path = "/tuning_params_states/pg_params.yml"
    # params_setting = yaml.load(Path(params_setting_path).read_text(), Loader=yaml.FullLoader)
    # configs = {
    #     **params_setting,
    # }
    # env = PgEnv(configs)
    # print()
