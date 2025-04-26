'''
Created on Nov 13, 2021

@author: immanueltrummer
'''
import math
import os
import shutil
import subprocess
# from pybullet_utils.util import set_global_seeds
from argparse import ArgumentParser
from configparser import ConfigParser

import gymnasium as gym
import joblib
import yaml
import pandas as pd
import matplotlib

import json

from IRL.Maximun_Entropy_IRL import load_reward_weights, reward_function, load_global_min_max
from LLM import LLM_model_names
from LLM.LLM_tools import gpt_recommend_conf, align_gpt_conf
from benchmark.ForestRegressionEstimater import ForestEstimateBenchmark
from benchmark.TPCC import TPCC
from benchmark.TPCH import TPCH
from benchmark.simulateBenchmark import SimulateBenchmark
from hintsClasses.Hint import get_all_hints
from utils.util import clear_folder, clear_progress_result, state_to_real, get_current_workload_str, action_to_real, \
    real_to_action

matplotlib.use('TkAgg')  # 使用TkAgg后端，你也可以根据需要选择其他后端

import dbms.factory

from benchmark.YCSB import YCSB
import gym_examples  # ！！！！！！！！！！！！这行不能删


def test_config(db, benchmark, configs):
    for config_name, config_value in configs.items():
        db.set_param(config_name, int(config_value))
    db.reconfigure()
    db.create_new_usertable(fieldcount)
    benchmark.load_data()
    throughput, latency = benchmark.run_benchmark(-1)
    return throughput, latency

script_dir = os.path.dirname(os.path.abspath(__file__))
arg_parser = ArgumentParser(description='DDPGFD-gpt: use gpt to guide RL parameter tuning')
arg_parser.add_argument('cpath', type=str, help='Path to configuration file')
arg_parser.add_argument('--eval', help='Evaluation mode', action='store_true', default=False)
arg_parser.add_argument('--collect', help='Collect Demonstration Data', action='store_true', default=False)
arg_parser.add_argument('-n_collect', help='Number of episode for demo collection', type=int, default=100)
args = arg_parser.parse_args()
config = ConfigParser()
config.read(args.cpath)

dbms_name = config['DATABASE']['dbms']
benchmark = config['BENCHMARK']['name']
workload = config['BENCHMARK']['workload']
# tuning_metric = config['BENCHMARK']['tuning_metric']
# tuning_times = int(config['BENCHMARK']['tuning_times'])
use_percent = True if config['LEARNING']['use_percent'] == 'True' else False
episode_len = int(config['LEARNING']['episode_len'])
goal = config['LEARNING']['goal']  # latency, throughput
fieldcount = int(config['WORKLOAD']['fieldcount'])
save_folder_name = config['SETTING']['save_name']
fake_test = True if config['LEARNING']['fake_test'] == 'True' else False
repeat_times = int(config['LEARNING']['repeat_times'])

config_path = os.path.join(script_dir, f"../../tuning_params_states/{dbms_name}/{dbms_name}_params.yml")
with open(config_path, "r") as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)
conf_names = [str(k) for k in {**conf}.keys()]

metrics_path = os.path.join(script_dir, f"../../tuning_params_states/{dbms_name}/{dbms_name}_metrics.yml")
with open(metrics_path, "r") as f:
    metrics = yaml.load(f, Loader=yaml.FullLoader)

workload_setting_path = os.path.join(script_dir, f'../../tuning_params_states/{dbms_name}/{dbms_name}_workload.yml')
with open(workload_setting_path, 'r') as f:
    workload = yaml.load(f, Loader=yaml.FullLoader)
# 建立dbms和benchamrk

dbms = dbms.factory.from_file(config)  # pg or mysql

bm = None
if benchmark == "YCSB":
    bm = YCSB(dbms_name=dbms_name, save_folder_name=save_folder_name)
elif benchmark == 'TPCH':
    bm = TPCH("tpch", "pg")  # 现在只有postgresql上的TPCH benchbase
elif benchmark == 'TPCC':
    bm = TPCC("tpcc", "pg")  # 现在只有postgresql上的TPCH benchbase
elif benchmark == 'Simulate':
    bm = SimulateBenchmark('simulate', conf)
elif benchmark == 'ForestEstimator':
    bm = ForestEstimateBenchmark('forestEstimator', conf, dbms_name)
else:
    raise KeyError(f"benchmark {benchmark} is not valid")

if args.collect:
    DDPGFD_config = 's0'
    print(f'collect data phrase, use {DDPGFD_config}')
else:
    DDPGFD_config = 's1'
    print(f'train phrase, use {DDPGFD_config}')

# 初始化数据库配置 测试当前负载的数据库状态


throughput_default, latency_default = 1, 2

conf_path = os.path.join(script_dir, f"../../algorithms/RL/DDPGFD/config/{DDPGFD_config}.yaml")
with open(conf_path, 'r') as file:
    s_conf = yaml.load(file, Loader=yaml.FullLoader)
restore = bool(s_conf['train_config']['restore'])
env = gym.make('gym_examples/PgEnv-v0', conf_path=conf_path, config=conf, db_config=config, metrics=metrics,
               workload=workload, dbms=dbms,
               benchmark=bm,
               throughput_default=throughput_default, latency_default=latency_default, episode_len=episode_len,
               goal=goal, fieldcount=fieldcount, fake_test=fake_test, use_percent=use_percent,
               repeat_times=repeat_times)

# configs_test_dict_default = {
#     "work_mem": 4096,
#     "wal_buffers": 512,
#     "temp_buffers": 1024,
#     "shared_buffers": 16384,
#     "effective_cache_size": 524288,
#     "maintenance_work_mem": 65536,
#     "max_connections": 100,
#     "bgwriter_lru_multiplier": 2,
#     "backend_flush_after": 0,
#     "bgwriter_delay": 200,
#     "max_parallel_workers": 8,
#     "hash_mem_multiplier": 2,
#     "checkpoint_flush_after": 32,
#     "max_wal_size": 1024,
#     "join_collapse_limit": 8,
#     "vacuum_cost_page_dirty": 20,
#     "min_parallel_table_scan_size": 1024,
#     "min_parallel_index_scan_size": 64,
#     "max_parallel_workers_per_gather": 2
# }
#
# configs_test_dict_best = {
#     "work_mem": 64,
#     "wal_buffers": 65535,
#     "temp_buffers": 100,
#     "shared_buffers": 1041568,
#     "effective_cache_size": 131072,
#     "maintenance_work_mem": 1048576,
#     "max_connections": 100,
#     "bgwriter_lru_multiplier": 0,
#     "backend_flush_after": 0,
#     "bgwriter_delay": 9948,
#     "max_parallel_workers": 8,
#     "hash_mem_multiplier": 995,
#     "checkpoint_flush_after": 255,
#     "max_wal_size": 137,
#     "join_collapse_limit": 996,
#     "vacuum_cost_page_dirty": 10000,
#     "min_parallel_table_scan_size": 0,
#     "min_parallel_index_scan_size": 131072,
#     "max_parallel_workers_per_gather": 1013
# }
configs_test_dict_default = {
  "innodb_buffer_pool_size": 1,
  "innodb_io_capacity": 200,
  "innodb_log_buffer_size": 16,
  "innodb_log_file_size": 48,
  "innodb_log_files_in_group": 2,
  "innodb_lru_scan_depth": 1024,
  "innodb_purge_threads": 4,
  "innodb_read_io_threads": 4,
  "innodb_write_io_threads": 4,
  "innodb_thread_concurrency": 0,
  "join_buffer_size": 2,
  "max_heap_table_size": 1024,
  "read_buffer_size": 16,
  "sort_buffer_size": 8,
  "preload_buffer_size": 32,
  "table_open_cache": 4000,
  "thread_cache_size": 9,
  "tmp_table_size": 16384,
  "net_buffer_length": 16
}

configs_test_dict_best ={
  "innodb_buffer_pool_size": 143,
  "innodb_io_capacity": 167,
  "innodb_log_buffer_size": 1,
  "innodb_log_file_size": 1020,
  "innodb_log_files_in_group": 2,
  "innodb_lru_scan_depth": 100,
  "innodb_purge_threads": 1,
  "innodb_read_io_threads": 1,
  "innodb_write_io_threads": 1,
  "innodb_thread_concurrency": 0,
  "join_buffer_size": 1995,
  "max_heap_table_size": 1,
  "read_buffer_size": 511,
  "sort_buffer_size": 1,
  "preload_buffer_size": 38,
  "table_open_cache": 1,
  "thread_cache_size": 0,
  "tmp_table_size": 52,
  "net_buffer_length": 1024
}



config_test_list = [configs_test_dict_default, configs_test_dict_best]
reward_results = []

for config_to_test in config_test_list:
    s, _ = env.reset()

    configs_test_list = [v for v in config_to_test.values()]
    a = real_to_action(dbms_name, configs_test_list)

    s2, r, done, _, performance_tuple = env.step(a)
    reward_weight_path = os.path.join(script_dir, '../../../results/reward_weights/reward_weight.txt')
    reward_weights = load_reward_weights(reward_weight_path)
    global_min, global_max = load_global_min_max(os.path.join(script_dir,
                                                              '../../../results/reward_weights/global_min_max.json'))
    reward = reward_function(s2, reward_weights, global_min, global_max)
    reward_results.append(reward)
    print(f"Test state: {s2}")
    print(f"Reward for test state: {reward}")

print(reward_results)

