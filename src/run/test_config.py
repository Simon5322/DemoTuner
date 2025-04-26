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
import random

import numpy as np
import yaml
import pandas as pd
import matplotlib


from IRL.Maximun_Entropy_IRL import load_reward_weights, reward_function, load_global_min_max
from benchmark.TPCH import TPCH
from benchmark.TPCC import TPCC
from hintsClasses.Hint import get_all_hints
from utils.util import clear_folder, clear_progress_result

matplotlib.use('TkAgg')  # 使用TkAgg后端，你也可以根据需要选择其他后端
import matplotlib.pyplot as plt

import dbms.factory

from benchmark.YCSB import YCSB

import gym_examples  # ！！！！！！！！！！！！这行不能删

script_dir = os.path.dirname(os.path.abspath(__file__))
os.path.dirname(os.path.abspath(__file__))

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

# 清空文件夹爱
clear_progress_result()

config_path = os.path.join(script_dir, f"../tuning_params_states/{dbms_name}/{dbms_name}_params.yml")
with open(config_path, "r") as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)
conf_names = [str(k) for k in {**conf}.keys()]
metrics_path = os.path.join(script_dir, f"../tuning_params_states/{dbms_name}/{dbms_name}_metrics.yml")
with open(metrics_path, "r") as f:
    metrics = yaml.load(f, Loader=yaml.FullLoader)
workload_setting_path = os.path.join(script_dir, f'../tuning_params_states/{dbms_name}/{dbms_name}_workload.yml')
with open(workload_setting_path, 'r') as f:
    workload = yaml.load(f, Loader=yaml.FullLoader)


# 建立dbms和benchamrk

def just_a_test(db, benchmark):
    db.create_new_usertable(fieldcount)
    benchmark.load_data()
    throughput, latency = benchmark.run_benchmark(-1)
    return throughput, latency


def test_config(db, benchmark, configs):
    for config_name, config_value in configs.items():
        db.set_param(config_name, int(config_value))
    db.reconfigure()
    db.create_new_usertable(fieldcount)
    benchmark.load_data()
    throughput, latency = benchmark.run_benchmark(-1)
    return throughput, latency


def test_each_config(db, benchmark, configs, repeat_times=1):
    throughput_list = []
    latency_list = []
    all_configs = []
    for param, value in configs.items():
        db.reset_config()
        db.make_conf_effect()
        db.reconfigure()
        db.set_param(param, value)
        db.make_conf_effect()
        db.reconfigure()
        print(str(param) + " : " + str(db.get_value_without_unit(param)))
        for i in range(repeat_times):
            db.create_new_usertable(fieldcount)
            benchmark.load_data()
            throughput, latency = benchmark.run_benchmark(-1)
            throughput_list.append(throughput)
            latency_list.append(latency)
            all_configs.append(param)
    return throughput_list, latency_list, all_configs


def random_test_configs_in_range(db, bench, name, num_steps=300):
    config_path = os.path.join(script_dir, f"../tuning_params_states/{dbms_name}/{dbms_name}_params.yml")
    with open(config_path, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    conf = {**conf}
    space = {}
    for k, v in conf.items():
        v_range = v.get('range')
        if v_range:  # discrete ranged parameter
            space[k] = (0, len(v_range))  # note: right-close range
        else:
            space[k] = (float(v['min']), float(v['max']))
    all_mean_throughput = []
    all_mean_latency = []
    config_lists = []
    config_names = [str(k) for k, v in space.items()]
    num_steps = num_steps  # 一共要测试多少次
    repeat_times = 1  # 每个配置重复测试多少次

    for i in range(num_steps, -1, -1):
        config_list = []
        for k, v in space.items():
            min_val, max_val = v
            # val = int(min_val + (max_val - min_val) * i / num_steps)
            val = random.randint(min_val, max_val)
            db.set_param(str(k), val)
            config_list.append(val)

        for l in range(repeat_times):
            config_lists.append(config_list)
        db.restart_dbms()
        db = dbms.factory.from_file(config)
        for n in range(repeat_times):
            db.create_new_usertable(fieldcount)
            bench.load_data()
            throughput, latency = bench.run_benchmark(-1)
            all_mean_throughput.append(throughput)
            all_mean_latency.append(latency)

    df_latency = pd.DataFrame(all_mean_latency, columns=['latency'])
    conf_col = pd.DataFrame(config_lists, columns=config_names)
    df_latency = pd.concat([df_latency, conf_col], axis=1)
    df_latency.to_excel(f'/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/results/test_configs/latency{name}.xlsx',
                        index=False)
    df_throughput = pd.DataFrame(all_mean_throughput, columns=['throughput'])
    conf_col = pd.DataFrame(config_lists, columns=config_names)
    df_throughput = pd.concat([df_throughput, conf_col], axis=1)
    df_throughput.to_excel(
        f'/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/results/test_configs/throughput{name}.xlsx', index=False)


def test_default_reward(test_state):
    work_space_dir = os.path.join(script_dir, '../../results/reward_weights')
    reward_weights = load_reward_weights(os.path.join(work_space_dir, 'reward_weight.txt'))
    global_min, global_max = load_global_min_max(os.path.join(work_space_dir, 'global_min_max.json'))
    reward = reward_function(test_state, reward_weights, global_min, global_max)
    return reward


db = dbms.factory.from_file(config)  # pg or mysql
db.reset_config()
db.make_conf_effect()
db.reconfigure()
ycsb = YCSB(dbms_name=dbms_name, save_folder_name=save_folder_name)
tpch = TPCH(benchmark_name='tpch', dbms_name=dbms_name)
tpcc = TPCC(benchmark_name='tpcc', dbms_name=dbms_name)
# random_test_configs_in_range(db, tpcc, 20)


# configs_to_test = {
# "innodb_buffer_pool_size": 144,
#   "innodb_io_capacity": 100,
#   "innodb_log_buffer_size": 5,
#   "innodb_log_file_size": 1020,
#   "innodb_log_files_in_group": 2,
#   "innodb_lru_scan_depth": 120,
#   "innodb_purge_threads": 1,
#   "innodb_read_io_threads": 2,
#   "innodb_write_io_threads": 1,
#   "innodb_thread_concurrency": 0,
#   "join_buffer_size": 1965,
#   "max_heap_table_size": 1101,
#   "read_buffer_size": 506,
#   "sort_buffer_size": 1,
#   "preload_buffer_size": 3174,
#   "table_open_cache": 70,
#   "thread_cache_size": 16384,
#   "tmp_table_size": 1,
#   "net_buffer_length": 1013
#     }

configs_to_test = {
  "work_mem": 64,
  "wal_buffers": 451,
  "temp_buffers": 262,
  "shared_buffers": 1048576,
  "effective_cache_size": 141361,
  "maintenance_work_mem": 1048576,
  "max_connections": 100,
  "bgwriter_lru_multiplier": 0,
  "backend_flush_after": 254,
  "bgwriter_delay": 10000,
  "max_parallel_workers": 17,
  "hash_mem_multiplier": 982,
  "checkpoint_flush_after": 256,
  "max_wal_size": 10086,
  "join_collapse_limit": 4,
  "vacuum_cost_page_dirty": 10000,
  "min_parallel_table_scan_size": 130506,
  "min_parallel_index_scan_size": 541
}

bench = ycsb
# for i in range(20):
#     random_test_configs_in_range(db, bench, str(i), 10)

execution_time = []
for i in range(20):
    _, p = test_config(db, bench, configs_to_test)
    execution_time.append(p)

print(execution_time)
#throughput, latency = just_a_test(db, bench)
# state = np.array([[latency, throughput]])
# reward = test_default_reward(state)
# print(f"reward:{reward}")



# configs_to_test = data = {'effective_cache_size': '250000', 'maintenance_work_mem': '1024000', 'max_parallel_workers_per_gather': '3', 'commit_delay': '0', 'max_wal_senders': '0', 'wal_writer_delay': '100', 'shared_buffers': '2000000', 'max_connections': '550', 'checkpoint_completion_target': '0'}
#test_config(db, ycsb, configs_to_test)



# pg.create_new_usertable(fieldcount=fieldcount)
# ycsb.load_data()


# best_config = {'innodb_thread_concurrency': '0', 'innodb_autoextend_increment': '512000000', 'read_rnd_buffer_size': '256000', 'join_buffer_size': '512000', 'innodb_lock_wait_timeout': '60', 'net_buffer_length': '1000000', 'innodb_max_purge_lag_delay': '300000', 'innodb_max_purge_lag': '500000', 'lock_wait_timeout': '43200', 'innodb_buffer_pool_size': '6000000000'}
#
#
#
# for param, value in best_config.items():
#     print(str(param) + " : " +str(db.get_value_without_unit(param)))
#
# all_configs = []
#
# for param, value in best_config.items():
#     db.reset_config()
#     db.make_conf_effect()
#     db.reconfigure()
#     db.set_param_without_unit(param, value)
#     db.make_conf_effect()
#     db.reconfigure()
#     print(str(param) + " : " + str(db.get_value_without_unit(param)))
#     for i in range(repeat_times):
#         db.create_new_usertable(fieldcount)
#         ycsb.load_data()
#         throughput, latency = ycsb.run_benchmark(-1)
#         all_mean_throughput.append(throughput)
#         all_mean_latency.append(latency)
#         all_configs.append(param)
#
#
# print(all_mean_latency)
# print(all_mean_throughput)
# print(all_configs)
