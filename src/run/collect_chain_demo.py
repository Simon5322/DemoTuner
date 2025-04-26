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

from LLM import LLM_model_names
from LLM.LLM_tools import gpt_recommend_conf, align_gpt_conf
from LLM.tools import get_conf_description
from agent import DATE_CHAIN_DEMO
from src.benchmark.ForestRegressionEstimater import ForestEstimateBenchmark
from benchmark.TPCC import TPCC
from benchmark.TPCH import TPCH
from benchmark.simulateBenchmark import SimulateBenchmark
from hintsClasses.Hint import get_all_hints
from utils.util import clear_folder, clear_progress_result, state_to_real, get_current_workload_str, action_to_real, \
    real_to_action, append_to_pkl, save_results_to_csv

matplotlib.use('TkAgg')  # 使用TkAgg后端，你也可以根据需要选择其他后端

import dbms.factory

from benchmark.YCSB import YCSB
import gym_examples  # ！！！！！！！！！！！！这行不能删

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
if not fake_test:
    dbms.reset_config()
    dbms.reconfigure()
    throughput_list, latency_list = [], []
    print('test the default configuration')
    for i in range(repeat_times):
        dbms.create_new_usertable(fieldcount=fieldcount)
        bm.load_data()
        throughput, latency = bm.run_benchmark(times=-1)
        throughput_list.append(throughput)
        latency_list.append(latency)
    throughput_default, latency_default = sum(throughput_list) / len(throughput_list), sum(latency_list) / len(
        latency_list)
else:
    throughput_default, latency_default = bm.run_benchmark(-1)

performance_default = (throughput_default, latency_default)

conf_path = os.path.join(script_dir, f"../algorithms/RL/DDPGFD/config/{DDPGFD_config}.yaml")
with open(conf_path, 'r') as file:
    s_conf = yaml.load(file, Loader=yaml.FullLoader)
restore = bool(s_conf['train_config']['restore'])
env = gym.make('gym_examples/PgEnv-v0', conf_path=conf_path, config=conf, db_config=config, metrics=metrics,
               workload=workload, dbms=dbms,
               benchmark=bm,
               throughput_default=throughput_default, latency_default=latency_default, episode_len=episode_len,
               goal=goal, fieldcount=fieldcount, fake_test=fake_test, use_percent=use_percent,
               repeat_times=repeat_times)

n_episodes = 5
n_steps = 20
workload_type = 'RW_test'
model_name_list = [LLM_model_names.WenXinYiYan_Baidu]
#[LLM_model_names.WenXinYiYan_Baidu, LLM_model_names.TengXunHunYuan, LLM_model_names.XunFeiXingHuo, LLM_model_names.gpt35_turbo, LLM_model_names.gpt4]

for model_name in model_name_list:
    latency_result = []
    throughput_result = []
    demo_save_name = os.path.join(script_dir,
                             f"../../results/chain_demonstrations/{dbms_name}/{workload_type}/{model_name}/demo_1.pkl")
    save_dir = os.path.dirname(demo_save_name)
    os.makedirs(save_dir, exist_ok=True)
    for episode in range(n_episodes):
        demo_record = []
        default_conf = {k: v['default'] for k, v in conf.items()}
        current_conf_description = get_conf_description(dbms_name, conf, default_conf) #str({k: v['default'] for k, v in conf.items()})
        s, _ = env.reset()
        state_str = str(state_to_real(dbms_name, s[0]))  # 这里只包含state中第一部分
        latency = env.get_wrapper_attr('temp_latency')
        throughput = env.get_wrapper_attr('temp_throughput')
        workload_str = str(get_current_workload_str(args.cpath))
        for step in range(n_steps):
            last_performance = str(latency)
            current_conf = gpt_recommend_conf(state_str, workload_str, current_conf_description, last_performance, dbms_name, model_name)
            aligned_conf = align_gpt_conf(current_conf, conf)
            a = real_to_action(dbms_name, aligned_conf)
            s2, r, done, _, performance_tuple = env.step(a)
            demo_record.append(
                (s, a, r, s2, done, DATE_CHAIN_DEMO))
            s = s2
            state_str = str(s[0])
            latency, throughput = performance_tuple['latency'], performance_tuple['throughput']
            latency_result.append(latency)
            throughput_result.append(throughput)
        try:
            # 保存记录到指定路径
            append_to_pkl(demo_record, demo_save_name)
        except Exception as e:
            # 捕获异常并提示错误信息
            print(f"Failed to save demo record. Error: {e}")

    print(latency_result)
    print(throughput_result)
    performance_result_save_path = os.path.join(os.path.dirname(demo_save_name), "latency_throughput_result.csv")
    save_results_to_csv(latency_result, throughput_result, performance_result_save_path)
