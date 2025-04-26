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
import yaml
import pandas as pd
import matplotlib

from benchmark.ForestRegressionEstimater import ForestEstimateBenchmark
from benchmark.TPCC import TPCC
from benchmark.TPCH import TPCH
from benchmark.simulateBenchmark import SimulateBenchmark
from hintsClasses.Hint import get_all_hints
from utils.util import clear_folder, clear_progress_result

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import dbms.factory

from algorithms.RL.DDPGFD.train import RLTrainer
from algorithms.RL.agent.Ddpg import DDPG, Ddpg
from algorithms.bo import BOenvOptimizer

from benchmark.YCSB import YCSB

import gym_examples

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

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

hints = get_all_hints(dbms_name=dbms_name, extracted=False)

evaluate = args.eval
print("evaluate 模式是" + str(evaluate))
if restore:
    shutil.copytree(
        os.path.join(script_dir, f"../../results/{dbms_name}Result/{save_folder_name}/progress"),
        os.path.join(script_dir, f"../algorithms/RL/DDPGFD/progress"),
        dirs_exist_ok=True)

trainer = RLTrainer(conf_path, env, len(conf), len(metrics) + len(workload), hints, performance_default, goal,
                    evaluate)

if args.eval:
    trainer.eval(save_fig=False)
elif args.collect:
    clear_folder(os.path.join(script_dir, "../algorithms/RL/DDPGFD/data/demo"))
    shutil.copy(os.path.join(script_dir, f"../../config/{dbms_name}.ini"),
                os.path.join(script_dir, f"../algorithms/RL/DDPGFD/data/demo/{dbms_name}.ini"))

    trainer.collect_demo_use_demonstration(args.n_collect)
else:
    if trainer.conf.pretrain_demo:  # use demonstration to pretrain
        trainer.pretrain()
    trainer.train()

os.makedirs(os.path.join(script_dir, f"../../results/{dbms_name}Result/{save_folder_name}"), exist_ok=True)
# if not (args.eval or args.collect):


##
if not args.collect:
    # 数据表格保存
    result_X = env.get_wrapper_attr('result_X')
    result = env.get_wrapper_attr('result')
    conf_names = [str(k) for k in {**conf}.keys()]
    df = pd.DataFrame(result_X, columns=conf_names)
    goal_series = pd.Series(result, name=str(goal))
    df = pd.concat([goal_series, df], axis=1)
    collect_name = '_collect' if args.collect else ''
    if not restore:
        df.to_excel(
            os.path.join(project_dir, f'results/{dbms_name}Result/{save_folder_name}/data_hint2_{collect_name}.xlsx'),
            index=False
        )
    else:
        df.to_excel(
            os.path.join(project_dir, f'results/{dbms_name}Result/{save_folder_name}/online_data_hint2_{collect_name}.xlsx'),
            index=False
        )

    # 画图
    epochs = list(range(1, len(result) + 1))
    matplotlib.use('TkAgg')
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, result, label='Method 1', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel(goal)
    plt.title(goal + ' Comparison')
    plt.legend()
    if not restore:
        plt.savefig(os.path.join(
            project_dir, f'results/{dbms_name}Result/{save_folder_name}/output_plot2_{collect_name}.png'
        ))
    else:
        plt.savefig(os.path.join(
            project_dir, f'results/{dbms_name}Result/{save_folder_name}/oneline_output_plot2_{collect_name}.png'
        ))
    plt.show()

    if not restore:
        # 保存数据
        shutil.copy(
            os.path.join(project_dir, f'config/{dbms_name}.ini'),
            os.path.join(project_dir, f'results/{dbms_name}Result/{save_folder_name}/{dbms_name}.ini')
        )
        shutil.copy(
            os.path.join(project_dir, f'src/algorithms/RL/DDPGFD/config/{DDPGFD_config}.yaml'),
            os.path.join(project_dir, f'results/{dbms_name}Result/{save_folder_name}/{DDPGFD_config}.yaml')
        )
        os.makedirs(
            os.path.join(project_dir, f'results/{dbms_name}Result/{save_folder_name}/progress'),
            exist_ok=True
        )
        os.makedirs(
            os.path.join(project_dir, f'results/{dbms_name}Result/{save_folder_name}/result'),
            exist_ok=True
        )
        shutil.copytree(
            os.path.join(project_dir, 'src/algorithms/RL/DDPGFD/progress'),
            os.path.join(project_dir, f'results/{dbms_name}Result/{save_folder_name}/progress'),
            dirs_exist_ok=True
        )
        shutil.copytree(
            os.path.join(project_dir, 'src/algorithms/RL/DDPGFD/result'),
            os.path.join(project_dir, f'results/{dbms_name}Result/{save_folder_name}/result'),
            dirs_exist_ok=True
        )


