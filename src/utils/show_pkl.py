import os
import pickle

import sys

import joblib
import numpy as np

import LLM.LLM_model_names
from IRL.Maximun_Entropy_IRL import load_reward_weights, reward_function, load_global_min_max, get_performance_states
from hintsClasses.Hint import get_all_hints


script_path = os.path.dirname(os.path.abspath(__file__))
def replace_r_in_pkl(source_path, save_path, reward_weight_path):
    """
    将pkl中原始的reward改为IRL的reward
    :param source_path:
    :param save_path:
    :return:
    """
    data = joblib.load(source_path)
    demo_record = []
    reward_weights = load_reward_weights(reward_weight_path)
    global_min, global_max = load_global_min_max(os.path.join(script_path,
                                                              f'../../results/reward_weights/global_min_max.json'))

    for d in data:
        # print(d)
        s, a, r, s2, done, hint_id = d
        reward = reward_function(s2, reward_weights, global_min, global_max)
        demo_record.append(
            (s, np.array(a), reward, s2, done, hint_id)
        )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        joblib.dump(demo_record, save_path)
    except Exception as e:
        print(f'dump demonstration with error {e}')
        raise e
    print(f'success dump demonstration of to path {save_path}')


def show_pkl_content(pkl_file_path):
    """

    :param pkl_file_path: 要展示的pkl文件的位置
    :return:
    """
    hints = get_all_hints('mysql')
    data = joblib.load(pkl_file_path)
    print('数量: ' + str(len(data)))
    hint_ids = []
    key = True
    for d in data:
        # print(d)
        s, a, r, s2, done, hint_id = d
        latency = get_performance_states(s2)
        print(latency)
        print(r)
        # for hint in hints:
        #     if hint.hint_id == hint_id:
        #         des = hint.to_string()
        #         print(des + "reward : " + str(r))
        # hint_ids.append(hint_id)
        # print(hint_id)
    #print(data)


script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../utils'))
pkl_file_path = os.path.join(script_dir, '../../results/chain_demonstrations/pg/WO/gpt-4o/chain_demonstration_IRL_reward.pkl')
demo_path = os.path.join(script_dir, '../../src/algorithms/RL/DDPGFD/data/demo/demo_1.pkl')
#os.path.join(script_dir, '../algorithms/RL/DDPGFD/data/demo/demo_1.pkl')

dbms_name = "mysql"
workload_type = "RO"
model = LLM.LLM_model_names.gpt4

reward_weight_path = os.path.join(script_dir,
                                  f'../../results/chain_demonstrations/{dbms_name}/{workload_type}/{model}/reward_weight.txt')
source_pkl_path = os.path.join(script_dir, f'../../results/chain_demonstrations/{dbms_name}/{workload_type}/{model}/demo_1.pkl')
save_pkl_path = os.path.join(script_dir, f'../../results/chain_demonstrations/{dbms_name}/{workload_type}/{model}/chain_demonstration_IRL_reward.pkl')
# 调用替换函数
#replace_r_in_pkl(source_pkl_path, save_pkl_path, reward_weight_path)
show_pkl_content(demo_path)
