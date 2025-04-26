import json
import os.path
import random
import shutil

import numpy as np
import joblib

from LLM import LLM_model_names

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
                                                              f'../../../results/reward_weights/global_min_max.json'))
    for d in data:
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
    print(f'success replace IRL reward for transition in {save_path}')


def compute_global_min_max(states):
    """
    计算整个数据集的全局最小值和最大值。
    参数:
        states (list): 由多个状态数组组成的列表，每个状态是一个二维数组。
    返回:
        tuple: (全局最小值, 全局最大值)
    """
    # 将所有状态展开为一个大的数组
    all_features = np.concatenate([s.flatten() for s in states])
    global_min = np.min(all_features)
    global_max = np.max(all_features)
    return global_min, global_max


# 特征提取函数
def feature_extractor(state, global_min, global_max):
    """
    提取并归一化状态特征。

    参数:
        state (np.ndarray): 二维数组表示的状态。
        global_min (float): 全局最小值，用于归一化。
        global_max (float): 全局最大值，用于归一化。

    返回:
        np.ndarray: 归一化后的特征向量。
    """
    # 展平状态数组
    features = state.flatten()

    # 使用全局的最小值和最大值归一化
    features_normalized = (features - global_min) / (global_max - global_min + 1e-8)  # 避免除以零

    # 调试信息
    # print(f"Original State:\n{state}")
    # print(f"Normalized Features:\n{features_normalized}")

    return features_normalized


# 最大熵 IRL 实现
def max_entropy_irl(states, actions, feature_dim, global_min, global_max, num_iterations=100, learning_rate=0.001):
    """
    最大熵IRL的实现

    参数:
        states (list of np.ndarray): 状态的列表
        actions (list): 动作的列表
        feature_dim (int): 特征维度
        global_min (float): 全局最小值
        global_max (float): 全局最大值
        num_iterations (int): 迭代次数
        learning_rate (float): 学习率

    返回:
        np.ndarray: 学习到的奖励权重
    """
    w = np.random.uniform(low=-0.1, high=0.1, size=feature_dim)  # 初始化权重
    for iteration in range(num_iterations):
        # 计算专家特征期望
        expert_feature_expectations = np.mean([feature_extractor(s, global_min, global_max) for s in states], axis=0)

        # 模拟特征期望（通过随机采样状态）
        model_feature_expectations = np.zeros(feature_dim)
        for _ in range(10):
            sampled_states = random.choices(states, k=len(states))
            model_feature_expectations += np.mean(
                [feature_extractor(s, global_min, global_max) for s in sampled_states], axis=0)
        model_feature_expectations /= 10

        # 计算梯度并更新权重
        grad = expert_feature_expectations - model_feature_expectations
        w += learning_rate * grad

        # 对权重进行约束，防止过大
        w = np.clip(w, -1, 1)

        # 打印调试信息
        print(f"Iteration {iteration + 1}: Gradient Norm = {np.linalg.norm(grad)}")
        print(f"Iteration {iteration + 1}: Reward Weights = {w}")

    return w


# 定义奖励函数
def reward_function(state, reward_weights, global_min, global_max):
    """
    根据学习到的权重计算奖励
    :param state: 当前状态
    :param reward_weights: 学习到的奖励权重
    :return: 奖励值
    """
    state = get_performance_states(state)
    return 83+1000 * np.dot(reward_weights, feature_extractor(state, global_min, global_max))  # RO 18 # rw:57  WO:42   Postgresql：RO 83 RW 55 WO 94


def save_reward_weights(reward_weights, output_path):
    with open(output_path, 'w') as f:
        f.write(" ".join(map(str, reward_weights)))


def load_reward_weights(file_path):
    """加载文本文件中的权重"""
    with open(file_path, 'r') as f:
        weights = list(map(float, f.read().strip().split()))
    return weights


def save_global_min_max(global_min, global_max, output_path):
    """
    保存全局最小值和最大值到文件
    """
    with open(output_path, 'w') as f:
        # 将 global_min 和 global_max 转换为原生 Python 的 float 类型
        json.dump({'global_min': float(global_min), 'global_max': float(global_max)}, f)


def load_global_min_max(input_path):
    """
    从文件加载全局最小值和最大值
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
        return data['global_min'], data['global_max']


def get_performance_states(state):
    """选择和性能相关的states execution time和latency"""
    new_state = state[0][-2:-1]
    return new_state


def save_to_target_model_path(source_file_path, target_model_dir):
    destination_dir = target_model_dir
    # 保持原来的文件名
    destination_file_path = os.path.join(destination_dir, os.path.basename(source_file_path))
    if os.path.exists(destination_file_path):
        os.remove(destination_file_path)
    # 移动文件
    shutil.copy2(source_file_path, destination_file_path)


def move_demo_transition(source_demo_path):
    destination_path = os.path.join(script_path, '../../../src/algorithms/RL/DDPGFD/data/demo/demo_1.pkl')
    shutil.copy2(source_demo_path, destination_path)
    print("success copy demo to DDPGfd demo_1.pkl")


# 主函数
def main():
    """
    主函数，执行强化学习过程，学习奖励函数权重并进行测试。
    """
    workload_type = 'RO'
    dbms_name = 'mysql'
    model_name = LLM_model_names.gpt4

    target_model_dir = os.path.join(script_path,
                                    f'../../../results/chain_demonstrations/{dbms_name}/{workload_type}/{model_name}')
    work_space_dir = os.path.join(script_path, '../../../results/reward_weights')

    # 载入链式演示数据
    chain_demonstration_path = os.path.join(target_model_dir, 'demo_1.pkl')
    demo_record = joblib.load(chain_demonstration_path)

    # 提取状态、动作和下一状态
    states = [record[0] for record in demo_record]
    actions = [record[1] for record in demo_record]
    next_states = [record[3] for record in demo_record]

    # 获取性能状态
    performance_states = [get_performance_states(state) for state in states]

    # 计算全局最小值和最大值
    global_min, global_max = compute_global_min_max(performance_states)

    # 计算特征维度
    feature_dim = len(feature_extractor(performance_states[0], global_min, global_max))

    # 学习奖励函数权重
    print("Learning reward function weights...")
    reward_weights = max_entropy_irl(
        states=performance_states,
        actions=actions,
        feature_dim=feature_dim,
        num_iterations=100,
        learning_rate=0.01,
        global_min=global_min,
        global_max=global_max
    )
    print(f"Learned reward weights: {reward_weights}")

    # 暂时保存学习到的奖励函数权重到workspace
    reward_weight_path = os.path.join(work_space_dir, 'reward_weight.txt')
    global_min_max_path = os.path.join(work_space_dir, 'global_min_max.json')
    save_reward_weights(reward_weights, reward_weight_path)
    save_global_min_max(global_min, global_max, global_min_max_path)

    # 测试奖励函数
    test_num = 5
    print(f"开始测试{test_num}次reward函数")
    reward_weights = load_reward_weights(os.path.join(work_space_dir, 'reward_weight.txt'))
    global_min, global_max = load_global_min_max(os.path.join(work_space_dir, 'global_min_max.json'))
    for i in range(test_num-1):
        # 取一个状态进行测试
        test_state = states[i]
        reward = reward_function(test_state, reward_weights, global_min, global_max)

        print(f"Test state: {get_performance_states(test_state)}")
        print(f"Reward for test state: {reward}")

    # 是否保存reward weight等信息
    confirm_save = input(
        f"是否保存reward weight和 global min max到 {target_model_dir} ? 输入 y 保存，其他任意输入则不保存\n")
    if confirm_save == 'y':
        save_to_target_model_path(reward_weight_path, target_model_dir)
        save_to_target_model_path(global_min_max_path, target_model_dir)
        # 置换reward并将demo移动到DDOGFD的demo中
        save_pkl_path = os.path.join(target_model_dir, 'chain_demonstration_IRL_reward.pkl')
        replace_r_in_pkl(chain_demonstration_path, save_pkl_path, reward_weight_path)
        move_demo_transition(save_pkl_path)
    else:
        print("不保存")


if __name__ == "__main__":
    main()
    # source_path = "/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/src/algorithms/RL/DDPGFD/data/demo/demo_1.pkl"
    # save_path = "/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/src/algorithms/RL/DDPGFD/data/demo/demo_1.pkl"
    # reward_path = "/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/results/reward_weights/reward_weight.txt"
    # replace_r_in_pkl(source_path, save_path, reward_weight_path=reward_path)
