from abc import ABC

import yaml
from matplotlib import pyplot as plt

import Global_Variables

import numpy as np

from benchmark.benchmark import benchmark


class SimulateBenchmark(benchmark):
    def __init__(self, benchmark_name, confs):
        self.benchmark_name = benchmark_name
        conf_real_value = {param: value['default'] for param, value in {**confs}.items()}
        confs_range = {param: (value['min'], value['max']) for param, value in {**confs}.items()}
        default_action = []
        for param, real_value in conf_real_value.items():
            max_val = confs_range[param][1]
            min_val = confs_range[param][0]
            default_a = 2 * (real_value - min_val) / (max_val - min_val) - 1
            default_action.append(default_a)
        Global_Variables.globalCurrentAction = default_action

    def load_data(self):
        pass

    def run_benchmark(self, times=-1):
        # 基本延迟
        base_latency = 100
        currentAction = Global_Variables.globalCurrentAction
        if currentAction is None or len(currentAction) == 0:
            raise KeyError('currentAction is not valid')
        # if currentAction is None:
        #     return 10, 200

        latency = base_latency

        final_latency = self.simple_simulate(latency, currentAction)
        return -1, final_latency

    def complex_simulate(self, latency, currentAction):
        for i, value in enumerate(currentAction):
            if i % 2 == 0:
                latency += 10 * np.sin(value * np.pi)  # 正弦变换（加法）
                latency -= 5 * np.log(value + 1.1)  # 对数变换（减法），避免log(0)
            else:
                latency -= 7 * np.cos(value * np.pi)  # 余弦变换（减法）
                latency += 3 * np.sqrt(abs(value))  # 平方根变换（加法）

            latency += 0.1 * value ** 2
        noise = np.random.normal(0, 5)
        final_latency = latency + noise
        return final_latency

    def simple_simulate(self, latency, currentAction):
        important_indices = [0, 1, 2, 3, 4, 5]
        for i in important_indices:
            value = currentAction[i]
            if i % 2 == 0:
                latency += 3 * np.sin(value * np.pi)  # 非线性变换（加法）
            else:
                latency -= 2 * np.log(np.abs(value) + 1)  # 非线性变换（减法）

            # 处理其他维度
        for i in range(6, len(currentAction)):
            value = currentAction[i]
            latency += 0.2 * value

            # 添加随机噪声
        noise = np.random.normal(0, 3)
        final_latency = latency + noise
        return final_latency
    def get_result(self, times):
        pass

    def find_best_latency(self, dimensions=19, num_samples=1000, goal='min'):
        min_latency = float('inf')
        max_latency = -float('inf')
        best_action = None

        for _ in range(num_samples):
            action = np.random.uniform(-1, 1, dimensions)
            Global_Variables.globalCurrentAction = action
            _, latency = self.run_benchmark(times=-1)
            if goal == 'min':
                if latency < min_latency:
                    min_latency = latency
                    best_action = action
            else:
                if latency > max_latency:
                    max_latency = latency
                    best_action = action

        best_latency = min_latency if goal == 'min' else max_latency
        return best_action, best_latency


