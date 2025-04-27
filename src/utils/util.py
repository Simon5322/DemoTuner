import logging
import os
import re
import shutil
from configparser import ConfigParser

import joblib
import matplotlib
import numpy as np
import pandas as pd
import psutil
import yaml

base_dir = os.path.dirname(os.path.abspath(__file__))

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

async def record_cpu_utilization(stop_recording_event, record_list):
    while not stop_recording_event.is_set():
        current_utilization = psutil.cpu_percent(interval=1)
        record_list.append(current_utilization)
        return record_list


def drop_unit(value_with_unit, unit):
    if unit == 'None':
        return value_with_unit
    unit_table = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
    }
    pattern = r'(\d+)(\w+)'
    result = re.search(pattern, unit)
    num = result.group(1)
    unit = result.group(2).upper()  # KB or MB or GB
    B_value = int(value_with_unit) * int(num) * int(unit_table[unit])
    return B_value


def take_unit(B_value, unit):
    if unit == 'None':
        return B_value
    unit_table = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
    }
    pattern = r'(\d+)(\w+)'
    result = re.search(pattern, unit)
    num = result.group(1)
    unit = result.group(2).upper()  # KB or MB or GB
    unit_value = int(B_value) / (int(num) * int(unit_table[unit]))
    return unit_value


def clear_folder(folder_path):
    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 获取文件夹中的所有文件和子文件夹
        items = os.listdir(folder_path)
        # 遍历文件夹中的所有项，并删除它们
        for item in items:
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                # 如果是文件，则删除
                os.remove(item_path)
            elif os.path.isdir(item_path):
                # 如果是子文件夹，则递归清空
                shutil.rmtree(item_path)
        print(f"The folder '{folder_path}' has been cleared.")
    else:
        print(f"The folder '{folder_path}' does not exist.")


# 将param中真实的数据转为action[-1, 1]
def action_to_real(db_name, action):
    #config_path = f'../tuning_params_states/{db_name}/{db_name}_params.yml'
    config_path = os.path.join(base_dir, f'../tuning_params_states/{db_name}/{db_name}_params.yml')
    with open(config_path, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    confs_keys = [key for key in {**conf}.keys()]
    confs_ranges = {param: (value['min'], value['max']) for param, value in {**conf}.items()}
    real_confs = {}
    for key, a in zip(confs_keys, action):
        min_val = confs_ranges[key][0]
        max_val = confs_ranges[key][1]
        real_val = (a + 1) * (max_val - min_val) / 2 + min_val
        real_confs[key] = int(real_val)

    return real_confs


# 将action[-1, 1]中转为真实的数据param
def real_to_action(db_name, real_values):
    """
    :param db_name: mysql, pg
    :param real_values: list type, the order is same to the conf file
    :return: n dimension of [-1, 1]
    """
    if db_name not in ['pg', 'mysql']:
        raise KeyError(f'{db_name} not valid')
    #config_path = f'../tuning_params_states/{db_name}/{db_name}_params.yml'
    config_path = os.path.join(base_dir, f'../tuning_params_states/{db_name}/{db_name}_params.yml')
    with open(config_path, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    confs_keys = [key for key in {**conf}.keys()]
    confs_ranges = {param: (value['min'], value['max']) for param, value in {**conf}.items()}
    actions = []
    for key, real_val in zip(confs_keys, real_values):
        min_val = confs_ranges[key][0]
        max_val = confs_ranges[key][1]
        a = 2 * (real_val - min_val) / (max_val - min_val)
        #print(a)
        a = a - 1
        actions.append(a)
    return np.array(actions)


def state_to_real(db_name, s):
    state_path = os.path.join(base_dir, f'../tuning_params_states/{db_name}/{db_name}_metrics.yml')
    with open(state_path, "r") as f:
        state_info = yaml.load(f, Loader=yaml.FullLoader)
    state_info_keys = [key for key in {**state_info}.keys()]
    real_state = {}
    for key, a in zip(state_info_keys, s):
        real_state[key] = int(a)
    return real_state


def trans_string_to_num(data_str, split_type):
    data = [int(num) for num in data_str.split(split_type)]
    return data


def clear_progress_result():
    progress_path = os.path.join(project_dir, 'src/algorithms/RL/DDPGFD/progress/')
    result_path = os.path.join(project_dir, 'src/algorithms/RL/DDPGFD/result/')

    clear_folder(progress_path + 's0')
    clear_folder(progress_path + 's1')
    clear_folder(result_path + 's0')
    clear_folder(result_path + 's1')

    file_paths = [
        os.path.join(result_path, 'hint_chosen.txt'),
        os.path.join(result_path, 'not_obey.txt'),
        os.path.join(result_path, 'hint_priority.xlsx')
    ]
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File '{file_path}' has been deleted.")
        else:
            print(f"File '{file_path}' does not exist, skipping deletion.")


def save_results_to_csv(latency_result, throughput_result, save_path):
    """
    Save latency and throughput results to a CSV file.

    :param latency_result: List of latency values
    :param throughput_result: List of throughput values
    :param save_path: File path to save the CSV
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 创建 DataFrame 并保存为 CSV
        df = pd.DataFrame({
            "latency": latency_result,
            "throughput": throughput_result
        })
        df.to_csv(save_path, index=False)
        print(f"Latency and throughput results saved successfully in .csv format! Path: {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"Failed to save latency and throughput results in .csv format. Error: {e}")
        raise e
def append_to_pkl(new_data, save_name):
    if os.path.exists(save_name):
        # 如果文件存在，先加载已有数据
        existing_data = joblib.load(save_name)
        # 合并新的数据
        existing_data.extend(new_data)
    else:
        # 如果文件不存在，直接使用新数据
        existing_data = new_data

    # 保存合并后的数据
    joblib.dump(existing_data, save_name)
    print(f"Demo record saved successfully! \nPath: {os.path.abspath(save_name)} \nRecords saved: {len(existing_data)}")


def quick_logging(log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(
        filename=log_file_path,  # 输出日志到文件
        level=logging.INFO,  # 设置日志级别
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
def get_current_workload_str(cpath):
    config = ConfigParser()
    config.read(cpath)
    fieldcount = int(config['WORKLOAD']['fieldcount'])
    recordcount = int(config['WORKLOAD']['recordcount'])
    operationcount = int(config['WORKLOAD']['operationcount'])
    threadcount = int(config['WORKLOAD']['threadcount'])

    fieldlength = int(config['WORKLOAD']['fieldlength'])
    minfieldlength = int(config['WORKLOAD']['minfieldlength'])

    readproportion = float(config['WORKLOAD']['readproportion'])
    updateproportion = float(config['WORKLOAD']['updateproportion'])
    insertproportion = float(config['WORKLOAD']['insertproportion'])
    scanproportion = float(config['WORKLOAD']['scanproportion'])

    workload_str = f"""
    Configuration Summary:
    ----------------------
    - Number of fields: {fieldcount}
    - Number of records: {recordcount}
    - Number of operations: {operationcount}
    - Number of threads: {threadcount}

    Field Lengths:
    - Field length: {fieldlength}
    - Minimum field length: {minfieldlength}

    Proportions:
    - Read proportion: {readproportion * 100}% 
    - Update proportion: {updateproportion * 100}% 
    - Insert proportion: {insertproportion * 100}%
    - Scan proportion: {scanproportion * 100}%
    """

    return workload_str

if __name__ == '__main__':
    clear_progress_result()
