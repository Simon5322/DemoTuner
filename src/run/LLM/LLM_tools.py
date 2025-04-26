import os
import re

import openai
import requests
import time
import json
import time
import os
import requests
import time
import json
import logging

import yaml

from LLM import LLM_model_names
from LLM.tools import divide_united_value, unit_table, get_united_value
from utils.util import quick_logging

script_dir = os.path.dirname(os.path.abspath(__file__))


def chat_completions(query, gpt_model):
    log_file_dir = os.path.join(script_dir, "../../../log/http_request.log")
    os.makedirs(os.path.dirname(log_file_dir), exist_ok=True)
    logging.basicConfig(
        filename=log_file_dir,  # 输出日志到文件
        level=logging.INFO,  # 设置日志级别
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    url = "https://api.zhizengzeng.com/v1/chat/completions"
    api_secret_key = 'cd4b0343e6836a2a584c75fd80b75a85'  # 你的api_secret_key
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json',
               'Authorization': "Bearer " + api_secret_key}
    params = {'user': '张三',
              'model': gpt_model,
              'messages': [{'role': 'user', 'content': query}]}
    print(f"waiting for gpt : {gpt_model} response...")
    retries = 3
    for attempt in range(retries):
        try:
            # 尝试发送请求
            r = requests.post(url, data=json.dumps(params), headers=headers)
            r.raise_for_status()  # 检查是否有HTTP错误
            logging.info(f"success request from {gpt_model}")
            return r  # 如果请求成功，返回响应
        except requests.exceptions.RequestException as e:
            # 捕获所有请求相关的异常（如网络问题、超时等）
            print(f"Attempt {attempt + 1} failed: {e}")
            logging.error(f"[ERROR] gpt request error: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {2 ** attempt} seconds...")
                time.sleep(60 * int(attempt))
            else:
                raise Exception("All retry attempts failed. Network issue or API down.")


def get_chat_result(response):
    content = None
    try_time = 0
    MAX_TRY_LIMIT = 5
    while try_time < MAX_TRY_LIMIT:
        try:
            # 检查必需的字段是否存在
            if 'choices' in response and len(response['choices']) > 0:
                # 使用get来避免KeyError
                message = response['choices'][0].get('message', {})
                content = message.get('content', None)
                if content is not None:
                    return content  # 成功获取到内容，返回
                else:
                    raise ValueError("Missing 'content' in message.")
            else:
                try_time += 1
                raise ValueError("Missing or empty 'choices' in response.")
        except (KeyError, ValueError) as e:
            quick_logging(os.path.join(script_dir, "../../../log/http_request.log"))
            logging.error(f"Error while extracting chat result: {str(e)}")
            time.sleep(1)  # 延时 1 秒后重试
            if try_time >= MAX_TRY_LIMIT:
                raise  # 重试次数超过限制，抛出异常
    return content


def get_tokens(response):
    prompt_token = response['usage']['prompt_tokens']
    completion_tokens = response['usage']['completion_tokens']
    total_tokens = response['usage']['total_tokens']
    return prompt_token, completion_tokens, total_tokens


def align_gpt_conf(gpt_conf, conf):
    """
    对齐 GPT 返回的配置与默认配置，并转换为 numpy 数组。

    :param gpt_conf: GPT 提供的配置（JSON 格式，dict 类型）
    :param conf: 默认配置，从 YAML 文件加载的配置（dict 类型）
    :return: numpy 数组，顺序与 conf 的键顺序一致
    """
    aligned_conf = []

    # 确保顺序与 conf 的键顺序一致
    for key, value in conf.items():
        if key in gpt_conf:
            aligned_conf.append(int(gpt_conf[key]))
        else:
            aligned_conf.append(int(value['default']))  # 使用默认值补充
    # 转换为 numpy 数组
    return aligned_conf


def extract_json_from_text(text):
    """
    从文本中提取 JSON 格式的内容
    """
    # 匹配 JSON 格式的数据（包括嵌套对象）
    json_pattern = r"\{.*?\}"
    matches = re.findall(json_pattern, text, re.DOTALL)

    if matches:
        # 返回第一个匹配到的 JSON 数据
        return matches[0]
    return None

def get_conf_range_description(conf):
    confs_range = {}
    for param, details in conf.items():
        full_unit = details.get('unit', 'None')
        if full_unit != 'None':
            pattern = r'(\d+)(\w+)'
            result = re.search(pattern, full_unit)
            num = result.group(1)
            unit = result.group(2)
            confs_range[param] = {
                'min':str(int(details['min']) * int(num)) + unit,
                'max':str(int(details['max']) * int(num)) + unit,
            }
        else:
            confs_range[param] = {'min':details['min'], 'max':details['max']}

    return confs_range

def gpt_recommend_conf(state, workload, last_conf, last_performance, dbms_name, gpt_model=LLM_model_names.WenXinYiYan_Baidu):
    """
    :param state:
    :param workload:
    :param last_conf:
    :param last_performance:
    :param gpt_model:
    :return: json format
    """
    logging.basicConfig(
        filename=os.path.join(script_dir, "../../../log/temp_gpt_recommendation.log"),  # 输出日志到文件
        level=logging.INFO,  # 设置日志级别
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    dbms = "MySQL" if dbms_name == 'mysql' else "PostgreSQL"
    config_path = os.path.join(script_dir, f"../../tuning_params_states/{dbms_name}/{dbms_name}_params.yml")
    with open(config_path, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    confs_range = get_conf_range_description(conf)

    optimization_goal = "execution time"
    prompt = f"""
        You are a system optimization assistant designed to recommend {dbms} database configurations that improve system performance based on the following information:
        1. **Current Configuration**: {last_conf}
        2. **Workload Characteristics**: {workload}
        3. **Current System State**: {state}
        4. **Performance of Current Configuration**: Execution time: {last_performance}
        5. **Hardware Information**: 24GB RAM, 200GB SSD, 8-core CPU

        Please provide the optimized configuration in **JSON format only，the format like Current Configuration I give you**. 
        Do not include any comments, explanations, or text outside of the JSON. 
        The recommended configuration should only include the parameters provided in the **Current Configuration** , and no new configuration parameters should be added,
        All recommended parameter values must be numerical.
        The recommended values must fall within the range provided: {confs_range}
        """
    #
    print(prompt)
    retry_time = 0
    MAX_TRY_TIMES = 10
    source_conf_gpt_recommend = ''
    while retry_time < MAX_TRY_TIMES:
        source_conf_gpt_recommend = get_chat_result(chat_completions(prompt, gpt_model).json())
        try:
            # 尝试解析 JSON 字符串
            conf_gpt_recommend = json.loads(source_conf_gpt_recommend)
            # 检查是否是字典
            if isinstance(conf_gpt_recommend, dict):
                # 检查字典中的值是否为数字类型
                all_values_valid = True
                for key, value in conf_gpt_recommend.items():
                    print(f"{key}: {value} ({type(value)})")
                    if isinstance(value, (int, float)) or re.fullmatch(r'\d+', str(value)):  # 纯数字
                        continue  # 合法，跳过
                    match = re.fullmatch(r'(\d+(?:\.\d+)?)(\w+)', value) # 解析数字+单位
                    if match:
                        num, unit = match.groups()
                        if unit.upper() in unit_table.keys():  # 确保单位是 unit_table.keys() 之一
                            continue
                    all_values_valid = False
                    break

                if all_values_valid and len(conf_gpt_recommend) == len(confs_range):
                    print("Valid JSON with numerical values:", conf_gpt_recommend)
                    for param, value in conf_gpt_recommend.items():
                        if isinstance(value, (int, float)) or re.fullmatch(r'\d+', str(value)):
                            conf_gpt_recommend[param] = value
                        else:
                            match = re.fullmatch(r'(\d+)(\w+)', value)  # 解析数字+单位
                            recommend_num, recommend_unit = match.groups()
                            recommend_num = int(recommend_num)
                            conf_num, conf_unit = get_united_value(conf, param)
                            conf_gpt_recommend[param]= (recommend_num/conf_num) * (unit_table[recommend_unit]/unit_table[conf_unit])
                    return conf_gpt_recommend
                else:
                    # 如果值不符合要求，跳过本次尝试，增加重试次数
                    retry_time += 1
                    print(f"Retry {retry_time}: Non-numeric value found, retrying...")
                    time.sleep(2)
                    continue
            else:
                raise json.JSONDecodeError(
                    "JSON is not an object (key-value pairs expected)",
                    source_conf_gpt_recommend,
                    0
                )
        except json.JSONDecodeError as e:
            # 如果 JSON 解析失败，尝试提取 JSON
            extracted_json_conf = extract_json_from_text(source_conf_gpt_recommend)  # 正则表达式提取
            if extracted_json_conf:
                try:
                    # 重新加载提取的 JSON
                    conf_gpt_recommend = json.loads(extracted_json_conf)
                    # 检查字典中的值是否为数字类型
                    all_values_valid = True
                    for key, value in conf_gpt_recommend.items():
                        if isinstance(value, (int, float)) or re.fullmatch(r'\d+', str(value)):   # 纯数字，合法
                            continue
                        match = re.fullmatch(r'(\d+(?:\.\d+)?)(\w+)', value) # 解析数字+单位
                        if match:
                            num, unit = match.groups()
                            if unit.upper() in unit_table.keys():  # 确保单位是 unit_table.keys() 之一
                                continue
                        all_values_valid = False
                        logging.warning(f"Value for key '{key}' is not a number. Retrying...")
                        break
                    if all_values_valid and len(conf_gpt_recommend) == len(confs_range):
                        print("Valid JSON with numerical values:", conf_gpt_recommend)
                        for param, value in conf_gpt_recommend.items():
                            if isinstance(value, (int, float)) or re.fullmatch(r'\d+', str(value)):
                                conf_gpt_recommend[param] = value
                            else:
                                match = re.fullmatch(r'(\d+)(\w+)', value)  # 解析数字+单位
                                recommend_num, recommend_unit = match.groups()
                                recommend_num = int(recommend_num)
                                conf_num, conf_unit = get_united_value(conf, param)
                                conf_gpt_recommend[param] = (recommend_num / conf_num) * (unit_table[recommend_unit]/unit_table[conf_unit])
                                print("Valid JSON after extraction with numerical values:", conf_gpt_recommend)
                        return conf_gpt_recommend
                    else:
                        # 如果值不符合要求，跳过本次尝试，增加重试次数
                        retry_time += 1
                        print(f"Retry {retry_time}: Non-numeric value found in extracted JSON, retrying...")
                        time.sleep(2)
                        continue
                except json.JSONDecodeError as e:
                    logging.error(
                        f"json load failed from {e}. The response result from {gpt_model} is {source_conf_gpt_recommend}")
            else:
                logging.error(
                    f"RE extracted failed from {e}. The response result from {gpt_model} is {source_conf_gpt_recommend}")

            # 增加重试次数，休眠 2 秒
            retry_time += 1
            print(f"Retry {retry_time}: Invalid JSON, retrying...")
            time.sleep(2)
    error_message = f"Failed to get valid JSON after {MAX_TRY_TIMES} attempts. The invalid response result from {gpt_model} is {source_conf_gpt_recommend}"
    logging.error(error_message)
    raise RuntimeError(error_message)


if __name__ == '__main__':
    query1 = """
     You are a system optimization assistant designed to recommend database configurations that improve system performance based on the following information:
            1. **Current Configuration**: {'work_mem': 4096, 'wal_buffers': 512, 'temp_buffers': 1024, 'shared_buffers': 16384, 'effective_cache_size': 524288, 'maintenance_work_mem': 65536, 'max_connections': 100, 'bgwriter_lru_multiplier': 2, 'backend_flush_after': 0, 'bgwriter_delay': 200, 'max_parallel_workers': 8, 'hash_mem_multiplier': 2, 'checkpoint_flush_after': 32, 'max_wal_size': 1024, 'join_collapse_limit': 8, 'vacuum_cost_page_dirty': 20, 'min_parallel_table_scan_size': 1024, 'min_parallel_index_scan_size': 64, 'max_parallel_workers_per_gather': 2}
            2. **Workload Characteristics**:
        Configuration Summary:
        ----------------------
        - Number of fields: 10
        - Number of records: 1000000
        - Number of operations: 500000
        - Number of threads: 10

        Field Lengths:
        - Field length: 100
        - Minimum field length: 1

        Proportions:
        - Read proportion: 10.0%
        - Update proportion: 40.0%
        - Insert proportion: 40.0%
        - Scan proportion: 10.0%

            3. **Current System State**: {'blks_read': 0, 'blks_hit': 217, 'temp_bytes': 0, 'xact_commit': 2, 'buffers_backend': 0, 'buffers_checkpoint': 0, 'checkpoints_req': 0, 'wal_records': 0, 'wal_write_time': 0, 'idx_scan': 0, 'vacuum_count': 0, 'cpu_changed': 13, 'cpu_usage': 13, 'mem_changed': 62, 'mem_usage': 62, 'write_speed_changed': 58163, 'io_latency_changed': 0, 'load1': 0, 'node_memory_Dirty_bytes': 598016}
            4. **Performance of Current Configuration**: Execution time: 52428.0
            5. **Hardware Information**: 24GB RAM, 200GB SSD, 8-core CPU

            Please provide the optimized configuration in **JSON format**.
            The recommended configuration should only include the parameters provided in the **Current Configuration** , and no new configuration parameters should be added,All recommended parameter values must be numerical..
    """

    prompt_mysql = """
    You are a system optimization assistant designed to recommend MySQL database configurations that improve system performance based on the following information:
        1. **Current Configuration**: {'innodb_buffer_pool_size': 1, 'innodb_io_capacity': 200, 'innodb_log_buffer_size': 16, 'innodb_log_file_size': 48, 'innodb_log_files_in_group': 2, 'innodb_lru_scan_depth': 1024, 'innodb_purge_threads': 4, 'innodb_read_io_threads': 4, 'innodb_write_io_threads': 4, 'innodb_thread_concurrency': 0, 'join_buffer_size': 2, 'max_heap_table_size': 1024, 'read_buffer_size': 16, 'sort_buffer_size': 8, 'preload_buffer_size': 32, 'table_open_cache': 4000, 'thread_cache_size': 9, 'tmp_table_size': 16384, 'net_buffer_length': 16}
        2. **Workload Characteristics**: 
    Configuration Summary:
    ----------------------
    - Number of fields: 10
    - Number of records: 50
    - Number of operations: 50
    - Number of threads: 10
    Field Lengths:
    - Field length: 100
    - Minimum field length: 1
    Proportions:
    - Read proportion: 10.0% 
    - Update proportion: 40.0% 
    - Insert proportion: 40.0%
    - Scan proportion: 10.0%
    
        3. **Current System State**: {'cpu_changed': 2, 'cpu_usage': 33, 'mem_changed': 1, 'mem_usage': 84, 'write_speed_changed': -764040, 'io_latency_changed': 0, 'load1': 1, 'node_memory_Dirty_bytes': 6414336, 'Innodb_buffer_pool_reads': 4, 'Innodb_buffer_pool_read_requests': 2790, 'Com_select': 32, 'Com_insert': 66, 'Com_update': 23, 'Com_delete': 0, 'Innodb_data_reads': 4, 'Innodb_data_writes': 597, 'Innodb_buffer_pool_pages_dirty': 0, 'Innodb_buffer_pool_pages_total': 8192, 'Innodb_log_waits': 0, 'Handler_read_rnd_next': 7957, 'Created_tmp_tables': 16, 'Created_tmp_disk_tables': 0, 'Sort_merge_passes': 0, 'Threads_connected': 2, 'data_set_num': 2}
        4. **Performance of Current Configuration**: Execution time: 369.0
        5. **Hardware Information**: 24GB RAM, 200GB SSD, 8-core CPU
        Please provide the optimized configuration in **JSON format only**. 
        Do not include any comments, explanations, or text outside of the JSON. 
        The recommended configuration should only include the parameters provided in the **Current Configuration** , and no new configuration parameters should be added,
        All recommended parameter values must be numerical.
       
    """
    #  The recommended values must fall within the range provided: {'innodb_buffer_pool_size': {'min': 1, 'max': 144}, 'innodb_io_capacity': {'min': 100, 'max': 5000}, 'innodb_log_buffer_size': {'min': 1, 'max': 1024}, 'innodb_log_file_size': {'min': 10, 'max': 1024}, 'innodb_log_files_in_group': {'min': 2, 'max': 10}, 'innodb_lru_scan_depth': {'min': 100, 'max': 4096}, 'innodb_purge_threads': {'min': 1, 'max': 32}, 'innodb_read_io_threads': {'min': 1, 'max': 64}, 'innodb_write_io_threads': {'min': 1, 'max': 64}, 'innodb_thread_concurrency': {'min': 0, 'max': 100}, 'join_buffer_size': {'min': 1, 'max': 2000}, 'max_heap_table_size': {'min': 1, 'max': 131072}, 'read_buffer_size': {'min': 1, 'max': 512}, 'sort_buffer_size': {'min': 1, 'max': 320}, 'preload_buffer_size': {'min': 1, 'max': 3200}, 'table_open_cache': {'min': 1, 'max': 4919}, 'thread_cache_size': {'min': 0, 'max': 16384}, 'tmp_table_size': {'min': 1, 'max': 262144}, 'net_buffer_length': {'min': 1, 'max': 1024}
    # query2 = "圆周率前10位"
    # query3 = "我问你的上一个问题是什么"
    result = chat_completions(prompt_mysql, LLM_model_names.XunFeiXingHuo).json()
    result = get_chat_result(result)
    # result = extract_json_from_text(result)
    # result = json.loads(result)
    print(result)
