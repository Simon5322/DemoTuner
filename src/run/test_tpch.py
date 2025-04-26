import glob
import json
import os
import subprocess
import threading
from argparse import ArgumentParser
from configparser import ConfigParser
import dbms.factory
from benchmark.TPCH import TPCH

benchmark_path = "/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/benchbase/target/benchbase-postgres"
test = "tpch"
benchbase_bencmarks = ['tpch', 'tpcc']
target_path = '/home/zhouyuxuan/workspace/pythonWorkspace/GPTuner-main/optimization_results/temp_results'


def clear_summary_dir(target_path):
    for filename in os.listdir(target_path):
        print(f"REMOVE {filename}")
        filepath = os.path.join(target_path, filename)
        os.remove(filepath)

def run_benchmark(dbms_name, test):
    process = None
    if dbms_name == "pg":
        if test in benchbase_bencmarks:
            java_path = '/usr/local/jdk21/bin/java'
            process = subprocess.Popen(
                [java_path, '-jar', 'benchbase.jar', '-b', test,
                 "-c", "config/postgres/sample_{}_config.xml".format(test),
                 "--create=false", "--clear=false", "--load=false", '--execute=true',
                 "-d", os.path.join("../../../", target_path)],
                cwd=benchmark_path)
    return process

def get_latest_summary_file():
    files = glob.glob(os.path.join(target_path, '*summary.json'))
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0] if files else None
def get_throughput():
    if test in benchbase_bencmarks:
        summary_file = get_latest_summary_file()
        with open(summary_file, 'r') as file:
            data = json.load(file)
        throughput = data["Throughput (requests/second)"]
        if throughput == -1 or throughput == 2147483647:
            raise ValueError(f"Benchbase return error throughput:{throughput}")
        print(f"Throughput: {throughput}")

        return throughput

def get_latency():
    if test in benchbase_bencmarks:
        summary_file = get_latest_summary_file()
        with open(summary_file, 'r') as file:
            data = json.load(file)
        average_latency = data["Latency Distribution"]["Average Latency (microseconds)"]
        if average_latency == -1 or average_latency == 2147483647:
            raise ValueError(f"Benchbase return error average_latency:{average_latency}")
        print(f"Latency: {average_latency}")
        return average_latency


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
use_percent = True if config['LEARNING']['use_percent'] == 'True' else False
episode_len = int(config['LEARNING']['episode_len'])
goal = config['LEARNING']['goal']  # latency, throughput
save_folder_name = config['SETTING']['save_name']
fake_test = True if config['LEARNING']['fake_test'] == 'True' else False
repeat_times = int(config['LEARNING']['repeat_times'])

dbms = dbms.factory.from_file(config)
tpch = TPCH("tpch", "pg")

tpch.load_data()
t, l = tpch.run_benchmark()
print("=====================")
print(f"latency is {l}")
print("=====================")
# summary_path = target_path
# clear_summary_dir(target_path)
# process = run_benchmark("pg", test)
# if process:
#     process.wait()
# throughput, average_latency = get_throughput(), get_latency()