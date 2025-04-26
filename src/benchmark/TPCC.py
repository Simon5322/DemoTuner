import glob
import json
import os
import subprocess
import threading
from argparse import ArgumentParser
from configparser import ConfigParser
import dbms.factory
from benchmark.benchmark import benchmark


class TPCC(benchmark):
    def __init__(self, benchmark_name, dbms_name):
        self.benchmark_name = benchmark_name
        #self.benchmark_path = "/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/benchbase/target/benchbase-postgres"
        self.benchmark_path = '/home/zhouyuxuan/workspace/pythonWorkspace/GPTuner-main/benchbase/target/benchbase-postgres'
        self.test = "tpcc"
        self.benchbase_bencmarks = ['tpch', 'tpcc']
        self.target_path = '/home/zhouyuxuan/workspace/pythonWorkspace/GPTuner-main/optimization_results/temp_results'
        self.dbms_name = dbms_name

    def clear_summary_dir(self, target_path):
        for filename in os.listdir(target_path):
            print(f"REMOVE {filename}")
            filepath = os.path.join(target_path, filename)
            os.remove(filepath)

    def load_data(self):
        self.clear_summary_dir(self.target_path)

        db_user = 'postgres'
        db_password = '123456'
        target_db = 'benchbasetpcc'
        template_db = 'benchbasetpcc_template'

        drop_command = f'PGPASSWORD={db_password} psql -U {db_user} -c "DROP DATABASE IF EXISTS {target_db};"'
        subprocess.run(drop_command, shell=True, check=True)
        print(f'Database {target_db} dropped.')

        # 创建新的数据库并使用模板数据库
        create_command = f'PGPASSWORD={db_password} psql -U {db_user} -c "CREATE DATABASE {target_db} WITH TEMPLATE {template_db};"'
        subprocess.run(create_command, shell=True, check=True)
        print(f'Database {target_db} created with template {template_db}.')

    def get_result(self, times):
        return self.get_latency()

    def run_benchmark(self, times=-1):
        process = None
        if self.dbms_name == "pg":
            if self.test in self.benchbase_bencmarks:
                java_path = '/usr/local/jdk21/bin/java'
                process = subprocess.Popen(
                    [java_path, '-jar', 'benchbase.jar', '-b', self.test,
                     "-c", "config/postgres/sample_{}_config.xml".format(self.test),
                     "--create=false", "--clear=false", "--load=false", '--execute=true',
                     "-d", os.path.join("../../../", self.target_path)],
                    cwd=self.benchmark_path)
        process.wait()
        throughput, latency = self.get_throughput(), self.get_latency()
        return throughput, latency


    def get_latest_summary_file(self):
        files = glob.glob(os.path.join(self.target_path, '*summary.json'))
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0] if files else None

    def get_throughput(self):
        if self.test in self.benchbase_bencmarks:
            summary_file = self.get_latest_summary_file()
            with open(summary_file, 'r') as file:
                data = json.load(file)
            throughput = data["Throughput (requests/second)"]
            if throughput == -1 or throughput == 2147483647:
                raise ValueError(f"Benchbase return error throughput:{throughput}")
            print(f"Throughput: {throughput}")

            return throughput

    def get_latency(self, latency_type = "99th Percentile Latency (microseconds)"):
        latency_types = ['Average Latency (microseconds)', '95th Percentile Latency (microseconds)', '99th Percentile Latency (microseconds)']
        if latency_type not in latency_types:
            raise KeyError(f"{latency_type} not in latency types")
        if self.test in self.benchbase_bencmarks:
            summary_file = self.get_latest_summary_file()
            with open(summary_file, 'r') as file:
                data = json.load(file)
            latency = data["Latency Distribution"][latency_type]
            if latency == -1 or latency == 2147483647:
                raise ValueError(f"Benchbase return error average_latency:{latency}")
            print(f"Latency: {latency}")
            return  latency


# script_dir = os.path.dirname(os.path.abspath(__file__))
# arg_parser = ArgumentParser(description='DDPGFD-gpt: use gpt to guide RL parameter tuning')
# arg_parser.add_argument('cpath', type=str, help='Path to configuration file')
# arg_parser.add_argument('--eval', help='Evaluation mode', action='store_true', default=False)
# arg_parser.add_argument('--collect', help='Collect Demonstration Data', action='store_true', default=False)
# arg_parser.add_argument('-n_collect', help='Number of episode for demo collection', type=int, default=100)
# args = arg_parser.parse_args()
# config = ConfigParser()
# config.read(args.cpath)
#
# dbms_name = config['DATABASE']['dbms']
# benchmark = config['BENCHMARK']['name']
# workload = config['BENCHMARK']['workload']
# use_percent = True if config['LEARNING']['use_percent'] == 'True' else False
# episode_len = int(config['LEARNING']['episode_len'])
# goal = config['LEARNING']['goal']  # latency, throughput
# save_folder_name = config['SETTING']['save_name']
# fake_test = True if config['LEARNING']['fake_test'] == 'True' else False
# repeat_times = int(config['LEARNING']['repeat_times'])
#
# dbms = dbms.factory.from_file(config)
# tpch = TPCH("tpch", "pg")
#
# tpch.load_data()
# process = tpch.run_benchmark()
# if process:
#     process.wait()
# average_latency = tpch.get_result(-1)
# print("=====================")
# print(f"latency is {average_latency}")
# print("=====================")


# summary_path = target_path
# clear_summary_dir(target_path)
# process = run_benchmark("pg", test)
# if process:
#     process.wait()
# throughput, average_latency = get_throughput(), get_latency()