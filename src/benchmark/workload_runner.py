import configparser
import shutil
import subprocess
import os
import glob
import json
import threading

from benchmark.YCSB import YCSB


class BenchbaseRunner:
    def __init__(self, dbms, test, target_path="/home/zhouyuxuan/workspace/pythonWorkspace/GPTuner-main/optimization_results/temp_results"):
        """target_path is the relative path under the GPTuner folder"""
        self.process = None
        self.test = test
        self.dbms = dbms
        self.target_path = target_path
        self.benchbase_bencmarks = ['tpch', 'tpcc']
        if isinstance(self.dbms, PgDBMS):
            self.benchmark_path = "./benchbase/target/benchbase-postgres"
        else:
            self.benchmark_path = "./benchbase/target/benchbase-mysql"

        self.workload_setting_path = r'/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/config/pg.ini'
        config = configparser.ConfigParser()
        config.read(self.workload_setting_path)
        if self.test == 'ycsb':
            self.fieldcount = config['WORKLOAD']['fieldcount']
        self.ycsb_latency = None
        self.ycsb_throughput = None
        self.ycsb = None
        self.stop_event = threading.Event()

    def run_benchmark(self):
        if isinstance(self.dbms, PgDBMS):
            if self.test in self.benchbase_bencmarks:
                java_path = '/usr/local/jdk21/bin/java'
                self.process = subprocess.Popen(
                    [java_path, '-jar', 'benchbase.jar', '-b', self.test,
                     "-c", "config/postgres/sample_{}_config.xml".format(self.test),
                     "--create=false", "--clear=false", "--load=false", '--execute=true',
                     "-d", os.path.join("../../../", self.target_path)],
                    cwd=self.benchmark_path
                )
            elif self.test == 'ycsb':
                self.ycsb = YCSB(self.dbms)
                self.dbms.create_new_usertable(fieldcount=self.fieldcount)
                if not self.stop_event.is_set():
                    self.ycsb.load_data()
                if not self.stop_event.is_set():
                    self.throughput, self.latency = self.ycsb.run_benchmark(times=-1)

        #     self.process = subprocess.Popen(
        #         ['java', '-jar', 'benchbase.jar', '-b', self.test,
        #          "-c", "config/postgres/sample_{}_config.xml".format(self.test),
        #          "--create=false", "--clear=false", "--load=false", '--execute=true',
        #          "-d", os.path.join("../../../", self.target_path)],
        #         cwd=self.benchmark_path
        #     )
        elif isinstance(self.dbms, MysqlDBMS):
            if self.test in self.benchbase_bencmarks:
                self.process = subprocess.Popen(
                    ['java', '-jar', 'benchbase.jar', '-b', self.test,
                     "-c", "config/mysql/sample_{}_config.xml".format(self.test),
                     "--create=false", "--clear=false", "--load=false", '--execute=true',
                     "-d", os.path.join("../../../", self.target_path)],
                    cwd=self.benchmark_path
                )
            elif self.test == 'ycsb':
                ycsb = YCSB(self.dbms)
                self.dbms.create_new_usertable(fieldcount=self.fieldcount)
                ycsb.load_data()
                self.throughput, self.latency = ycsb.run_benchmark(times=-1)

        if self.test in self.benchbase_bencmarks:
            self.process.wait()


    def clear_summary_dir(self):
        for filename in os.listdir(self.target_path):
            print(f"REMOVE {filename}")
            filepath = os.path.join(self.target_path, filename)
            os.remove(filepath)
            # if os.path.isfile(filepath):
            #     os.remove(filepath)
            # elif os.path.isdir(filepath):
            #     shutil.rmtree(filepath)

    def get_latest_summary_file(self):
        files = glob.glob(os.path.join(self.target_path, '*summary.json'))
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0] if files else None

    def get_latest_raw_file(self):
        files = glob.glob(os.path.join(self.target_path, '*raw.csv'))
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0] if files else None

    def get_throughput(self):
        if self.test in self.benchbase_bencmarks:
            summary_file = self.get_latest_summary_file()
            try:
                with open(summary_file, 'r') as file:
                    data = json.load(file)
                throughput = data["Throughput (requests/second)"]
                if throughput == -1 or throughput == 2147483647:
                    raise ValueError(f"Benchbase return error throughput:{throughput}")
                print(f"Throughput: {throughput}")
            except Exception as e:
                print(f'Exception for JSON: {e}')
                throughput = self.penalty - 2
            return throughput
        elif self.test == 'ycsb':
            return self.throughput

    def get_latency(self) -> object:
        if self.test in self.benchbase_bencmarks:
            summary_file = self.get_latest_summary_file()
            try:
                with open(summary_file, 'r') as file:
                    data = json.load(file)
                average_latency = data["Latency Distribution"]["Average Latency (microseconds)"]
                if average_latency == -1 or average_latency == 2147483647:
                    raise ValueError(f"Benchbase return error average_latency:{average_latency}")
                print(f"Latency: {average_latency}")
            except Exception as e:
                print(f'Exception for JSON: {e}')
                average_latency = self.penalty - 2
            return average_latency
        elif self.test == 'ycsb':
            return self.latency
