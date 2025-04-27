import configparser
import os
import subprocess
from abc import ABC, abstractmethod

from benchmark.benchmark import benchmark

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
class YCSB(benchmark):
    def __init__(self, dbms_name, save_folder_name, workload="workloada", YCSB_path=None):
        super().__init__(self.__class__.__name__)
        self.workload = workload
        self.dbms_name = dbms_name
        self.result_path = os.path.join(project_dir, f'results/{dbms_name}Result',
            save_folder_name)

        self.YCSB_path = YCSB_path
        self.workload_setting_path = os.path.join(project_dir, f'config/{dbms_name}.ini')
        self.myworkload_path = os.path.join(YCSB_path, 'workloads/myworkload')
        self.jdbc_jar = 'mysql-connector-java-8.3.0.jar' if self.dbms_name == 'mysql' else 'postgresql-42.6.0.jar'
        config = configparser.ConfigParser()
        config.read(self.workload_setting_path)


        typ_list1 = ['ordered', 'hashed']
        typ_list2 = ['uniform', 'zipfian', 'hotspot', 'sequential', 'exponential', 'latest']
        config_info = {
            'workload': 'com.yahoo.ycsb.workloads.CoreWorkload',
            'recordcount': int(config['WORKLOAD']['recordcount']),
            'operationcount': int(config['WORKLOAD']['operationcount']),
            'threadcount': int(config['WORKLOAD']['threadcount']),
            'fieldcount': int(config['WORKLOAD']['fieldcount']),
            'fieldlength': int(config['WORKLOAD']['fieldlength']),
            'minfieldlength': int(config['WORKLOAD']['minfieldlength']),
            'readproportion': float(config['WORKLOAD']['readproportion']),
            'updateproportion': float(config['WORKLOAD']['updateproportion']),
            'insertproportion': float(config['WORKLOAD']['insertproportion']),
            'scanproportion': float(config['WORKLOAD']['scanproportion']),
            'readmodifywriteproportion': float(config['WORKLOAD']['readmodifywriteproportion']),
            'minscanlength': int(config['WORKLOAD']['minscanlength']),
            'maxscanlength': int(config['WORKLOAD']['maxscanlength']),
            'insertstart': int(config['WORKLOAD']['insertstart']),
            'insertcount': int(config['WORKLOAD']['insertcount']),
            'zeropadding': int(config['WORKLOAD']['zeropadding']),
            'readallfields': True if int(config['WORKLOAD']['readallfields']) == 0 else False,
            'writeallfields': True if int(config['WORKLOAD']['writeallfields']) == 0 else False,
            'insertorder': typ_list1[int(config['WORKLOAD']['insertorder'])],
            'scanlengthdistribution': typ_list2[int(config['WORKLOAD']['scanlengthdistribution'])],
            'requestdistribution': typ_list2[int(config['WORKLOAD']['requestdistribution'])],

        }
        # 打开配置文件并写入配置信息
        with open(self.myworkload_path, 'w') as f:
            for key, value in config_info.items():
                f.write('{}={}\n'.format(key, str(value)))



    def load_data(self):
        print('===================load data===================')
        os.chdir(self.YCSB_path)
        # load_data_cmd = 'bin/ycsb load jdbc -P workloads/myworkload -P ./jdbc-binding/conf/db.properties -cp pg-42.6.0.jar'
        # load_data_cmd = ['python2', 'bin/ycsb', 'load', 'jdbc', '-P', 'workloads/myworkload', '-P',
        #                  f'./jdbc-binding/conf/{self.dbms_name}Db.properties', '-cp ' + f'{self.jdbc_jar}']
        load_data_cmd = [
            'python2',
            'bin/ycsb',
            'load',
            'jdbc',
            '-P', 'workloads/myworkload',
            '-P', f'./jdbc-binding/conf/{self.dbms_name}Db.properties',
            '-cp', f'{self.jdbc_jar}'
        ]

        # python2 bin/ycsb load jdbc -P workloads/myworkload -P ./jdbc-binding/conf/db.properties -cp pg-42.6.0.jar
        try:
            subprocess.run(load_data_cmd, shell=False)

        except subprocess.CalledProcessError as e:
            print(f"命令执行失败，返回码为 {e.returncode}。")
            print(e.output)
        except Exception as e:
            print(f"发生错误：{e}")

    # times: 第几轮的测试

    # times: 大于等于0 第几轮的结果
    #        -1 ： 直接获得结果
    def run_benchmark(self, times=0):
        print('===================run benchmark===================')
        throughput = None
        latency = None
        run_data_cmd = None
        run_data_cmd = ''

        if times >= 0:
            if not os.path.exists(self.result_path):
                os.makedirs(self.result_path)
            result_file_path = self.result_path + "/pg_ycsb" + str(times) + ".txt"
            if not os.path.exists(result_file_path):
                with open(result_file_path, "w") as result_file:
                    pass  # 创建一个空的txt文件
            run_data_cmd = ['python2', 'bin/ycsb', 'run', 'jdbc', '-P', 'workloads/myworkload', '-P',
                            f'./jdbc-binding/conf/{self.dbms_name}Db.properties', '-cp ' + f'{self.jdbc_jar}']
            # print(run_data_cmd)
        elif times == -1:
            run_data_cmd = [
                'python2',
                'bin/ycsb',
                'run',
                'jdbc',
                '-P', 'workloads/myworkload',
                '-P', f'./jdbc-binding/conf/{self.dbms_name}Db.properties',
                '-cp', f'{self.jdbc_jar}'
            ]

            # 运行 Python 部分 python2 bin/ycsb run jdbc -P workloads/myworkload -P ./jdbc-binding/conf/db.properties -cp pg-42.6.0.jar

        os.chdir(self.YCSB_path)
        try:
            # subprocess.run(load_data_cmd,shell=True)
            result = subprocess.run(run_data_cmd, shell=False, stdout=subprocess.PIPE, text=True)
            result = result.stdout
            print(result)
            lines = str(result).split("\n")
            for line in lines:
                if line.startswith("[OVERALL], Throughput(ops/sec)"):
                    throughput = float(line.split(", ")[2])
                # elif line.startswith("[READ], AverageLatency(us)"):
                elif line.startswith("[OVERALL], RunTime(ms)"):
                    latency = float(line.split(", ")[2])
            return throughput, latency
        except subprocess.CalledProcessError as e:
            print(f"命令执行失败，返回码为 {e.returncode}。")
            print(e.output)
        except Exception as e:
            print(f"发生错误：{e}")

    def get_result(self, times):
        result_path = self.result_path + "/pg_ycsb" + str(times) + ".txt"
        throughput = None
        latency = None
        with open(result_path, "r") as file:
            lines = file.read().split("\n")
            for line in lines:
                if line.startswith("[OVERALL], Throughput(ops/sec)"):
                    throughput = float(line.split(", ")[2])
                elif line.startswith("[READ], AverageLatency(us)"):
                    latency = float(line.split(", ")[2])
        return throughput, latency



