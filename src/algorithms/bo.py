import subprocess

import numpy as np
import yaml
from skopt import gp_minimize
import GPT_get

np.random.seed(237)
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
from skopt.plots import plot_gaussian_process
import matplotlib.pyplot as plt
import dbms

noise_level = 0.1
env = {}
use_env = True


class BOenvOptimizer:
    def __init__(self, config, dbms, benchmark,tuning_metric,use_env=False, conf={}):
        self.config = {**config}
        default = {}
        bo_space = {}
        for k, v in self.config.items():
            default[k] = v.get('default')
            v_range = v.get('range')
            if v_range:  # discrete ranged parameter
                bo_space[k] = (0, len(v_range))  # note: right-close range
            else:
                bo_space[k] = (float(v['min']), float(v['max']))
        self.space = bo_space
        self.blank_space = [s for s in self.space.values()]
        self.config_default = default
        self.X = []
        self.Y = []
        self.dbms = dbms
        self.benchmark = benchmark
        self.times = -1
        self.use_env = use_env
        self.tuning_metric = tuning_metric
        # print("配置信息 "+self.dbms.get_value("work_mem"))

    def f(self, x):
        # self.set_params(x)
        # x = x[0]
        self.get_env()
        if self.use_env:
            x = self.deal_env(env, x)
        self.times = self.times + 1
        print("times is " + str(self.times))
        # print("f配置信息"+str(self.dbms.get_value("work_mem")))
        throughput, latency = self.benchmark_evaluate(x, self.times)
        if self.tuning_metric == "throughput":
            return -throughput
        elif self.tuning_metric == "latency":
            return latency
        else:
            raise KeyError(str(self.tuning_metric)+"not a tuning metric")
    def benchmark_evaluate(self, x, times):
        self.set_params(x)
        # self.benchmark.load_data()
        self.benchmark.run_benchmark(times)
        result_path = "/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/results/pgResult"
        result_path = result_path + "/pg_ycsb" + str(times) + ".txt"
        throughput, latency = self.benchmark.get_result(result_path)
        return throughput, latency

    def add_observation(self, x, y):
        # self.benchmark.get_result(result_path)
        self.X.append(x)
        self.Y.append(y)

    def get_env(self):
        if self.dbms.__class__.__name__ == "PgConfig":
            if self.benchmark.benchmark_name == "YCSB":
                hints = GPT_get.get_hints()
                useful_hints = self.get_useful_hints()

        return

    def get_useful_hints(self):
        pass

    def deal_env(self, env, x):
        return [1.5 * xi for xi in x]

    def get_res(self, n_calls):
        res = gp_minimize(self.f,  # the function to minimize
                          self.blank_space,  # the bounds on each dimension of x
                          x0=self.X,
                          y0=self.Y,
                          acq_func="EI",  # the acquisition function
                          n_calls=n_calls,  # the number of evaluations of f
                          random_state=1234
                          )  # the random seed
        return res

    def set_params(self, x):
        # self.dbms.set_param(list(self.config.keys())[0], x[0])
        self.dbms.set_param("work_mem", str(int(x[0])) + "kB")
        self.dbms.make_conf_effect()
        # self.restart_postgres()
        # print("当前dbms对象"+str(self.dbms))
        print("配置信息" + str(self.dbms.get_value("work_mem")))
        # print("当前配置的信息是"+str(self.dbms.get_value("work_mem")))

    def restart_postgres(self):
        # 停止 PostgreSQL
        stop_cmd = "sudo service pg stop"
        stop_result = subprocess.run(stop_cmd, shell=True, input="85581238\n", text=True, check=True)
        if stop_result.returncode == 0:
            print("PostgreSQL stopped successfully.")
        else:
            print("Error stopping PostgreSQL:")
            print(stop_result.stderr)

        # 启动 PostgreSQL
        start_cmd = "sudo service pg start"
        start_result = subprocess.run(start_cmd, shell=True, input="85581238\n", text=True, check=True)
        if start_result.returncode == 0:
            print("PostgreSQL started successfully.")
        else:
            print("Error starting PostgreSQL:")
            print(start_result.stderr)

# if __name__ == '__main__':
