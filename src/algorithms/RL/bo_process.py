import os
from pathlib import Path

import pandas as pd

import dbms
import yaml

from algorithms.bo import BOenvOptimizer
from benchmark.YCSB import YCSB
from tuning_run import tuning_metric, tuning_times
script_dir = os.path.dirname(os.path.abspath(__file__))  

def bo_tuning(config):
    n = 5
    x = []
    fun = []
    params_setting_path = "/tuning_params_states/pg_params.yml"
    while(n>0):
        #建立dbms和benchamrk
        pg = dbms.factory.from_file(config)
        benchmark = YCSB()

        # 输入配置范围

        params_setting = yaml.load(Path(params_setting_path).read_text(), Loader=yaml.FullLoader)
        configs = {
            **params_setting,
        }

        # 建立优化器 并加入初始集合
        bo = BOenvOptimizer(configs, pg, benchmark, tuning_metric, True)
        work_mem_default = bo.config_default["work_mem"]
        params = []
        params.append(work_mem_default)
        thr = bo.f(params)
        bo.add_observation(bo.config_default["work_mem"], thr)

        #进行迭代和优化
        res = bo.get_res(tuning_times)

        x.append(res.x[0])
        fun.append(res.fun)
        print(res)
        n=n-1

    df = pd.DataFrame([x, fun])
    output_file = '/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/results/execlResult/Env.xlsx'
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    df.to_excel(output_file, index=False, header=False)