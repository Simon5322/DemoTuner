import joblib
import pandas as pd
import yaml

import Global_Variables
from benchmark.benchmark import benchmark
from utils.util import real_to_action, trans_string_to_num


class ForestEstimateBenchmark(benchmark):
    def __init__(self, benchmark_name, confs, dbms_name):
        self.default_action = None
        self.benchmark_name = benchmark_name
        joblib_file = f"../../results/estimate_models/random_forest_model_{dbms_name}.pkl"
        self.loaded_rf_model = joblib.load(joblib_file)
        self.confs = confs
        self.dbms_name = dbms_name
        self.confs_range = {param: (value['min'], value['max']) for param, value in {**confs}.items()}
        self.confs_keys = [key for key in {**confs}.keys()]
        self.set_default_global_conf()  #用于第一次默认值的设定

    def load_data(self):
        pass

    def run_benchmark(self, times):
        action = Global_Variables.globalCurrentAction
        if action is None or len(action) == 0:
            raise KeyError('currentAction is not valid')

        real_confs = {}
        for key, a in zip(self.confs_keys, action):
            min_val = self.confs_range[key][0]
            max_val = self.confs_range[key][1]
            real_val = (a + 1)*(max_val - min_val)/2 + min_val
            real_confs[key] = [int(real_val)]

        confs = pd.DataFrame(real_confs)
        predicted_latency = self.loaded_rf_model.predict(confs)
        #print(f"Predicted Latency: {int(predicted_latency[0])}")
        return -1, int(predicted_latency[0])


    def get_result(self, times):
        pass

    def set_default_global_conf(self):
        confs = self.confs
        conf_real_value = {param: value['default'] for param, value in {**confs}.items()}
        confs_range = {param: (value['min'], value['max']) for param, value in {**confs}.items()}
        default_action = []
        for param, real_value in conf_real_value.items():
            max_val = confs_range[param][1]
            min_val = confs_range[param][0]
            default_a = 2 * (real_value - min_val) / (max_val - min_val) - 1
            default_action.append(default_a)
        Global_Variables.globalCurrentAction = default_action
        self.default_action = default_action

if __name__ == '__main__':
    dbms_name = 'pg'
    config_path = f'/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/src/tuning_params_states/{dbms_name}/{dbms_name}_params.yml'
    with open(config_path, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    f = ForestEstimateBenchmark('forestEstimater', conf, dbms_name)
    string_data_mysql = '143	100	1	1023	2	100	1	1	1	0	2000	1	4	3	2	22	0	2166	1'
    string_data_pg = '4096	512	1024	16384	524288	65536	100	2	0	200	8	2	32	1024	8	20	1024	64	2'
    real_values = trans_string_to_num(string_data_pg, split_type='\t')
    actions = real_to_action(dbms_name, real_values)
    Global_Variables.globalCurrentAction = actions
    _, latency = f.run_benchmark(-1)
    print(latency)



