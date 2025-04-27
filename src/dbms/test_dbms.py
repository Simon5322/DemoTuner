import os
import time
from configparser import ConfigParser
from argparse import ArgumentParser

import dbms.factory
from benchmark.YCSB import YCSB

from jinja2 import Template
from dbms.myMySql import MySQLconfig
from dbms.postgres import PgConfig

# from myMySql import MySQLconfig
# from dbms.postgres import PgConfig

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
fieldcount = int(config['WORKLOAD']['fieldcount'])
save_folder_name = config['SETTING']['save_name']
fake_test = True if config['LEARNING']['fake_test'] == 'True' else False
repeat_times = int(config['LEARNING']['repeat_times'])
# self.update('update mysql.server_cost set cost_value = NULL')
# self.update('update mysql.engine_cost set cost_value = NULL')
ycsb = YCSB('mysql', save_folder_name=save_folder_name)
dbms = dbms.factory.from_file(config)

print(dbms.dbms_name)
print(type(dbms))
print(MySQLconfig.__name__)
print(isinstance(dbms, MySQLconfig))#PgCon
print(PgConfig.__name__)
print(isinstance(dbms, PgConfig))
mb = int(1024 * 1024)
kb = int(1024)


def render(config_dict):
    with open('/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/src/dbms/template/mysqld_template.cnf',
              'r') as file:
        mysql_template = file.read()
    template = Template(mysql_template)
    context = config_dict
    rendered_config = template.render(context)
    with open("/etc/mysql/mysql.conf.d/mysqld.cnf", "w") as config_file:
        config_file.write(rendered_config)


unit = kb
config_name = 'preload_buffer_size'
value = 30
config_name2 = 'innodb_buffer_pool_size'
value2 = 3
config_value = dbms.get_value(config_name)
config_value2 = dbms.get_value(config_name2)


print('原始置参数值： ' + str(config_value) + '  ' + str(int(config_value) / 1*kb))
print('原始置参数值： ' + str(config_value2) + '  ' + str(int(config_value2) / 128*mb))

dbms.set_param(config_name, value)
dbms.set_param(config_name2, value2)
dbms.reconfigure()
config_value = dbms.get_value(config_name)
config_value2 = dbms.get_value(config_name2)
print('新置参数值： ' + str(config_value) + '  ' + str(int(config_value) / 1*kb))
print('新置参数值： ' + str(config_value2) + '  ' + str(int(config_value2) / 128*mb))


def test_config():
    throughput_list = []
    latency_list = []
    for config_value in range(1, 144, 10):
        dbms.set_param(config_name, int(config_value))
        dbms.make_conf_effect()
        dbms.reconfigure()
        throughput_temp = []
        latency_temp = []
        for i in range(3):
            dbms.create_new_usertable(fieldcount)
            ycsb.load_data()
            throughput, latency = ycsb.run_benchmark(-1)
            throughput_temp.append(throughput)
            latency_temp.append(latency)
            print('吞吐量 : ' + str(throughput) + '延迟 ： ' + str(latency))
            throughput_list.append(sum(throughput_temp) / len(throughput_temp))
            latency_list.append(sum(latency_temp) / len(latency_temp))

# success = dbms.set_param('innodb_buffer_pool_size', '134217728')
# dbms.reconfigure()
# time.sleep(3)
# <dbms.postgres.PgConfig object at 0x7fb90019f7f0>
# metrics = pg.get_metrics()


# 初始化数据库配置 测试当前负载的数据库状态
# dbms.reset_config()
# dbms.reconfigure()
# dbms.make_conf_effect()
# dbms.restart_dbms()
# dbms = dbms.factory.from_file(config)  # this step may be not necessary
# throughput_list, latency_list = [], []
# for i in range(repeat_times):
#     dbms.create_new_usertable(fieldcount=fieldcount)
#     ycsb.load_data()
#     throughput, latency = ycsb.run_benchmark(times=-1)
#     throughput_list.append(throughput)
#     latency_list.append(latency)
# throughput_default, latency_default = sum(throughput_list) / len(throughput_list), sum(latency_list) / len(
#     latency_list)
