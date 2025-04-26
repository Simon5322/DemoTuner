import os
import re
import tracemalloc
from argparse import ArgumentParser
from configparser import ConfigParser


import yaml

import dbms.factory
import psutil
# 假设这里是你的程序运行的代码
import time
import multiprocessing
# 获取 CPU 利用率
import psutil
import time

from benchmark.YCSB import YCSB

from utils.prometheus import Prometheus

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

innodb_sql = 'SHOW ENGINE INNODB STATUS'
global_sql = 'SHOW GLOBAL STATUS'
thread_sql = "SHOW STATUS LIKE 'Threads_connected'"
sql_reads = "SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_reads';"
select_sql = "SHOW GLOBAL STATUS LIKE 'Com_select'"
insert_sql = "SHOW GLOBAL STATUS LIKE 'Com_insert'"
update_sql = "SHOW GLOBAL STATUS LIKE 'Com_update'"
delete_sql = "SHOW GLOBAL STATUS LIKE 'Com_delete'"

result_reads = int(dbms.query_all(select_sql)[0][1])
result_insert = int(dbms.query_all(insert_sql)[0][1])
result_update = int(dbms.query_all(update_sql)[0][1])
result_delete = int(dbms.query_all(delete_sql)[0][1])
all = result_delete+result_update+result_insert+result_reads
print(all)

dbms.create_new_usertable(fieldcount)
ycsb.load_data()
throughput, latency = ycsb.run_benchmark(-1)

result_reads = int(dbms.query_all(select_sql)[0][1])
result_insert = int(dbms.query_all(insert_sql)[0][1])
result_update = int(dbms.query_all(update_sql)[0][1])
result_delete = int(dbms.query_all(delete_sql)[0][1])
all2 = result_delete+result_update+result_insert+result_reads
print(all2)
print(f'相减{all2 - all}')

a =1
# result= dbms.query_all(thread_sql)
# buffer_pool_reads = int(result[0][1])
#
# print(buffer_pool_reads)
# print('buffer_pool_reads' + str(buffer_pool_reads))
# # 获取Buffer Pool Requests
# sql_requests = "SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_read_requests';"
# result_requests = dbms.query_all(sql_requests)
# buffer_pool_requests = int(result_requests[0][1])
# print('buffer_pool_requests'+str(buffer_pool_requests))
# if buffer_pool_requests > 0:
#     hit_rate = (1 - (buffer_pool_reads / buffer_pool_requests)) * 100
#     print(f"Buffer Pool Hit Rate: {hit_rate:.2f}%")
# else:
#     print("无有效的Buffer Pool请求数据进行命中率计算")






sql_reads = "SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_reads';"
result_reads = dbms.query_all(sql_reads)
buffer_pool_reads2 = int(result_reads[0][1])
print('buffer_pool_reads' + str(buffer_pool_reads2))
# 获取Buffer Pool Requests
sql_requests = "SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_read_requests';"
result_requests = dbms.query_all(sql_requests)
buffer_pool_requests2 = int(result_requests[0][1])
print('buffer_pool_requests'+str(buffer_pool_requests2))
if buffer_pool_requests2 > 0:
    buffer_pool_reads = buffer_pool_reads2 - buffer_pool_reads
    buffer_pool_requests = buffer_pool_requests2 - buffer_pool_requests
    hit_rate = (1 - (buffer_pool_reads / buffer_pool_requests)) * 100
    print(f"Buffer Pool Hit Rate: {hit_rate:.2f}%")
else:
    print("无有效的Buffer Pool请求数据进行命中率计算")

a =1



def test_pg():
    prometheus = Prometheus()
    cpu_init = prometheus.query('100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)')[0]['value'][1]
    memery_init = prometheus.query('100 * (1 - node_memory_MemFree_bytes / node_memory_MemTotal_bytes)')[0]['value'][1]
    write_speed_init = prometheus.query('irate(node_disk_written_bytes_total[15s])')[0]['value'][1]
    IO_time_init = prometheus.query('irate(node_disk_io_time_seconds_total{job="node", device="sda"}[10s])')[0]['value'][1]
    dirty_byte_init = prometheus.query('node_memory_Dirty_bytes')[0]['value'][1]
    load_init = prometheus.query('node_load1')[0]['value'][1]
    cpu = psutil.cpu_percent(interval=0, percpu=True)
    buffer_backend_init = prometheus.query('pg_stat_bgwriter_buffers_backend_total')[0]['value'][1]
    buffers_checkpoint_init = prometheus.query('pg_stat_bgwriter_buffers_checkpoint_total')[0]['value'][1]

    dbms.create_new_usertable(fieldcount)
    ycsb.load_data()
    throughput, latency = ycsb.run_benchmark(-1)
    dbms.query_one('select pg_stat_clear_snapshot()')

    result2 = dbms.get_metrics()

    prometheus = Prometheus()
    cpu1 = psutil.cpu_percent(interval=0, percpu=False)
    cpu_after = prometheus.query('100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)')[0]['value'][1]
    memery_after = prometheus.query('100 * (1 - node_memory_MemFree_bytes / node_memory_MemTotal_bytes)')[0]['value'][1]
    write_speed_after = prometheus.query('irate(node_disk_written_bytes_total[15s])')[0]['value'][1]
    IO_time_after = prometheus.query('irate(node_disk_io_time_seconds_total{job="node", device="sda"}[10s])')[0]['value'][1] # / irate(node_disk_writes_completed_total{job="node", device="sda"}[10s])')
    dirty_byte_after = prometheus.query('node_memory_Dirty_bytes')[0]['value'][1]
    buffer_backend_after = prometheus.query('pg_stat_bgwriter_buffers_backend_total')[0]['value'][1]
    buffers_checkpoint_after = prometheus.query('pg_stat_bgwriter_buffers_checkpoint_total')[0]['value'][1]
    load_after = prometheus.query('node_load1')[0]['value'][1]
    print(f' CPU : {cpu1}%')
    print(f'初始 MEM 利用率: {memery_init}%, 最终 MEM 利用率: {memery_after}%')
    print(f'初始 CPU 利用率: {cpu_init}%,最终 CPU 利用率: {cpu_after}%')
    print(f'初始 write 速率: {write_speed_init},最终 write 速率: {write_speed_after}')
    print(f'初始 write 延迟: {IO_time_init},最终 write 延迟: {IO_time_after}')
    print(f'初始 dirty bytes : {dirty_byte_init},最终 dirty byte : {dirty_byte_after}')
    print(f'初始 load : {load_init},最终 load : {load_after}')
    print(f'初始 buffers_backend : {buffer_backend_init},最终 buffers_backend : {buffer_backend_after}')
    print(f'初始 buffers_checkpoint : {buffers_checkpoint_init},最终 buffers_checkpoint : {buffers_checkpoint_after}')


# cpu_usage_per_core = psutil.cpu_percent(interval=None, percpu=True)
#total_cpu_usage = sum(cpu_usage_per_core) / len(cpu_usage_per_core)







