import math
import os
from typing import List

import numpy as np
import yaml

from hintsClasses.Condition import Condition
from utils.util import drop_unit, take_unit


script_dir = os.path.dirname(os.path.abspath(__file__))
class TuningValue:
    """
    type:
    0: 绝对值 包含一个单位
    1: 相对值 包含一个值和一个相对的单位
    2: 调优趋势 大 小 修改
    """
    typ: int
    value: float

    def __init__(self, typ: int, value: float, unit: str = None):
        self.typ = typ
        self.value = value
        if unit:
            self.unit = unit
        else:
            self.unit = None

    def get_value(self, current_value, min_val, max_val, conf_name):
        # if conf_name in Hint.pg_configs_names:

        if conf_name not in Hint.mysql_configs_names + Hint.pg_configs_names:
            raise KeyError(f'{conf_name} not in mysql or pg')

        suggest_value = 0
        if current_value < min_val or current_value > max_val:
            raise ValueError(
                f'current value {current_value} smaller or larger than the range, min_val : {min_val} max_val : {max_val}')
        if current_value is None:
            raise ValueError('从Hint获取推荐值需要当前值')
        # 一个绝对值


        conf_8kb = ['wal_buffers', 'temp_buffers', 'shared_buffers', 'effective_cache_size', 'backend_flush_after',
                    'checkpoint_flush_after', 'min_parallel_table_scan_size', 'min_parallel_index_scan_size']
        conf_mb = ['max_wal_size']

        if self.typ == 0:
            unit_multipy = [1, 1024, 1024 * 1024]
            unit_types = ['kb', 'mb', 'gb']
            if self.unit is None:
                suggest_value = self.value
            elif self.unit in unit_types:   # all transfer to kb
                suggest_value = self.value * unit_multipy[unit_types.index(self.unit)]
            elif self.unit not in unit_types:
                raise KeyError('{} is not in kb, mb, gb'.format(str(self.unit)))
            if conf_name in conf_8kb:
                suggest_value = suggest_value / 8
            if conf_name in conf_mb:
                suggest_value = suggest_value / 1024
        elif self.typ == 1:
            if self.unit == 'mem':
                mem = 16 * 1024 * 1024
                suggest_value = self.value * mem
            elif self.unit == 'cpu':
                cpu_cores = 8
                suggest_value = self.value * cpu_cores
            else:
                raise KeyError("{} is not in 'mem'".format(self.unit))
            if conf_name in conf_8kb:
                suggest_value = suggest_value / 8
            if conf_name in conf_mb:
                suggest_value = suggest_value / 1024
        elif self.typ == 2:
            if current_value != 0:
                log_a = math.log(current_value)
            else:
                log_a = math.log(0.01)
            divided_parts_num = 6
            if min_val != 0:
                log_values = np.linspace(math.log(min_val), math.log(max_val), divided_parts_num)  # 指数分成divided_parts_num份
            else:
                log_values = np.linspace(math.log(0.01), math.log(max_val), divided_parts_num)  # 指数分成divided_parts_num)份

            vals = [math.exp(n) for n in log_values]  # 范围根据指数分为divided_parts_num份
            increase_val = 0
            for idx, log_value in enumerate(log_values):
                if current_value < min_val and idx == 0:
                    raise KeyError(f'a smaller than the min range . log_a：{log_a}, log_value:{log_value}')
                elif log_a < log_value and idx != 0:
                    increase_part = 3   # the part of increase of a gap
                    if idx == 1:
                        idx = 2
                        increase_val = ((vals[idx] - vals[idx - 1]) / increase_part) / 2
                    else:
                        increase_val = (vals[idx] - vals[idx - 1]) / increase_part
                    break
            if self.value == 0:  # increase
                suggest_value = round(float(current_value) + increase_val)
            elif self.value == 1:  # decrease
                suggest_value = round(float(current_value) - increase_val)
            else:
                raise ValueError('Invalid tuning value: {}'.format(self.value))
        else:
            raise ValueError('Invalid tuning type : {}'.format(self.typ))

        if conf_name in Hint.mysql_configs_names and self.typ != 2:
            suggest_value = take_unit(suggest_value*1024, Hint.mysql_config[conf_name]['unit'])

        suggest_value = round(suggest_value)
        if suggest_value > max_val:
            suggest_value = max_val
        elif suggest_value < min_val:
            suggest_value = min_val

        return suggest_value

    # elif conf_name in Hint.mysql_configs_names:
    #
    #     pass
    # else:
    #     raise KeyError(f'{conf_name} is not a mysql or pg config in file')

    def to_string(self):
        if self.typ == 0 and self.unit is not None:
            return str(self.value) + self.unit
        elif self.typ == 0 and self.unit is None:
            return str(self.value)
        elif self.typ == 1:
            return str(self.value) + self.unit
        elif self.typ == 2:
            return 'increase' if self.value == 0 else 'decrease'


class Hint:
    # config_name = ""
    # tuning_value
    config_name: str
    tuningValue: TuningValue
    conditions: List[str]
    hint_id: int

    script_path = os.path.dirname(os.path.abspath(__file__))
    mysql_configs_path = os.path.join(script_path, '../tuning_params_states/mysql/mysql_params.yml')
    pg_configs_path = os.path.join(script_path, '../tuning_params_states/pg/pg_params.yml')
    with open(mysql_configs_path) as file:
        mysql_config = yaml.load(file, Loader=yaml.FullLoader)
    mysql_configs_names = list(mysql_config.keys())
    with open(pg_configs_path) as file:
        pg_config = yaml.load(file, Loader=yaml.FullLoader)
    pg_configs_names = list(pg_config.keys())

    def __init__(self, config_name, tuningValue, condition: List[str], hint_id=-1):
        self.config_name = config_name
        self.tuningValue = tuningValue
        self.condition = condition
        self.hint_id = hint_id

    def set_hint_id(self, hint_id):
        if hint_id <= 0:
            raise KeyError('hint_id 应该大于0,而你赋值了{}'.format(hint_id))
        self.hint_id = hint_id

    def get_conf_name(self):
        return self.config_name

    def get_tuningValue(self):
        return self.tuningValue

    def get_conditions(self):
        return self.condition

    def get_hint_id(self):
        if self.hint_id > 0:
            return self.hint_id
        elif self.hint_id == 0:
            raise ValueError(f'当前hint_id为0 : {self.hint_id}')
        else:
            raise ValueError(f'当前hint_id为无效值 : {self.hint_id}')

    def to_string(self):
        return 'Hint: ( ' + self.config_name + ' , ' + self.tuningValue.to_string() + ' , ' + str(
            self.condition) + f' )  hint_id : {self.get_hint_id()} '

def get_all_hints(dbms_name, extracted=False):
    conditionTypes_path = os.path.join(script_dir, '../hintsClasses/ConditionType.txt')

    conditionTypes = []
    with open(conditionTypes_path, 'r') as f:
        types = f.read()

    for line in types:
        if line:
            conditionTypes.append(line)
    if dbms_name == 'pg':
        if not extracted:
            hints = [
                Hint('work_mem', TuningValue(0, 8, 'mb'), []),
                Hint('work_mem', TuningValue(0, 36, 'mb'), []),
                Hint('work_mem', TuningValue(0, 64, 'mb'), []),
                Hint('work_mem', TuningValue(1, 0.01, 'mem'), []),
                Hint('work_mem', TuningValue(1, 0.03, 'mem'), []),
                Hint('work_mem', TuningValue(1, 0.05, 'mem'), []),
                Hint('work_mem', TuningValue(2, 0), ['sort', 'memory available']),
                Hint('work_mem', TuningValue(2, 0), ['hash']),
                Hint('work_mem', TuningValue(2, 0), ['complex query']),
                Hint('work_mem', TuningValue(1, 0.25, 'mem'), ['analytics']),
                Hint('work_mem', TuningValue(2, 0), ['write heavy', 'dirty data in kernel page']),

                Hint('wal_buffers', TuningValue(0, 1, 'mb'), []),
                Hint('wal_buffers', TuningValue(0, 8, 'mb'), []),
                Hint('wal_buffers', TuningValue(0, 16, 'mb'), ['wal operation']),  # pg_stat_wal;
                Hint('wal_buffers', TuningValue(2, 0), ['write heavy', 'memory available']),
                Hint('wal_buffers', TuningValue(2, 1), ['not write heavy']),
                Hint('wal_buffers', TuningValue(2, 0), ['wal operation']),
                Hint('wal_buffers', TuningValue(2, 0), ['delay in wal data to disk']),  # pg_stat_wal;
                # Hint('wal_buffers', TuningValue(1, 0.03, 'shared_buffers'), ['wal operation']),
                Hint('wal_buffers', TuningValue(2, 0), ['many connections']),
                Hint('wal_buffers', TuningValue(2, 0), ['many concurrent transactions']),

                Hint('temp_buffers', TuningValue(2, 0), ['many connections']),
                Hint('temp_buffers', TuningValue(2, 0), ['frequent access temporary table']),
                # SELECT * FROM pg_stat_user_tables WHERE schemaname = 'pg_temp';

                Hint('shared_buffers', TuningValue(0, 2, 'gb'), []),
                Hint('shared_buffers', TuningValue(0, 2.5, 'gb'), []),
                Hint('shared_buffers', TuningValue(0, 4, 'gb'), []),
                Hint('shared_buffers', TuningValue(1, 0.2, 'mem'), []),
                Hint('shared_buffers', TuningValue(1, 0.4, 'mem'), []),
                Hint('shared_buffers', TuningValue(1, 0.25, 'mem'), []),
                Hint('shared_buffers', TuningValue(1, 0.15, 'mem'), ['not memory available']),
                Hint('shared_buffers', TuningValue(2, 0), ['more than 1GB mem']),
                Hint('shared_buffers', TuningValue(2, 1), ['less than 1GB mem']),
                Hint('shared_buffers', TuningValue(2, 0), ['issues with caching data']),  # 如果缓冲区命中率较低，表示数据大部分时间都没有在缓冲区中
                Hint('shared_buffers', TuningValue(2, 0), ['read heavy']),

                Hint('effective_cache_size', TuningValue(1, 0.5, 'mem'), []),
                Hint('effective_cache_size', TuningValue(1, 0.7, 'mem'), ['memory available']),
                Hint('effective_cache_size', TuningValue(1, 0.75, 'mem'), ['memory available']),
                Hint('effective_cache_size', TuningValue(1, 0.8, 'mem'), ['memory available']),
                Hint('effective_cache_size', TuningValue(2, 0), ['read heavy']),
                Hint('effective_cache_size', TuningValue(2, 0), ['read heavy', 'no index']),
                Hint('effective_cache_size', TuningValue(2, 0), ['index']),

                Hint('maintenance_work_mem', TuningValue(0, 256, 'mb'), []),
                Hint('maintenance_work_mem', TuningValue(0, 512, 'mb'), []),
                Hint('maintenance_work_mem', TuningValue(0, 1, 'gb'), ['maintenance operations']),
                Hint('maintenance_work_mem', TuningValue(1, 0.1, 'mem'), ['maintenance operations']),
                Hint('maintenance_work_mem', TuningValue(1, 0.15, 'mem'), ['maintenance operations']),
                Hint('maintenance_work_mem', TuningValue(1, 0.2, 'mem'), ['maintenance operations']),
                Hint('maintenance_work_mem', TuningValue(1, 0.05, 'mem'), []),
                Hint('maintenance_work_mem', TuningValue(2, 0), ['maintenance operations']),

                Hint('max_connections', TuningValue(2, 0), ['many connections']),
                Hint('max_connections', TuningValue(1, 4, 'cpu'), []),
                Hint('max_connections', TuningValue(1, 8, 'cpu'), []),
                Hint('max_connections', TuningValue(0, 500), []),
                Hint('max_connections', TuningValue(2, 1), ['not memory available']),
                Hint('max_connections', TuningValue(0, 25), ['not many cpus']),
                Hint('max_connections', TuningValue(0, 500), ['many cpus']),

                Hint('bgwriter_lru_multiplier', TuningValue(0, 1.5), ['write heavy']),
                Hint('bgwriter_lru_multiplier', TuningValue(0, 1), ['write heavy']),
                Hint('bgwriter_lru_multiplier', TuningValue(0, 2.5), ['write heavy']),
                Hint('bgwriter_lru_multiplier', TuningValue(0, 1), ['not write heavy']),
                Hint('bgwriter_lru_multiplier', TuningValue(0, 0.5), ['not write heavy']),
                Hint('bgwriter_lru_multiplier', TuningValue(0, 0.7), ['not write heavy']),
                Hint('bgwriter_lru_multiplier', TuningValue(0, 0.9), ['not write heavy']),
                Hint('bgwriter_lru_multiplier', TuningValue(2, 0), ['IO available']),
                Hint('bgwriter_lru_multiplier', TuningValue(2, 1), ['not IO available']),

                Hint('backend_flush_after', TuningValue(2, 0), ['write heavy', 'dirty data in kernel page']),
                Hint('backend_flush_after', TuningValue(2, 1), ['not write heavy', 'not dirty data in kernel page']),
                Hint('backend_flush_after', TuningValue(0, 20), []),
                Hint('backend_flush_after', TuningValue(0, 22), []),
                Hint('backend_flush_after', TuningValue(0, 25), []),

                Hint('bgwriter_delay', TuningValue(2, 1), ['write heavy']),
                Hint('bgwriter_delay', TuningValue(2, 0), ['write heavy', 'frequent dirty buffers']),
                Hint('bgwriter_delay', TuningValue(2, 1), ['not write heavy', 'not frequent dirty buffers']),

                Hint('max_parallel_workers', TuningValue(2, 0),
                     ['parallel operations', 'large dataset', 'many cpus', 'IO available']),
                Hint('max_parallel_workers', TuningValue(2, 1), ['parallel operations', 'not many cpus']),
                Hint('max_parallel_workers', TuningValue(2, 1), ['parallel operations', 'not IO available']),
                Hint('max_parallel_workers', TuningValue(1, 4, 'cpu'), []),
                Hint('max_parallel_workers', TuningValue(1, 8, 'cpu'), []),
                Hint('max_parallel_workers', TuningValue(2, 0), ['large queries']),
                Hint('max_parallel_workers', TuningValue(1, 7, 'cpu'), ['not Dynamic Shared Memory available']),

                Hint('hash_mem_multiplier', TuningValue(2, 0), ['hash', 'memory available']),
                Hint('hash_mem_multiplier', TuningValue(2, 0), ['large queries']),

                Hint('checkpoint_flush_after', TuningValue(2, 1), ['not memory available']),
                # , 'small shared_buffers']),
                Hint('checkpoint_flush_after', TuningValue(2, 0), ['memory available']),  # 'large shared_buffers']),

                Hint('max_wal_size', TuningValue(0, 8, 'gb'), []),
                Hint('max_wal_size', TuningValue(2, 0), ['large dataset']),
                Hint('max_wal_size', TuningValue(2, 0), ['write heavy']),
                Hint('max_wal_size', TuningValue(2, 1), ['not write heavy']),
                Hint('max_wal_size', TuningValue(2, 0), ['frequent checkpoint']),
                Hint('max_wal_size', TuningValue(2, 0), ['large queries']),
                Hint('max_wal_size', TuningValue(2, 1), ['not large queries']),

                Hint('vacuum_cost_page_dirty', TuningValue(2, 1), ['write heavy']),
                Hint('vacuum_cost_page_dirty', TuningValue(2, 1), ['insert or update']),
                Hint('vacuum_cost_page_dirty', TuningValue(2, 0), ['not write heavy']),

                Hint('min_parallel_table_scan_size', TuningValue(0, 8, 'mb'), []),
                Hint('min_parallel_table_scan_size', TuningValue(2, 1), ['small table']),
                Hint('min_parallel_table_scan_size', TuningValue(2, 0), ['large table']),
                Hint('min_parallel_table_scan_size', TuningValue(2, 1), ['read heavy']),
                Hint('min_parallel_table_scan_size', TuningValue(2, 0), ['not read heavy']),

                Hint('min_parallel_index_scan_size', TuningValue(0, 512, 'kb'), []),
                Hint('min_parallel_index_scan_size', TuningValue(2, 1), ['no index']),
                Hint('min_parallel_index_scan_size', TuningValue(2, 1), ['read heavy']),
                Hint('min_parallel_index_scan_size', TuningValue(2, 0), ['index']),
                Hint('min_parallel_index_scan_size', TuningValue(2, 0), ['not read heavy']),

                #Hint('max_parallel_workers_per_gather', TuningValue(0, 4), [])
            ]
        else:
            hints = [
                Hint('work_mem', TuningValue(0, 8, 'mb'), []),
                Hint('work_mem', TuningValue(0, 36, 'mb'), []),
                Hint('work_mem', TuningValue(0, 64, 'mb'), []),
                Hint('work_mem', TuningValue(1, 0.01, 'mem'), []),
                Hint('work_mem', TuningValue(1, 0.03, 'mem'), []),
                Hint('work_mem', TuningValue(1, 0.05, 'mem'), []),
                Hint('work_mem', TuningValue(2, 0), ['sort', 'memory available']),
                Hint('work_mem', TuningValue(2, 0), ['hash']),
                Hint('work_mem', TuningValue(2, 0), ['complex query']),
                Hint('work_mem', TuningValue(1, 0.25, 'mem'), ['analytics']),
                Hint('work_mem', TuningValue(2, 0), ['write heavy', 'dirty data in kernel page']),

                Hint('wal_buffers', TuningValue(0, 1, 'mb'), []),
                Hint('wal_buffers', TuningValue(0, 8, 'mb'), []),
                Hint('wal_buffers', TuningValue(0, 16, 'mb'), ['wal operation']),  # pg_stat_wal;
                Hint('wal_buffers', TuningValue(2, 0), ['write heavy', 'memory available']),
                Hint('wal_buffers', TuningValue(2, 1), ['not write heavy']),
                Hint('wal_buffers', TuningValue(2, 0), ['wal operation']),
                Hint('wal_buffers', TuningValue(2, 0), ['delay in wal data to disk']),  # pg_stat_wal;
                # Hint('wal_buffers', TuningValue(1, 0.03, 'shared_buffers'), ['wal operation']),
                Hint('wal_buffers', TuningValue(2, 0), ['many connections']),
                Hint('wal_buffers', TuningValue(2, 0), ['many concurrent transactions']),

                Hint('temp_buffers', TuningValue(2, 0), ['many connections']),
                Hint('temp_buffers', TuningValue(2, 0), ['frequent access temporary table']),
                # SELECT * FROM pg_stat_user_tables WHERE schemaname = 'pg_temp';

                Hint('shared_buffers', TuningValue(0, 2, 'gb'), []),
                Hint('shared_buffers', TuningValue(0, 2.5, 'gb'), []),
                Hint('shared_buffers', TuningValue(0, 4, 'gb'), []),
                Hint('shared_buffers', TuningValue(1, 0.2, 'mem'), []),
                Hint('shared_buffers', TuningValue(1, 0.4, 'mem'), []),
                Hint('shared_buffers', TuningValue(1, 0.25, 'mem'), []),
                Hint('shared_buffers', TuningValue(1, 0.15, 'mem'), ['not memory available']),
                Hint('shared_buffers', TuningValue(2, 0), ['more than 1GB mem']),
                Hint('shared_buffers', TuningValue(2, 1), ['less than 1GB mem']),
                Hint('shared_buffers', TuningValue(2, 0), ['issues with caching data']),  # 如果缓冲区命中率较低，表示数据大部分时间都没有在缓冲区中
                Hint('shared_buffers', TuningValue(2, 0), ['read heavy']),

                Hint('effective_cache_size', TuningValue(1, 0.5, 'mem'), []),
                Hint('effective_cache_size', TuningValue(1, 0.7, 'mem'), ['memory available']),
                Hint('effective_cache_size', TuningValue(1, 0.75, 'mem'), ['memory available']),
                Hint('effective_cache_size', TuningValue(1, 0.8, 'mem'), ['memory available']),
                Hint('effective_cache_size', TuningValue(2, 0), ['read heavy']),
                Hint('effective_cache_size', TuningValue(2, 0), ['read heavy', 'no index']),
                Hint('effective_cache_size', TuningValue(2, 0), ['index']),

                Hint('maintenance_work_mem', TuningValue(0, 256, 'mb'), []),
                Hint('maintenance_work_mem', TuningValue(0, 512, 'mb'), []),
                Hint('maintenance_work_mem', TuningValue(0, 1, 'gb'), ['maintenance operations']),
                Hint('maintenance_work_mem', TuningValue(1, 0.1, 'mem'), ['maintenance operations']),
                Hint('maintenance_work_mem', TuningValue(1, 0.15, 'mem'), ['maintenance operations']),
                Hint('maintenance_work_mem', TuningValue(1, 0.2, 'mem'), ['maintenance operations']),
                Hint('maintenance_work_mem', TuningValue(1, 0.05, 'mem'), []),
                Hint('maintenance_work_mem', TuningValue(2, 0), ['maintenance operations']),

                Hint('max_connections', TuningValue(2, 0), ['many connections']),
                Hint('max_connections', TuningValue(1, 4, 'cpu'), []),
                Hint('max_connections', TuningValue(1, 8, 'cpu'), []),
                Hint('max_connections', TuningValue(0, 500), []),
                Hint('max_connections', TuningValue(2, 1), ['not memory available']),
                Hint('max_connections', TuningValue(0, 25), ['not many cpus']),
                Hint('max_connections', TuningValue(0, 500), ['many cpus']),

                Hint('bgwriter_lru_multiplier', TuningValue(0, 1.5), ['write heavy']),
                Hint('bgwriter_lru_multiplier', TuningValue(0, 1), ['write heavy']),
                Hint('bgwriter_lru_multiplier', TuningValue(0, 2.5), ['write heavy']),
                Hint('bgwriter_lru_multiplier', TuningValue(0, 1), ['not write heavy']),
                Hint('bgwriter_lru_multiplier', TuningValue(0, 0.5), ['not write heavy']),
                Hint('bgwriter_lru_multiplier', TuningValue(0, 0.7), ['not write heavy']),
                Hint('bgwriter_lru_multiplier', TuningValue(0, 0.9), ['not write heavy']),
                Hint('bgwriter_lru_multiplier', TuningValue(2, 0), ['IO available']),
                Hint('bgwriter_lru_multiplier', TuningValue(2, 1), ['not IO available']),

                Hint('backend_flush_after', TuningValue(2, 0), ['write heavy', 'dirty data in kernel page']),
                Hint('backend_flush_after', TuningValue(2, 1), ['not write heavy', 'not dirty data in kernel page']),
                Hint('backend_flush_after', TuningValue(0, 20), []),
                Hint('backend_flush_after', TuningValue(0, 22), []),
                Hint('backend_flush_after', TuningValue(0, 25), []),

                Hint('bgwriter_delay', TuningValue(2, 1), ['write heavy']),
                Hint('bgwriter_delay', TuningValue(2, 0), ['write heavy', 'frequent dirty buffers']),
                Hint('bgwriter_delay', TuningValue(2, 1), ['not write heavy', 'not frequent dirty buffers']),

                Hint('max_parallel_workers', TuningValue(2, 0),
                     ['parallel operations', 'large dataset', 'many cpus', 'IO available']),
                Hint('max_parallel_workers', TuningValue(2, 1), ['parallel operations', 'not many cpus']),
                Hint('max_parallel_workers', TuningValue(2, 1), ['parallel operations', 'not IO available']),
                Hint('max_parallel_workers', TuningValue(1, 4, 'cpu'), []),
                Hint('max_parallel_workers', TuningValue(1, 8, 'cpu'), []),
                Hint('max_parallel_workers', TuningValue(2, 0), ['large queries']),
                Hint('max_parallel_workers', TuningValue(1, 7, 'cpu'), ['not Dynamic Shared Memory available']),

                Hint('hash_mem_multiplier', TuningValue(2, 0), ['hash', 'memory available']),
                Hint('hash_mem_multiplier', TuningValue(2, 0), ['large queries']),

                Hint('checkpoint_flush_after', TuningValue(2, 1), ['not memory available']),
                # , 'small shared_buffers']),
                Hint('checkpoint_flush_after', TuningValue(2, 0), ['memory available']),  # 'large shared_buffers']),

                Hint('max_wal_size', TuningValue(0, 8, 'gb'), []),
                Hint('max_wal_size', TuningValue(2, 0), ['large dataset']),
                Hint('max_wal_size', TuningValue(2, 0), ['write heavy']),
                Hint('max_wal_size', TuningValue(2, 1), ['not write heavy']),
                Hint('max_wal_size', TuningValue(2, 0), ['frequent checkpoint']),
                Hint('max_wal_size', TuningValue(2, 0), ['large queries']),
                Hint('max_wal_size', TuningValue(2, 1), ['not large queries']),

                Hint('vacuum_cost_page_dirty', TuningValue(2, 1), ['write heavy']),
                Hint('vacuum_cost_page_dirty', TuningValue(2, 1), ['insert or update']),
                Hint('vacuum_cost_page_dirty', TuningValue(2, 0), ['not write heavy']),

                Hint('min_parallel_table_scan_size', TuningValue(0, 8, 'mb'), []),
                Hint('min_parallel_table_scan_size', TuningValue(2, 1), ['small table']),
                Hint('min_parallel_table_scan_size', TuningValue(2, 0), ['large table']),
                Hint('min_parallel_table_scan_size', TuningValue(2, 1), ['read heavy']),
                Hint('min_parallel_table_scan_size', TuningValue(2, 0), ['not read heavy']),

                Hint('min_parallel_index_scan_size', TuningValue(0, 512, 'kb'), []),
                Hint('min_parallel_index_scan_size', TuningValue(2, 1), ['no index']),
                Hint('min_parallel_index_scan_size', TuningValue(2, 1), ['read heavy']),
                Hint('min_parallel_index_scan_size', TuningValue(2, 0), ['index']),
                Hint('min_parallel_index_scan_size', TuningValue(2, 0), ['not read heavy']),
            ]
    elif dbms_name == 'mysql':
        hints = [
            Hint('innodb_buffer_pool_size', TuningValue(0, 6, 'gb'), []),
            Hint('innodb_buffer_pool_size', TuningValue(0, 5, 'gb'), []),
            Hint('innodb_buffer_pool_size', TuningValue(0, 8, 'gb'), ['large dataset']),
            Hint('innodb_buffer_pool_size', TuningValue(1, 0.7, 'mem'), ['low buffer ratio']),
            Hint('innodb_buffer_pool_size', TuningValue(1, 0.8, 'mem'), ['low buffer ratio']),
            Hint('innodb_buffer_pool_size', TuningValue(1, 0.75, 'mem'), ['write heavy']),
            Hint('innodb_buffer_pool_size', TuningValue(1, 0.75, 'mem'), ['read heavy']),
            Hint('innodb_buffer_pool_size', TuningValue(1, 0.5, 'mem'), ['memory available', 'low buffer ratio']),
            Hint('innodb_buffer_pool_size', TuningValue(1, 0.6, 'mem'), ['memory available', 'many connections']),
            Hint('innodb_buffer_pool_size', TuningValue(1, 0.75, 'mem'), ['memory available']),
            Hint('innodb_buffer_pool_size', TuningValue(1, 0.72, 'mem'), ['memory available']),
            Hint('innodb_buffer_pool_size', TuningValue(2, 0), ['low buffer ratio']),
            Hint('innodb_buffer_pool_size', TuningValue(2, 0), ['large dataset']),
            Hint('innodb_buffer_pool_size', TuningValue(2, 0), ['large queries']),

            Hint('innodb_io_capacity', TuningValue(2, 0), ['dirty data in buffer pool']),
            Hint('innodb_io_capacity', TuningValue(0, 100), ['HDD']),
            Hint('innodb_io_capacity', TuningValue(0, 1000), ['SSD']),
            Hint('innodb_buffer_pool_size', TuningValue(2, 0), ['write heavy']),

            Hint('innodb_log_buffer_size', TuningValue(2, 0), ['log wait']),
            Hint('innodb_log_buffer_size', TuningValue(0, 8, 'mb'), []),
            #Hint('innodb_log_buffer_size', TuningValue(0, 64, 'mb'), []),   # new
            Hint('innodb_log_buffer_size', TuningValue(0, 256, 'mb'), []),
            Hint('innodb_log_buffer_size', TuningValue(2, 0), ['large queries']),
            Hint('innodb_log_buffer_size', TuningValue(2, 0), ['write heavy']),

            Hint('innodb_log_file_size', TuningValue(0, 512, 'mb'), ['write heavy']),
            Hint('innodb_log_file_size', TuningValue(0, 256, 'mb'), ['write heavy']),
            Hint('innodb_log_file_size', TuningValue(2, 0), ['write heavy']),
            Hint('innodb_log_file_size', TuningValue(0, 64, 'mb'), []),
            Hint('innodb_log_file_size', TuningValue(0, 128, 'mb'), []),
            Hint('innodb_log_file_size', TuningValue(0, 500, 'mb'), []),
            Hint('innodb_log_file_size', TuningValue(0, 1024, 'mb'), []),
            Hint('innodb_log_file_size', TuningValue(0, 20, 'gb'), []),
            Hint('innodb_log_file_size', TuningValue(0, 4, 'gb'), ['write heavy']),
            Hint('innodb_log_file_size', TuningValue(0, 2, 'gb'), []),

            Hint('innodb_log_files_in_group', TuningValue(2, 0), ['write heavy']),
            Hint('innodb_log_files_in_group', TuningValue(0, 5, 'gb'), []),

            Hint('innodb_lru_scan_depth', TuningValue(2, 1), ['write heavy']),

            Hint('innodb_purge_threads', TuningValue(2, 0), ['purge operation']),

            Hint('innodb_read_io_threads', TuningValue(0, 8), ['many connections']),
            Hint('innodb_read_io_threads', TuningValue(1, 1, 'cpu'), ['many connections']),
            Hint('innodb_read_io_threads', TuningValue(2, 0), ['many connections']),

            Hint('innodb_write_io_threads', TuningValue(0, 8), ['write heavy']),

            Hint('innodb_thread_concurrency', TuningValue(0, 16), []),
            Hint('innodb_thread_concurrency', TuningValue(2, 0), ['write heavy']),

            Hint('join_buffer_size', TuningValue(2, 0), ['read heavy', 'lacking index']),
            Hint('join_buffer_size', TuningValue(2, 0), ['join operation']),
            Hint('join_buffer_size', TuningValue(0, 50, 'mb'), ['join operation']),
            Hint('join_buffer_size', TuningValue(0, 256, 'kb'), []),

            Hint('max_heap_table_size', TuningValue(2, 0), ['heap table operations']),
            Hint('max_heap_table_size', TuningValue(0, 64, 'mb'), []),
            Hint('max_heap_table_size', TuningValue(0, 48, 'mb'), []),
            Hint('max_heap_table_size', TuningValue(0, 32, 'mb'), []),

            Hint('read_buffer_size', TuningValue(0, 2, 'mb'), ['sequential scan']),
            Hint('read_buffer_size', TuningValue(2, 0), ['read heavy']),
            Hint('read_buffer_size', TuningValue(2, 0), ['sequential scan']),

            Hint('sort_buffer_size', TuningValue(2, 0), ['sort', 'join operation', 'insert', 'update']),
            Hint('sort_buffer_size', TuningValue(0, 50, 'mb'), ['sort']),
            Hint('sort_buffer_size', TuningValue(2, 0), ['read heavy']),

            Hint('preload_buffer_size', TuningValue(2, 0), ['no lacking index']),

            Hint('table_open_cache', TuningValue(2, 0), ['many connections']),
            Hint('table_open_cache', TuningValue(0, 2000), []),

            Hint('thread_cache_size', TuningValue(2, 0), ['many connections']),
            Hint('thread_cache_size', TuningValue(0, 8), ['many connections']),

            Hint('tmp_table_size', TuningValue(2, 0), ['WO_noHIntUpdate tables']),
            Hint('tmp_table_size', TuningValue(0, 1, 'gb'), ['complex queries']),
            Hint('tmp_table_size', TuningValue(0, 32, 'mb'), []),
            Hint('tmp_table_size', TuningValue(0, 48, 'mb'), []),
            Hint('tmp_table_size', TuningValue(0, 64, 'mb'), []),

            Hint('net_buffer_length', TuningValue(0, 1, 'mb'), []),

            # Hint('thread_pool_size', TuningValue(1, 1, 'cpu'), ['many connections']),
            # Hint('thread_pool_size', TuningValue(1, 1, 'cpu'), ['write heavy']),
        ]
    else:
        raise KeyError(f'{dbms_name} is not valid in mysql or pg')

    # Hint('join_collapse_limit', TuningValue(2, 1), ['large join']),
    # Hint('join_collapse_limit', TuningValue(2, 0), ['not large join']),

    for hint_id, hint in enumerate(hints, start=1):
        hint.set_hint_id(hint_id)
    return hints


if __name__ == '__main__':
    hints = get_all_hints('mysql')
    with open(Hint.mysql_configs_path, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    for hint in hints:
        conf_name = hint.get_conf_name()
        default = configs[conf_name]['default']
        min = configs[conf_name]['min']
        max = configs[conf_name]['max']
        v = hint.get_tuningValue().get_value(default, min, max, conf_name)
        try:
            print(conf_name + '  ' + str(hint.get_tuningValue().get_value(default, min, max, conf_name)))
        except:
            raise KeyError(f'!!!!!!!!!!!!!!!!!!!!!!{hint.to_string()}')


        #
