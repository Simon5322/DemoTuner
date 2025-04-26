'''
Created on Apr 7, 2021

@author: immanueltrummer
'''
import shutil
import subprocess

import yaml
import re

from jinja2 import Template

from dbms.generic_dbms import ConfigurableDBMS

import mysql.connector

import os
from parameters.util import is_numerical
import time

from utils.prometheus import Prometheus
from utils.util import take_unit


class MySQLconfig(ConfigurableDBMS):
    """ Represents configurable MySQL database. """

    def __init__(self, db, user, password,
                 restart_cmd, recovery_cmd, timeout_s):
        """ Initialize DB connection with given credentials. 
        
        Args:
            db: name of MySQL database
            user: name of MySQL user
            password: password for database
            restart_cmd: command to restart server
            recovery_cmd: command to recover database
            timeout_s: per-query timeout in seconds
        """

        unit_to_size = {'KB': '000', 'MB': '000000', 'GB': '000000000',
                        'K': '000', 'M': '000000', 'G': '000000000'}
        super().__init__(db, user, password, unit_to_size,
                         restart_cmd, recovery_cmd, timeout_s)
        self.global_vars = [t[0] for t in self.query_all(
            'show global variables') if is_numerical(t[1])]
        self.server_cost_params = [t[0] for t in self.query_all(
            'select cost_name from mysql.server_cost')]
        self.engine_cost_params = [t[0] for t in self.query_all(
            'select cost_name from mysql.engine_cost')]
        self.all_variables = self.global_vars + \
                             self.server_cost_params + self.engine_cost_params
        config_path = '/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/src/tuning_params_states/mysql/mysql_params.yml'
        with open(config_path, "r") as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        self.conf = {**conf}
        self.unit_table = {
            'B': 1,
            'KB': 1024,
            'MB': 1024 * 1024,
            'GB': 1024 * 1024 * 1024,
        }
        self.dbms_name = 'mysql'
        print(f'Global variables: {self.global_vars}')
        print(f'Server cost parameters: {self.server_cost_params}')
        print(f'Engine cost parameters: {self.engine_cost_params}')
        print(f'All parameters: {self.all_variables}')

    @classmethod
    def from_file(cls, config):
        """ Initialize DBMS from configuration file. 
        
        Args:
            cls: class (currently: MySQLconfig only)
            config: configuration read from file
            
        Returns:
            new MySQL DBMS object
        """
        db_user = config['DATABASE']['user']
        db_name = config['DATABASE']['database_name']
        password = config['DATABASE']['password']
        restart_cmd = config['DATABASE']['restart_cmd']
        recovery_cmd = config['DATABASE']['recovery_cmd']
        timeout_s = config['LEARNING']['timeout_s']
        return cls(db_name, db_user, password,
                   restart_cmd, recovery_cmd, timeout_s)

    def __del__(self):
        """ Close DBMS connection if any. """
        super().__del__()

    def copy_db(self, source_db, target_db):
        """ Copy source to target database. """
        ms_clc_prefix = f'mysql -u{self.user} -p{self.password} '
        ms_dump_prefix = f'mysqldump -u{self.user} -p{self.password} '
        os.system(ms_dump_prefix + f' {source_db} > copy_db_dump')
        print('Dumped old database')
        os.system(ms_clc_prefix + f" -e 'drop database if exists {target_db}'")
        print('Dropped old database')
        os.system(ms_clc_prefix + f" -e 'create database {target_db}'")
        print('Created new database')
        os.system(ms_clc_prefix + f" {target_db} < copy_db_dump")
        print('Initialized new database')

    def _connect(self):
        """ Establish connection to database, returns success flag. 
        
        Returns:
            True if connection attempt is successful
        """
        print(f'Trying to connect to {self.db} with user {self.user}')
        # Need to recover in case of bad configuration
        try:
            self.connection = mysql.connector.connect(
                database=self.db, user=self.user,
                password=self.password, host="localhost")
            self.set_timeout(self.timeout_s)
            self.failed_connections = 0
            return True
        except Exception as e:
            print(f'Exception while trying to connect to MySQL: {e}')
            self.failed_connections += 1
            print(f'Had {self.failed_connections} failed tries.')
            if self.failed_connections < 3:
                print(f'Trying recovery with "{self.recovery_cmd}" ...')
                os.system(self.recovery_cmd)
                os.system(self.restart_cmd)
                self.reset_config()
                self.reconfigure()
            return False

    def _disconnect(self):
        """ Disconnect from database. """
        if self.connection:
            print('Disconnecting ...')
            self.connection.close()

    def exec_file(self, path):
        """ Executes all SQL queries in given file and returns error flag. """
        try:
            self.connection.autocommit = True
            with open(path) as file:
                sql = file.read()
                for query in sql.split(';'):
                    cursor = self.connection.cursor(buffered=True)
                    cursor.execute(query)
                    cursor.close()
            error = False
        except Exception as e:
            error = True
            print(f'Exception executing {path}: {e}')
        return error

    def query_one(self, sql):
        """ Runs SQL query_one and returns one result if it succeeds. """
        try:
            cursor = self.connection.cursor(buffered=True)
            cursor.execute(sql)
            return cursor.fetchone()[0]
        except Exception:
            return None

    def query_all(self, sql):
        """ Runs SQL query and returns all results if it succeeds. """
        try:
            cursor = self.connection.cursor(buffered=True)
            cursor.execute(sql)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            print(f'Exception in mysql.query_all: {e}')
            return None

    def update(self, sql):
        """ Runs an SQL update and returns true iff the update succeeds. """
        # print(f'Trying update {sql}')
        self.connection.autocommit = True
        cursor = self.connection.cursor(buffered=True)
        try:
            cursor.execute(sql)
            success = True
        except Exception as e:
            # print(f'Exception during update: {e}')
            success = False
        cursor.close()
        return success

    def is_param(self, param):
        """ Returns True iff the given parameter can be configured. """
        return param in self.all_variables

    def get_value(self, param):
        """ Returns current value for given parameter. """
        if param in self.global_vars:
            source_value = self.query_one(f'select @@{param}')
            return take_unit(source_value, self.conf[param]['unit'])

        elif param in self.server_cost_params:
            return self.query_one(
                "select case when cost_value is NULL then default_value " \
                "else cost_value end from mysql.server_cost " \
                f"where cost_name='{param}'")
        elif param in self.engine_cost_params:
            return self.query_one(
                "select case when cost_value is NULL then default_value " \
                "else cost_value end from mysql.engine_cost " \
                f"where cost_name='{param}'")
        else:
            return None
    def get_value_without_unit(self, param):
        source_value = self.query_one(f'select @@{param}')
        return source_value

    def set_param_without_unit(self, param, value):
        success = True
        if success:
            self.config[param] = value
        return success

    def set_param(self, param, value):
        full_unit = self.conf.get(param)['unit']
        if full_unit != 'None':
            pattern = r'(\d+)(\w+)'
            result = re.search(pattern, full_unit)
            num = result.group(1)
            unit = result.group(2)  # KB or MB or GB
            value = int(value) * int(num) * int(self.unit_table[unit])
        else:
            value = value
        """ Set parameter to given value. """
        # if param in self.global_vars:
        #     success = self.update(f'set global {param}={value}')
        #     #success = self.update_config_file(param, value)
        # elif param in self.server_cost_params:
        #     success = self.update(
        #         f"update mysql.server_cost set cost_value={value} where cost_name='{param}'")
        # elif param in self.engine_cost_params:
        #     success = self.update(
        #         f"update mysql.engine_cost set cost_value={value} where cost_name='{param}'")
        # else:
        #     success = False
        success = True
        if success:
            self.config[param] = value
        return success

    def set_timeout(self, timeout_s):
        """ Set per-query timeout. """
        timeout_ms = int(timeout_s * 1000)
        self.update(f"set session max_execution_time = {timeout_ms}")

    def all_params(self):
        """ Returns list of tuples, containing configuration parameters and values. """
        return self.all_variables

    def reset_config(self):
        """ Reset all parameters to default values. """
        self.config = {}
        # self._disconnect()
        # os.system(self.restart_cmd)
        # time.sleep(2)
        # self._connect()
        # self.update('update mysql.server_cost set cost_value = NULL')
        # self.update('update mysql.engine_cost set cost_value = NULL')
        print('reset all configurations to default')

    def make_conf_effect(self, restart=False):
        """ Get current value of given parameter. """
        return self.update('flush optimizer_costs')

    def reconfigure(self):
        """ Makes all parameter changes take effect (may require restart). 
        
        Returns:
            Whether reconfiguration was successful
        """
        # Optimizer cost parameters requires flush and reconnect
        self.update('flush optimizer_costs')
        self.render(self.config)
        self._disconnect()
        os.system(self.restart_cmd)
        time.sleep(2)
        self._connect()
        # Currently, we consider no MySQL parameters requiring restart
        return True

    def restart_dbms(self):
        self._disconnect()
        os.system(self.restart_cmd)
        time.sleep(2)
        self._connect()
        # self.update('update mysql.server_cost set cost_value = NULL')
        # self.update('update mysql.engine_cost set cost_value = NULL')
        # self.config = {}
        return True

    def execute(self, sql):
        """ Executes query_one and returns first result table cell or None. """
        try:
            self.connection.autocommit = True
            cursor = self.connection.cursor()
            cursor.execute(sql)
            return None
        except Exception as e:
            print("error" + str(e))
            return None

    def check_conf_set(self, conf_name, conf_value):
        real_conf_value = self.get_value(conf_name)
        full_unit = self.conf.get(conf_name)['unit']
        if full_unit == 'None':
            print(f'conf_name : {conf_name}, new_value : {int(conf_value)}, real_value : {real_conf_value}')
            return
        pattern = r'(\d+)(\w+)'
        result = re.search(pattern, full_unit)
        num = result.group(1)
        unit = result.group(2)  # KB or MB or GB
        print(f'conf_name : {conf_name}, new_value : {int(conf_value) * int(num) * int(self.unit_table.get(unit))}, real_value : {real_conf_value}')

    def create_new_usertable(self, fieldcount):
        # Drop the existing table if it exists
        self.execute('DROP TABLE IF EXISTS usertable;')

        # Start building the SQL statement for creating the new table
        sql = "CREATE TABLE usertable (YCSB_KEY VARCHAR(255) PRIMARY KEY"

        # Dynamically add column definitions to the SQL statement
        for i in range(fieldcount):
            sql += ", FIELD{} TEXT".format(i)

        # Close the CREATE TABLE statement
        sql += ");"

        # Execute the CREATE TABLE statement
        self.execute(sql)

    import subprocess
    import re

    def render(self, config_dict):
        with open('/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/src/dbms/template/mysqld_template.cnf',
                  'r') as file:
            mysql_template = file.read()
        template = Template(mysql_template)
        rendered_config = template.render(config_dict)
        local_des = '/home/zhouyuxuan/workspace/pythonWorkspace/bertProject/src/dbms/config_files/mysqld.cnf'
        target_file = '"/etc/mysql/mysql.conf.d/mysqld.cnf"'
        with open(local_des, "w") as config_file:
            config_file.write(rendered_config)
        sudo_command = f'sudo cp {local_des} {target_file}'
        os.system(sudo_command)

    def get_metrics(self):
        metrics = {}
        script_directory = os.path.dirname(os.path.abspath(__file__))
        # 构建相对路径
        metrics_path = os.path.join(script_directory, '../tuning_params_states/mysql/mysql_metrics.yml')
        with open(metrics_path, "r") as f:
            metrics_range = yaml.load(f, Loader=yaml.FullLoader)
        metrics_keys = list(metrics_range.keys())

        # 系统指标
        prometheus = Prometheus()
        cpu_usage = prometheus.query('100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[60s])) * 100)')[0][
            'value'][1]
        memery_usage = prometheus.query('100 * (1 - node_memory_MemFree_bytes / node_memory_MemTotal_bytes)')[0]['value'][1]
        write_speed = prometheus.query('irate(node_disk_written_bytes_total{job="node", device="sda"}[60s])')[0]['value'][1]
        io_time = prometheus.query('irate(node_disk_io_time_seconds_total{job="node", device="sda"}[60s])')[0]['value'][1]
        load1 = prometheus.query('node_load1')[0]['value'][1]
        dirty_byte = prometheus.query('node_memory_Dirty_bytes')[0]['value'][1]

        metrics['cpu_changed'] = float(cpu_usage)
        metrics['cpu_usage'] = float(cpu_usage)
        metrics['mem_changed'] = float(memery_usage)
        metrics['mem_usage'] = float(memery_usage)
        metrics['write_speed_changed'] = float(write_speed)
        metrics['io_latency_changed'] = float(io_time)
        metrics['load1'] = float(load1)
        metrics['node_memory_Dirty_bytes'] = int(dirty_byte)

        # mysql指标

        global_sql = "SHOW GLOBAL STATUS LIKE "
        global_metrics = [key for key, value in metrics_range.items() if value.get('view') == 'global']

        status_sql = "SHOW STATUS LIKE "
        status_metrics = [key for key, value in metrics_range.items() if value.get('view') == 'status']

        data_set_num_sql = "select count(*) from usertable"

        try:
            for metric_name in global_metrics:
                sql = global_sql + f"'{metric_name}'"
                result = self.query_all(sql)
                metric_value = int(result[0][1])
                metrics[metric_name] = metric_value

            for metric_name in status_metrics:
                sql = status_sql + f"'{metric_name}'"
                result = self.query_all(sql)
                metric_value = int(result[0][1])
                metrics[metric_name] = metric_value

            result = self.query_all(data_set_num_sql)
            metric_value = int(result[0][0])
            metrics['data_set_num'] = metric_value

            return metrics
        except Exception as e:
            print("error" + str(e))
            return None

    def update_config_file(self, param, value, config_file_path='/etc/mysql/mysql.conf.d/mysqld_template.cnf'):
        # 读取原始配置文件
        with open(config_file_path, 'r') as file:
            lines = file.readlines()

        # 配置参数是否已存在于文件中
        param_exists = False
        param_pattern = re.compile(r'^' + re.escape(param) + r'\s*=.*$', re.IGNORECASE)

        # 更新或添加配置参数
        for i, line in enumerate(lines):
            if param_pattern.match(line.strip()):
                lines[i] = f"{param} = {value}\n"
                param_exists = True
                break

        if not param_exists:
            # 如果参数不存在，则添加到文件末尾
            lines.append(f"{param} = {value}\n")

        # 将更改写入临时文件
        temp_file_path = '/tmp/mysqld_template.cnf'
        with open(temp_file_path, 'w') as file:
            file.writelines(lines)

        # 使用 sudo 权限移动文件
        move_cmd = ['sudo', 'mv', temp_file_path, config_file_path]
        password = "85581238".encode()
        subprocess.run(move_cmd, check=True, input=password)
        return True