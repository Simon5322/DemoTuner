
import subprocess

import yaml

from dbms.generic_dbms import ConfigurableDBMS
import os
import psycopg2
import time
import psutil

from utils.prometheus import Prometheus


class PgConfig(ConfigurableDBMS):
    """ Reconfigurable Postgres DBMS instance. """

    def __init__(self, db, user, password, restart_cmd,
                 recovery_cmd, timeout_s):
        """ Initialize DB connection with given credentials. 
        
        Args:
            db: name of Postgres database
            user: name of database user
            password: database password
            restart_cmd: command for restarting server
            recovery_cmd: command for recovering DB
            timeout_s: per-query timeout in seconds
        """
        unit_to_size = {'KB': 'kB', 'MB': '000kB', 'GB': '000000kB',
                        'K': 'kB', 'M': '000kB', 'G': '000000kB'}
        super().__init__(db, user, password, unit_to_size,
                         restart_cmd, recovery_cmd, timeout_s)
        self.dbms_name = 'pg'
    @classmethod
    def from_file(cls, config):
        """ Initializes PgConfig object from configuration file. 
        
        Args:
            cls: class (currently, only PgConfig)
            config: configuration read from file
            
        Returns:
            new Postgres DBMS object
        """
        db_user = config['DATABASE']['user']
        db_name = config['DATABASE']['database_name']
        password = config['DATABASE']['password']
        restart_cmd = config['DATABASE']['restart_cmd']
        recovery_cmd = config['DATABASE']['recovery_cmd']
        timeout_s = config['LEARNING']['timeout_s']

        # print(db_user)
        # print(db_name)
        # print(restart_cmd)
        # print(recovery_cmd)
        # print(timeout_s)
        return cls(db_name, db_user, password,
                   restart_cmd, recovery_cmd, timeout_s)

    def __del__(self):
        """ Close DBMS connection if any. """
        super().__del__()

    def copy_db(self, source_db, target_db):
        """ Copy source to target database. """
        self.update(f'drop database if exists {target_db}')
        self.update(f'create database {target_db} with template {source_db}')

    def _connect(self):
        """ Establish connection to database, returns success flag. 
        
        Returns:
            True iff connection to Postgres was established
        """
        print(f'Trying to connect to {self.db} with user {self.user}')
        # Need to recover in case of bad configuration
        try:
            self.connection = psycopg2.connect(
                database=self.db, user=self.user,
                password=self.password)  # , host = "localhost")
            # cur = self.connection.cursor()
            # # Execute a query
            # cur.execute("SELECT * FROM company")
            # records = cur.fetchall()
            # print(records)
            self.set_timeout(self.timeout_s)
            self.failed_connections = 0
            return True
        except Exception as e:
            # Delete changes to default configuration and restart
            print(f'Exception while trying to connect: {e}')
            self.failed_connections += 1
            print(f'Had {self.failed_connections} failed connections.')
            if self.failed_connections < 3:
                print(f'Trying recovery with "{self.recovery_cmd}" ...')
                os.system(self.recovery_cmd)
                self.reconfigure()
            return False

    def _disconnect(self):
        """ Disconnect from database. """
        if self.connection:
            print('Disconnecting ...')
            self.connection.close()

    def all_params(self):
        """ Return names of all tuning parameters. """
        all_params_sql = "select name from pg_settings " \
                         "where vartype in ('bool', 'integer', 'real')"
        cursor = self.connection.cursor()
        cursor.execute(all_params_sql)
        var_vals = cursor.fetchall()
        cursor.close()
        return [v[0] for v in var_vals]

    def exec_file(self, path):
        """ Executes all SQL queries in given file and returns error flag. """
        error = True
        try:
            with open(path) as file:
                sql = file.read()
                self.connection.autocommit = True
                cursor = self.connection.cursor()
                cursor.execute(sql)
                cursor.close()
            error = False
        except Exception as e:
            print(f'Exception execution {path}: {e}')
        return error

    def query_one(self, sql):
        """ Executes query_one and returns first result table cell or None. """
        try:
            self.connection.autocommit = True
            cursor = self.connection.cursor()
            cursor.execute(sql)
            return cursor.fetchone()[0]
        except Exception as e:
            print("error" + str(e))
            return None

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

    def update(self, sql):
        """ Executes update and returns true iff the update succeeds. """
        try:
            self.connection.autocommit = True
            cursor = self.connection.cursor()
            cursor.execute(sql)
            return True
        except Exception:
            return False

    def is_param(self, param):
        """ Returns True iff given parameter exists. """
        return self.can_query(f'show {param}')

    def get_value(self, param):
        """ Get current value of given parameter. """
        return self.query_one(f'show {param}')

    def check_conf_set(self, conf_name, conf_value):
        real_conf_value = self.get_value(conf_name)
        print(f'conf_name : {conf_name}, new_value : {conf_value}, real_value : {real_conf_value}')
        # if real_conf_value != conf_value:
        #     raise ValueError(f'{conf_name} 应被赋值为{conf_value},而真实值是{real_conf_value}')
        # else:
        #     return True

    def create_new_usertable(self, fieldcount):
        self.execute('DROP TABLE IF EXISTS usertable;')
        sql = """
        CREATE TABLE usertable (
            YCSB_KEY VARCHAR(255) PRIMARY KEY
            );
        DO $$ 
        DECLARE 
            i INT := 0;
        BEGIN
            FOR i IN 0..({}-1) LOOP
                EXECUTE 'ALTER TABLE usertable ADD COLUMN FIELD' || i || ' TEXT';
            END LOOP;
        END $$;
        """.format(fieldcount)
        self.execute(sql)

    def delete_from_usertable(self):
        return self.query_one(f'DELETE FROM usertable')

    def get_metrics(self):
        metrics = {}
        script_directory = os.path.dirname(os.path.abspath(__file__))
        # 构建相对路径
        metrics_path = os.path.join(script_directory, '../tuning_params_states/pg/pg_metrics.yml')
        with open(metrics_path, "r") as f:
            metrics_range = yaml.load(f, Loader=yaml.FullLoader)
        metrics_keys = list(metrics_range.keys())

        # 系统指标
        prometheus = Prometheus()
        cpu_usage = prometheus.query('100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)')[0]['value'][1]
        memery_usage = prometheus.query('100 * (1 - node_memory_MemFree_bytes / node_memory_MemTotal_bytes)')[0]['value'][1]
        write_speed = prometheus.query('irate(node_disk_written_bytes_total{job="node", device="sda"}[30s])')[0]['value'][1]
        io_time = prometheus.query('irate(node_disk_io_time_seconds_total{job="node", device="sda"}[30s])')[0]['value'][1]
        load1 = prometheus.query('node_load1')[0]['value'][1]
        dirty_byte= prometheus.query('node_memory_Dirty_bytes')[0]['value'][1]
        # buffers_backend= prometheus.query('pg_stat_bgwriter_buffers_backend_total')[0]['value'][1]
        # buffers_checkpoint = prometheus.query('pg_stat_bgwriter_buffers_checkpoint_total')[0]['value'][1]

        metrics['cpu_changed'] = float(cpu_usage)
        metrics['cpu_usage'] = float(cpu_usage)
        metrics['mem_changed'] = float(memery_usage)
        metrics['mem_usage'] = float(memery_usage)
        metrics['write_speed_changed'] = float(write_speed)
        metrics['io_latency_changed'] = float(io_time)
        metrics['load1'] = float(load1)
        metrics['node_memory_Dirty_bytes'] = int(dirty_byte)
        # metrics['buffers_backend'] = int(buffers_backend)
        # metrics['buffers_checkpoint'] = int(buffers_checkpoint)

        # pg 指标
        view_name = 'pg_stat_database'
        database_metrics = [key for key, value in metrics_range.items() if value.get('view') == view_name]
        database_statement = f"SELECT {', '.join(database_metrics)} FROM {metrics_range[database_metrics[0]]['view']} WHERE datname='test';"

        view_name = 'pg_stat_bgwriter'
        bgwriteer_metrics = [key for key, value in metrics_range.items() if value.get('view') == view_name]
        bgwriter_statement = f"SELECT {', '.join(bgwriteer_metrics)} FROM {metrics_range[bgwriteer_metrics[0]]['view']};" if len(
            bgwriteer_metrics) > 1 else f"SELECT {bgwriteer_metrics[0]} FROM {metrics_range[bgwriteer_metrics[0]]['view']};"

        view_name = 'pg_stat_wal'
        wal_metrics = [key for key, value in metrics_range.items() if value.get('view') == view_name]
        wal_statement = f"SELECT {', '.join(wal_metrics)} FROM {metrics_range[wal_metrics[0]]['view']};" if len(
            wal_metrics) > 1 else f"SELECT {wal_metrics[0]} FROM {metrics_range[wal_metrics[0]]['view']};"

        view_name = 'pg_stat_user_tables'
        table_metrics = [key for key, value in metrics_range.items() if value.get('view') == view_name]
        table_statement = f"SELECT {', '.join(table_metrics)} FROM {metrics_range[table_metrics[0]]['view']};" if len(
            table_metrics) > 1 else f"SELECT {table_metrics[0]} FROM {metrics_range[table_metrics[0]]['view']};"
        try:
            self.connection.autocommit = True
            cursor = self.connection.cursor()

            cursor.execute(database_statement)
            result_database = cursor.fetchall()
            for key, result in zip(database_metrics, result_database[0]):
                metrics[key] = result

            cursor.execute(wal_statement)
            result_wal_records = cursor.fetchall()
            for key, result in zip(wal_metrics, result_wal_records[0]):
                metrics[key] = result

            cursor.execute(table_statement)
            result_table = cursor.fetchall()
            for key, result in zip(table_metrics, result_table[0]):
                metrics[key] = result

            cursor.execute(bgwriter_statement)
            result_bgwriter = cursor.fetchall()
            for key, result in zip(bgwriteer_metrics, result_bgwriter[0]):
                metrics[key] = result

            # cursor.execute(pg_stat_activity_sql)
            # pg_stat_activity = cursor.fetchone()[0]
            # metrics["pg_stat_activity"] = pg_stat_activity

            return metrics
        except Exception as e:
            print("error" + str(e))
            return None

    def set_param(self, param, value):
        """ Set given parameter to given value. """
        self.config[param] = value
        query_one = f'alter system set {param} to \'{value}\''
        try:
            result = self.update(query_one)
            return result
        except Exception as e:
            print(f"update params error:{e}")
            return None

    def set_timeout(self, timeout_s):
        """ Set per-query timeout. """
        self.update(f"set statement_timeout = '{timeout_s}s'")

    def reset_config(self):
        """ Reset all parameters to default values. """
        self.update('alter system reset all')
        self.config = {}

    def restart_dbms(self):
        pwd = "85581238"
        stop_cmd = "sudo service postgresql stop"
        stop_result = subprocess.run(stop_cmd, shell=True, input="85581238\n", text=True, check=True)
        if stop_result.returncode == 0:
            print("PostgreSQL stopped successfully.")
        else:
            print("Error stopping PostgreSQL:")
            print(stop_result.stderr)

        # 启动 PostgreSQL
        start_cmd = "sudo service postgresql start"
        start_result = subprocess.run(start_cmd, shell=True, input="85581238\n", text=True, check=True)
        if start_result.returncode == 0:
            print("PostgreSQL started successfully.")
        else:
            print("Error starting PostgreSQL:")
            print(start_result.stderr)

    def make_conf_effect(self, restart=False):
        """ Get current value of given parameter. """
        return self.query_one(f'SELECT pg_reload_conf()')

    def reconfigure(self):
        """ Makes parameter settings take effect. Returns true if successful.
        
        Returns:
            True iff reconfiguration was successful
        """
        self.make_conf_effect()
        self._disconnect()
        os.system(self.restart_cmd)
        time.sleep(3)
        success = self._connect()
        return success

