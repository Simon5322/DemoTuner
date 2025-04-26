import time

from prometheus_api_client import PrometheusConnect


class Prometheus:
    def __init__(self):
        prometheus_url = "http://localhost:9090"
        self.prometheus_conn = PrometheusConnect(url=prometheus_url)
        self.retry_cnt = 0

    def query(self, query_string):
        try:
            result = self.prometheus_conn.custom_query(query=query_string)
            # 检查result是否为空，如果是则抛出异常
            if not result:
                raise Exception("Empty result received")
        except Exception as e:
            self.retry_cnt += 1
            print(f"{str(e)}")
            if self.retry_cnt < 5:
                print(f"retrying {str(self.retry_cnt)} times")
                time.sleep(1)
                result = self.query(query_string)
            else:
                raise
        self.retry_cnt = 0
        return result


# 查询内存使用率的指标
if __name__ == '__main__':
    prometheus = Prometheus()
    while(True):
        result = prometheus.query('pg_stat_bgwriter_buffers_backend_total')[0]['value'][1]
        print(result)
        time.sleep(1)
    #write_speed = prometheus.query('irate(node_disk_written_bytes_total{job="node", device="sda"}[30s])')[0]['value'][1]

    # memory_query = '100 * (1 - node_memory_MemFree_bytes / node_memory_MemTotal_bytes)'
    # result = []
    # for i in range(5):
    #     result.append(prometheus.query(memory_query))
    #     time.sleep(1)
    #
    # # 打印结果
    # for r in result:
    #     print(r)




