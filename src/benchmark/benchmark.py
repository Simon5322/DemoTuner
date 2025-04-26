from abc import ABC, abstractmethod


class benchmark(ABC):
    @abstractmethod
    def __init__(self, benchmark_name):
        self.benchmark_name = benchmark_name

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def run_benchmark(self, times):
        pass

    @abstractmethod
    def get_result(self, times):
        pass

    def set_default_global_conf(self):
        pass
