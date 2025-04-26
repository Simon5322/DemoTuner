import torch


class BaseOptimizer:
    def __init__(self):
        self.set_optimizer()

    def set_optimizer(self):
        pass

    def step(self):
        pass


