from torch import optim


# 使用线性增加学习率的自定义优化器
class LrFinderOptimizer(optim.Adam):
    def __init__(self, params, initial_lr, max_lr, num_episode):
        super().__init__(params, lr=initial_lr)
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.num_episode = num_episode
        self.iteration = 0

    def step(self, closure=None):
        self.iteration += 1
        lr = self.initial_lr + (self.max_lr - self.initial_lr) * (self.iteration / self.num_episode)
        for param_group in self.param_groups:
            param_group['lr'] = lr
        super().step(closure)


