
class expActionGuideOptimizer:
    def __init__(self, initial_lr):
        self.episode = None
        self.initial_lr = initial_lr

    def get_optimizer(self):
        final_lr = initial_lr / float(self.conf.lr_decrease_rate)
        if self.conf.action_guide:
            gamma = (final_lr / initial_lr) ** (1 / (self.conf.n_episode - self.conf.action_guide_episode))
        else:
            gamma = (final_lr / initial_lr) ** (1 / self.conf.n_episode)

        self.optimizer_actor = torch.optim.Adam(self.agent.actor_b.parameters(), lr=self.conf.lr_rate,
                                                weight_decay=self.conf.w_decay)
        self.actor_scheduler = ExponentialLR(self.optimizer_actor, gamma=gamma)

        self.optimizer_critic = torch.optim.Adam(self.agent.critic_b.parameters(), lr=self.conf.lr_rate,
                                                 weight_decay=self.conf.w_decay)
        self.critic_scheduler = ExponentialLR(self.optimizer_critic, gamma=gamma)