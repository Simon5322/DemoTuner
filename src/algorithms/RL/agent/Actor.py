from torch import nn


class Actor(nn.Module):
    """
    n_state:state的维度
    action_dim:action的维度
    hidden1: 每一层的单元数
    """

    def __init__(self, n_states, action_dim, hidden1):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, 1)
        )

    def forward(self, state):
        return self.net(state)
