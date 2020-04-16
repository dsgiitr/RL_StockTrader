import torch.nn as nn
    
# Network Architecture
class Q_Network(nn.Module):

    def __init__(self, obs_len, hidden_size_1, hidden_size_2, actions_n):
        super(Q_Network, self).__init__()

        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, actions_n)
            )

    def forward(self, x):
        h = self.fc_val(x)
        return h