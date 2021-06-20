import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ActorCriticNetwork(nn.Module):
    def __init__(self, outputs):
        super(ActorCriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5_1 = nn.Linear(512, 128)
        self.fc5_2 = nn.Linear(512, 128)
        self.policy_logits = nn.Linear(128, outputs)
        self.sotmax = nn.Softmax(dim=1)
        self.values = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))

        x1 = F.relu(self.fc5_1(x))
        logits = self.policy_logits(x1)
        action_probs = torch.distributions.categorical.Categorical(self.sotmax(logits))

        x2 = F.relu(self.fc5_2(x))
        values = self.values(x2)

        return values, action_probs

    def parallel_input(self, inps):
        values = []
        action_probs = []
        for inp in inps:
            value, action_prob = self.forward(inp)
            values.append(value)
            action_probs.append(action_prob)
        
        return values, action_probs
