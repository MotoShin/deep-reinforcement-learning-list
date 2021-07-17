import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import *


class ActorCriticNetwork(nn.Module):
    def __init__(self, batch_size, outputs):
        self.nhid = 300
        self.batch_size = batch_size

        super(ActorCriticNetwork, self).__init__()

        # conv layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # common layer
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        
        # actor layer
        self.fc5_1 = nn.Linear(512, 128)
        self.policy_logits = nn.Linear(128, outputs)
        self.softmax = nn.Softmax(dim=1)

        # critic layer
        self.fc5_2 = nn.Linear(512, 128)
        self.layerLSTM = nn.LSTMCell(128, self.nhid, bias=True)
        self.hiddenLSTM = self._init_hidden(self.batch_size)
        self._init_lstmCellWeight()
        self.layerLinearPostLSTM = nn.Linear(self.nhid, 128)
        self.values = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))

        x1 = F.relu(self.fc5_1(x))
        logits = self.policy_logits(x1)
        action_probs = torch.distributions.categorical.Categorical(self.softmax(logits))

        x2 = F.relu(self.fc5_2(x))
        self.hiddenLSTM = self.layerLSTM(x2, self.hiddenLSTM)
        x2 = F.relu(self.layerLinearPostLSTM(self.hiddenLSTM[0]))
        values = self.values(x2)

        return values, action_probs

    def learning(self, x):
        self.hiddenLSTM = self._init_hidden(AGENTS_NUM * TRAJECTORY_LENGTH)
        values, action_probs = self.forward(x)
        self.hiddenLSTM = self._init_hidden(self.batch_size)
        return values, action_probs

    def _init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(bsz, 300).to(DEVICE),
                weight.new_zeros(bsz, 300).to(DEVICE))

    def _init_lstmCellWeight(self):
        # the xavier initialization here is different than the one at the original code of critic_cartpole.
		# taken from: https://discuss.pytorch.org/t/initializing-parameters-of-a-multi-layer-lstm/5791
        for name, param in self.layerLSTM.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 1.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
