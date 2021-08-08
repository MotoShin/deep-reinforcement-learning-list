import sys
import os
import torch

sys.path.append(os.path.abspath("../.."))
from a2c.network import ActorCriticNetwork
from environment.cartpole import CartPole


input = torch.randn(1, 4, 84, 84)
network = ActorCriticNetwork(2)
value, probs = network(input)
print(value)
print(probs)
