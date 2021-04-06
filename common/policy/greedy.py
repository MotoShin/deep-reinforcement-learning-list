import torch


class Greedy(object):
    @staticmethod
    def select(lst: torch.Tensor):
        if len(lst) == 1:
            # select action
            return lst.max(1)[1].view(1, 1).item()
        
        # replay buffer
        return torch.max(lst, 1)[1].unsqueeze(1)
