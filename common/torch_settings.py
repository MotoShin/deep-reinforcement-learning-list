import torch
import torch.autograd as autograd

DTYPE = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if torch.cuda.is_available():
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)
