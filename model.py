from torch import nn 
import torch

class MLP(nn.Module):
    def __init__(self, in_n, hidden_n, hidden_activation, out_n, out_activation):
        super().__init__()
        self.linear1 = nn.Linear(in_n, hidden_n)
        self.act1 = nn.__getattribute__(hidden_activation)()

        self.linear2 = nn.Linear(hidden_n, out_n)
        self.act2 = nn.__getattribute__(out_activation)()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        return x