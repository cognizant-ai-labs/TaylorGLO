import torch
import torch.nn as nn

class LossferatuActivation(torch.nn.Module):
    

    def __init__(self, activation_str):
        super(LossferatuActivation, self).__init__()
        self.activation_str = activation_str

    def forward(self, x):
        tf = torch
        return eval(self.activation_str, globals(), locals())
