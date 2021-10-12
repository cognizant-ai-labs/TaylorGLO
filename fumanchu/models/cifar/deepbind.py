# Copyright (c) 2019 Cognizant Digital Business.
# Issued under this Academic Public License: github.com/leaf-ai/muir/pytorch/muir/LICENSE.
#
# Hypermodule implementation of deepbind model
#
# The deepbind model family was originally described in the following papers:
#
# B. Alipanahi, A. Delong, M. T. Weirauch, and B. J. Frey. Predicting the sequence
# specificities of dna-and rna-binding proteins by deep learning. Nature biotechnology, 2015.
#
# H. Zeng, M. D. Edwards, G. Liu, and D. K. Gifford. Convolutional neural network
# architectures for predicting dna-protein binding. Bioinformatics, 2016.
#

import torch.nn as nn
from torch.nn import functional as F
from layers.hyperconv1d import HyperConv1d
from layers.hyperlinear import HyperLinear

__all__ = ['deepbind']

class DeepBind(nn.Module):
    def __init__(self, context_size, block_in, block_out,
                 model_config, hyper=False,
                 filters=16, hidden_units=32):
        super(DeepBind, self).__init__()

        self.context_size = context_size
        self.block_in = block_in
        self.block_out = block_out
        self.hyper = hyper

        self.hyperlayers = []

        self.embedding = nn.Embedding(5, filters)

        self.conv1 = self.create_conv(filters, filters, 24)
        self.fc1 = self.create_fc(filters, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 1)

        self.num_projectors = sum([l.num_projectors for l in self.hyperlayers])

    def create_conv(self, in_channels, out_channels, kernel_size):
        if self.hyper:
            layer = HyperConv1d(in_channels, out_channels, kernel_size,
                                self.context_size, self.block_in, self.block_out)
            self.hyperlayers.append(layer)
        else:
            layer = nn.Conv1d(in_channels, out_channels, kernel_size)
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        return layer

    def create_fc(self, in_units, out_units):
        if self.hyper:
            layer = HyperLinear(in_units, out_units,
                                self.context_size, self.block_in, self.block_out)
            self.hyperlayers.append(layer)
        else:
            layer = nn.Linear(in_units, out_units)
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        return layer

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=x.size()[-1])
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

def deepbind(**kwargs):
    model = DeepBind(**kwargs)
    return model


if __name__ == '__main__':
    net = DeepBind(4, 16, 16, {'context_size': 20}, hyper=True, filters=16, hidden_units=32)
    print(net)
    print(net.num_projectors)

    net = DeepBind(0, 16, 16, {'context_size': 20}, hyper=False, filters=256, hidden_units=256)
    from model_utils import count_parameters
    print(count_parameters(net))