# Copyright (c) 2019 Cognizant Digital Business.
# Issued under this Academic Public License: github.com/leaf-ai/muir/pytorch/layers/LICENSE.
#
# Class for hyperlinear layers in pytorch
#

import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter
# from muir.hyper_utils import pack_dense

class HyperLinear(nn.Module):

    def __init__(self, in_features, out_features,
                 context_size, block_in, block_out,
                 frozen_context=False,
                 bias=True):
        super(HyperLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.context_size = context_size
        self.block_in = block_in
        self.block_out = block_out

        # Create context parameters.
        assert (in_features % block_in == 0) and (out_features % block_out == 0)
        self.num_block_rows = out_features // block_out
        self.num_block_cols = in_features // block_in
        self.num_blocks = int(self.num_block_rows * self.num_block_cols)
        self.num_projectors = self.num_blocks
        self.context = Parameter(torch.Tensor(self.num_blocks, context_size, 1))

        # Create bias vector.
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        initial_context = math.sqrt(2) / math.sqrt(self.in_features)
        init.constant_(self.context, initial_context)
        if self.bias is not None:
            init.zeros_(self.bias)

    # def set_weight(self, params):
    #     self.weight = pack_dense(params, self.in_features, self.out_features)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

if __name__ == '__main__':
    print(HyperLinear(10, 20, 3, 5, 4))

