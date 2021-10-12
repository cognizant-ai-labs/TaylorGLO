# Copyright (c) 2019 Cognizant Digital Business.
# Issued under this Academic Public License: github.com/leaf-ai/muir/layers/LICENSE.
#
# Hyper LSTMs in pytorch
#

import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter
# from muir.hyper_utils import pack_dense

class HyperLSTM(nn.Module):

    def __init__(self, input_size, hidden_size,
                 context_size, block_in, block_out,
                 frozen_context=False,
                 num_layers=1):
        super(HyperLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.context_size = context_size
        self.block_in = block_in
        self.block_out = block_out

        # Create LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)

        # Compute number of context projectors needed
        self.projectors_per_param = {}
        self.num_projectors = 0
        for param in self.lstm._parameters:
            if 'weight' in param:
                direct_tensor = self.lstm._parameters[param]
                out_size, in_size = direct_tensor.size()
                assert (in_size % block_in == 0) and (out_size % block_out == 0)
                num_block_rows = out_size / block_out
                num_block_cols = in_size / block_in
                self.projectors_per_param[param] = int(num_block_rows * num_block_cols)
                self.num_projectors += num_block_rows * num_block_cols
        self.num_projectors = int(self.num_projectors)
        context_size = int(context_size)

        # Create context parameters
        self.context = Parameter(torch.Tensor(self.num_projectors, context_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        initial_context = 1. / math.sqrt(self.hidden_size)
        init.constant_(self.context, initial_context)

    # def set_weight(self, params):
    #     start = 0
    #     for param in self.lstm._parameters:
    #         if 'weight' in param:
    #             end = start + int(self.projectors_per_param[param])
    #             out_size, in_size = self.lstm._parameters[param].size()
    #             self.lstm._parameters[param] = pack_dense(params[start:end],
    #                                                       in_size, out_size)
    #             start = end

    def forward(self, input, hx=None):
        return self.lstm(input, hx)


if __name__ == '__main__':
    print(HyperLSTM(4, 8, 2, 2, 2))

