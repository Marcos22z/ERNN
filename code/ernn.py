import os
import math
import random
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter
from torch import Tensor
from torch.autograd import Variable
from typing import Tuple
from sklearn.metrics import *

cuda = True if torch.cuda.is_available() else False

class RLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, state_table):
        super(RLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_ii = Parameter(Tensor(hidden_size, input_size))
        self.w_hi = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(Tensor(hidden_size, 1))
        self.b_hi = Parameter(Tensor(hidden_size, 1))

        self.w_if = Parameter(Tensor(hidden_size, input_size))
        self.w_hf = Parameter(Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(Tensor(hidden_size, 1))
        self.b_hf = Parameter(Tensor(hidden_size, 1))

        self.w_io = Parameter(Tensor(hidden_size, input_size))
        self.w_ho = Parameter(Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(Tensor(hidden_size, 1))
        self.b_ho = Parameter(Tensor(hidden_size, 1))

        self.w_ic = Parameter(Tensor(hidden_size, input_size))
        self.w_hc = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ic = Parameter(Tensor(hidden_size, 1))
        self.b_hc = Parameter(Tensor(hidden_size, 1))

        self.fc = nn.Linear(hidden_size, output_size, bias=False)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.state_tatble = state_table
        self.MTU = 1500

        self.reset_parameters()

    def forward(self, inputs: Tensor, state: Tuple[Tensor] = None, train: bool = True):
        if state is None:
            h_t = torch.zeros(1, self.hidden_size).t()
            c_t = torch.zeros(1, self.hidden_size).t()
        else:
            (h, c) = state
            h_t = h.squeeze(0).t()
            c_t = c.squeeze(0).t()

        if cuda:
            h_t = h_t.cuda()
            c_t = c_t.cuda()

        hidden_seq = [h_t]
        seq_size = inputs.size(1)

        # 0: normal 1: loss 2: repeat 3: out-of-order
        st = 0

        for t in range(seq_size):
            normal_max = self.state_tatble[st][0] + 0
            loss_max = self.state_tatble[st][1] + normal_max
            ......

        out = hidden_seq[-1].squeeze()
        ......

        return out