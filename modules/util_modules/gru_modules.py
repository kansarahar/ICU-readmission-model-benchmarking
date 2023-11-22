import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
import numpy as np

from .ode_modules import ODENet

def abs_time_to_delta(times):
    '''
    For each value in a given time sequence, return the difference between itself and its adjacent value
    '''
    delta = torch.cat((torch.unsqueeze(times[:, 0], dim=-1), times[:, 1:] - times[:, :-1]), dim=1)
    delta = torch.clamp(delta, min=0)
    return delta

class GRUExponentialDecay(nn.Module):
    '''
    GRU RNN where the hidden state decays exponentially

    Inputs:
        input - a tensor with embeddings in the last dimension
        times - a tensor with the same shape as input containing the recorded times (but no embedding dimension).

    Ouputs:
        output - hidden states of the RNN
    '''

    def __init__(self, input_size: int, hidden_size: int, bias=True, device=None):
        super(GRUExponentialDecay, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.decays = nn.Parameter(torch.tensor(hidden_size, dtype=torch.float32))

    def forward(self, inputs, times):
        hn = torch.zeros(inputs.size(0), self.hidden_size).to(self.device)
        for seq in range(inputs.size(1)):
            hn = self.gru_cell(inputs[:, seq, :], hn)
            hn = hn*torch.exp(-torch.clamp(torch.unsqueeze(times[:, seq], dim=-1) * self.decays, min=0))
        return hn


class GRUODEDecay(nn.Module):
    '''
    Inputs:
        input - a tensor with embeddings in the last dimension
        times - a tensor with the same shape as inputs, minus the last embedding dim, containing the recorded times

    Outputs:
        output - hidden state of the RNN
    '''

    def __init__(self, input_size: int, hidden_size: int, bias=True):
        super(GRUODEDecay, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.decays = nn.Parameter(torch.Tensor(hidden_size))
        self.odenet = ODENet(self.input_size, self.input_size)
    def forward(self, input, times):
        hn = torch.zeros(input.size(0), self.hidden_size) # batch_size x hidden_size
        out = torch.zeros(input.size(0), input.size(1), self.hidden_size) # batch_size x seq_len x hidden_size

        for seq in range(input.size(1)):
            hn = self.gru_cell(input[:, seq, :], hn)
            out[:, seq, :] = hn

            times_unique, inverse_indices = torch.unique(times[:, seq], sorted=True, return_inverse=True)
            if times_unique.size(0) > 1:
                hn = self.odenet(hn, times_unique)
                hn = hn[inverse_indices, torch.arange(0, inverse_indices.size(0)), :]
        return out