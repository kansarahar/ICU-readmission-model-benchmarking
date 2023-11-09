import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
import numpy as np

MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver

class Attention(torch.nn.Module):
    '''
    Dot-product attention module

    Inputs:
        input - a tensor with embeddings in the last dimension
        mask - a tensor with the same dim as "input" exept for the embedding dimension.
            values are 0 for 0-padding in the input, else 1

    Outputs:
        output - the input tensor whose embeddings in the last dimensions have undergone a weighted average
            the second-to-last dimension of the tensor is removed
        weights - attention weights given to each embedding
    '''

    def __init__(self, embedding_dim: int):
        super(Attention, self).__init__()
        self.context = nn.Parameter(torch.Tensor(embedding_dim)) # will be a vector of size embedding_dim
        self.hidden = nn.Linear(embedding_dim, embedding_dim)
        nn.init.normal_(self.context) # initialize values according to normal dist

    def forward(self, input: torch.tensor, mask: torch.tensor):
        hidden = torch.tanh(self.hidden(input))
        importance = torch.sum(hidden * self.context, dim=-1)
        importance = importance.masked_fill(mask==0, -1e9)
        attention_weights = F.softmax(importance, dim=-1)
        weighted_projection = input * torch.unsqueeze(attention_weights, dim=-1)
        output = torch.sum(weighted_projection, dim=-2)
        return output, attention_weights



class ODEFunc(nn.Module):
    '''
    MLP for modeling the derivative of an ODE system

    t - tensor representing current time, shape (1,)
    x - shape (batch_size, input_dim)
    '''

    def __init__(self, input_dim: int, hidden_dim: int):
        super(ODEFunc, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # if it were time-dependent, we would have nn.Linear(self.input_dim + 1, hidden_dim) and concat x with t, but we're assuming it's not
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, t: torch.tensor, x: torch.tensor):
        return self.layers(x)

class ODEBlock(nn.Module):
    '''
    Solves the ODE defined by ODEFunc

    Inputs:
        x - tensor of shape (batch_size, ODEFunc.input_dim)
        eval_times - None or tensor
    Outputs:
        out - returns full ODE trajectory evaluated at points in eval_times
    '''

    def __init__(self, odefunc: ODEFunc, tol=1e-3):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, eval_times: torch.tensor, x: torch.tensor):
        out = odeint(self.odefunc, x, eval_times, rtol=self.tol, atol=self.tol, method='euler', options={'max_num_steps': MAX_NUM_STEPS})
        return out

class ODENet(nn.Module):
    '''
    An ODE Func and ODE Block

    Inputs:
        x - tensor of shape ODEFunc.input_dim
        eval_times - None or tensor
    Outputs:
        out - returns full ODE trajectory evaluated at points in eval_times
    '''
    def __init__(self, input_dim: int, hidden_dim: int):
        super(ODENet, self).__init__()
        self.odefunc = ODEFunc(input_dim, hidden_dim)
        self.odeblock = ODEBlock(self.odefunc)

    def forward(self, eval_times: torch.tensor, x: torch.tensor):       
        return self.odeblock(eval_times, x)



class GRUExponentialDecay(nn.Module):
    '''
    GRU RNN where the hidden state decays exponentially

    Inputs:
        input - a tensor with embeddings in the last dimension
        times - a tensor with the same shape as input containing the recorded times (but no embedding dimension).

    Ouputs:
        output - hidden states of the RNN
    '''

    def __init__(self, input_size: int, hidden_size: int, bias=True):
        super(GRUExponentialDecay, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.decays = nn.Parameter(torch.tensor(hidden_size))

    def forward(self, inputs, times):
        hn = torch.zeros(inputs.size(0), self.hidden_size)
        out = torch.zeros(inputs.size(0), inputs.size(1), self.hidden_size)
        for seq in range(inputs.size(1)):
            hn = self.gru_cell(inputs[:, seq, :], hn)
            out[:, seq, :] = hn
            hn = hn*torch.exp(-torch.clamp(torch.unsqueeze(times[:, seq], dim=-1) * self.decays, min=0))
        return out


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