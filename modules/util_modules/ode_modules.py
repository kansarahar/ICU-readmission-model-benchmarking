import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
import numpy as np

MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver

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
        out = odeint(self.odefunc, x, eval_times, rtol=self.tol, atol=self.tol, method='euler')
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
