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
