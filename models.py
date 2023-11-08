import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules import Attention, GRUExponentialDecay, GRUODEDecay, ODENet

# abstract class for all networks used
class Network(nn.Module):
    def __init__(self, num_static: int, num_dp_codes: int, num_cp_codes: int):
        super(Network, self).__init__()

        self.dp_embedding_dim = int(num_dp_codes**0.25)
        self.cp_embedding_dim = int(num_cp_codes**0.25)
        self.embed_dp = nn.Embedding(num_embeddings=num_dp_codes, embedding_dim=self.dp_embedding_dim, padding_idx=0)
        self.embed_cp = nn.Embedding(num_embeddings=num_cp_codes, embedding_dim=self.cp_embedding_dim, padding_idx=0)

        self.static_dim = num_static

    # abstract method, must be overridden
    def forward(self, static, dp, cp, dp_times, cp_times):
        pass


class ODE_RNN(Network):
    def __init__(self, num_static, num_dp_codes, num_cp_codes):
        super(ODE_RNN, self).__init__(num_static, num_dp_codes, num_cp_codes)

        self.is_bidirectional = False
        rnn_output_dim = 2 if self.is_bidirectional else 1

        # ODE layers
        self.ode_dp = ODENet(input_dim=self.dp_embedding_dim, hidden_dim=self.dp_embedding_dim)
        self.ode_cp = ODENet(input_dim=self.cp_embedding_dim, hidden_dim=self.cp_embedding_dim)

        # RNN layers
        self.gru_dp = nn.GRU(input_size=self.dp_embedding_dim, hidden_size=self.dp_embedding_dim, num_layers=1, bidirectional=self.is_bidirectional, batch_first=True)
        self.gru_cp = nn.GRU(input_size=self.cp_embedding_dim, hidden_size=self.cp_embedding_dim, num_layers=1, bidirectional=self.is_bidirectional, batch_first=True)

        # FC layers
        self.fc_dp = nn.Linear(rnn_output_dim * self.dp_embedding_dim, 1)
        self.fc_cp = nn.Linear(rnn_output_dim * self.cp_embedding_dim, 1)
        self.fc_final = nn.Linear(num_static + 2, 1)

    def forward(self, static, dp, cp, dp_times, cp_times):
        print(self.dp_embedding_dim, type(self.dp_embedding_dim))

        # Embedding for dp and cp
        embedded_dp = self.embed_dp(dp)
        embedded_cp = self.embed_cp(cp)

        # RNN for dp and cp
        rnn_out_dp, rnn_hn_dp = self.gru_dp(embedded_dp)
        rnn_out_cp, rnn_hn_cp = self.gru_cp(embedded_cp)
        rnn_dp = torch.cat((rnn_hn_dp[0], rnn_hn_dp[1]), dim=-1) if self.is_bidirectional else torch.flatten(rnn_hn_dp[0]) # concatenate forward and backward passes
        rnn_cp = torch.cat((rnn_hn_cp[0], rnn_hn_cp[1]), dim=-1) if self.is_bidirectional else torch.flatten(rnn_hn_cp[0]) # concatenate forward and backward passes

        # FC layers
        dp_value = self.fc_dp(rnn_dp)
        cp_value = self.fc_cp(rnn_cp)
        combined = torch.cat((static, dp_value, cp_value), dim=-1)
        out = self.fc_final(combined)

        return out


if __name__ == '__main__':
    data = np.load('./data/preprocessed/data_arrays.npz', allow_pickle=True)
    static, dp, cp, dp_times, cp_times = data['static'].astype('float32'), data['dp'], data['cp'], data['dp_times'], data['cp_times']
    num_static, num_dp_codes, num_cp_codes = len(static[0]), dp.max() + 1, cp[0].max() + 1
    ode_rnn = ODE_RNN(num_static, num_dp_codes, num_cp_codes)
    ode_rnn.forward(torch.tensor(static[0], dtype=torch.float32), torch.tensor(dp[0], dtype=torch.int64), torch.tensor(cp[0], dtype=torch.int64), torch.tensor(dp_times[0], dtype=torch.float32), torch.tensor(cp_times[0], dtype=torch.float32))










