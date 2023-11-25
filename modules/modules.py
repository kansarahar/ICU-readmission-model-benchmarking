import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .util_modules.ode_modules import ODENet
from .util_modules.gru_modules import abs_time_to_delta, GRUExponentialDecay

# abstract class for all networks used
class Network(nn.Module):
    def __init__(self, num_static: int, num_dp_codes: int, num_cp_codes: int, dropout_probability=0.2, is_bidirectional=True, device=None):
        super(Network, self).__init__()

        self.device = device

        self.num_static = num_static
        self.num_dp_codes = num_dp_codes
        self.num_cp_codes = num_cp_codes

        self.dp_embedding_dim = int(num_dp_codes**0.25)
        self.cp_embedding_dim = int(num_cp_codes**0.25)
        self.embed_dp = nn.Embedding(num_embeddings=num_dp_codes, embedding_dim=self.dp_embedding_dim, padding_idx=0)
        self.embed_cp = nn.Embedding(num_embeddings=num_cp_codes, embedding_dim=self.cp_embedding_dim, padding_idx=0)

        self.dropout = nn.Dropout(p=dropout_probability)


    # abstract method, must be overridden
    def forward(self, static, dp, cp, dp_times, cp_times):
        pass



class ODE_RNN(Network):
    def __init__(self, num_static, num_dp_codes, num_cp_codes, dropout_probability=0.2, is_bidirectional=True, device=None):
        super(ODE_RNN, self).__init__(num_static, num_dp_codes, num_cp_codes, dropout_probability)

        self.is_bidirectional = is_bidirectional
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

        # Embedding for dp and cp
        embedded_dp = self.embed_dp(dp)
        embedded_cp = self.embed_cp(cp)


        # ODE for dp and cp
        dp_times = torch.round(dp_times, decimals=3)
        cp_times = torch.round(cp_times, decimals=3)

        dp_t_flat = dp_times.flatten()
        dp_t_flat_unique, inverse_indices = torch.unique(dp_t_flat, sorted=True, return_inverse=True)
        ode_dp_flat = self.ode_dp(dp_t_flat_unique, embedded_dp.view(-1, self.dp_embedding_dim))
        ode_dp_flat = ode_dp_flat[inverse_indices, torch.arange(0, inverse_indices.size(0)), :]
        ode_dp = ode_dp_flat.view(dp.size(0), dp.size(1), self.dp_embedding_dim)

        cp_t_flat = cp_times.flatten()
        cp_t_flat_unique, inverse_indices = torch.unique(cp_t_flat, sorted=True, return_inverse=True)
        ode_cp_flat = self.ode_cp(cp_t_flat_unique, embedded_cp.view(-1, self.cp_embedding_dim))
        ode_cp_flat = ode_cp_flat[inverse_indices, torch.arange(0, inverse_indices.size(0)), :]
        ode_cp = ode_cp_flat.view(cp.size(0), cp.size(1), self.cp_embedding_dim)


        # Dropout
        ode_dp = self.dropout(ode_dp)
        ode_cp = self.dropout(ode_cp)


        # RNN for dp and cp
        rnn_out_dp, rnn_hn_dp = self.gru_dp(ode_dp)
        rnn_out_cp, rnn_hn_cp = self.gru_cp(ode_cp)
        rnn_dp = torch.cat((rnn_hn_dp[0], rnn_hn_dp[1]), dim=-1) if self.is_bidirectional else torch.flatten(rnn_hn_dp[0]) # concatenate forward and backward passes
        rnn_cp = torch.cat((rnn_hn_cp[0], rnn_hn_cp[1]), dim=-1) if self.is_bidirectional else torch.flatten(rnn_hn_cp[0]) # concatenate forward and backward passes


        # FC layers
        dp_value = self.fc_dp(rnn_dp)
        cp_value = self.fc_cp(rnn_cp)
        combined = torch.cat((static, dp_value, cp_value), dim=-1)
        out = self.fc_final(combined)

        return out



class RNN_Exp_Decay(Network):
    def __init__(self, num_static, num_dp_codes, num_cp_codes, dropout_probability=0.2, is_bidirectional=True, device=None):
        super(RNN_Exp_Decay, self).__init__(num_static, num_dp_codes, num_cp_codes, dropout_probability)

        self.is_bidirectional = is_bidirectional
        rnn_output_dim = 2 if self.is_bidirectional else 1

        # RNN layers
        self.gru_dp_fw = GRUExponentialDecay(input_size=self.dp_embedding_dim, hidden_size=self.dp_embedding_dim, device=device)
        self.gru_dp_bw = GRUExponentialDecay(input_size=self.dp_embedding_dim, hidden_size=self.dp_embedding_dim, device=device)
        self.gru_cp_fw = GRUExponentialDecay(input_size=self.cp_embedding_dim, hidden_size=self.cp_embedding_dim, device=device)
        self.gru_cp_bw = GRUExponentialDecay(input_size=self.cp_embedding_dim, hidden_size=self.cp_embedding_dim, device=device)

        # FC layers
        self.fc_dp = nn.Linear(rnn_output_dim * self.dp_embedding_dim, 1)
        self.fc_cp = nn.Linear(rnn_output_dim * self.cp_embedding_dim, 1)
        self.fc_final = nn.Linear(num_static + 2, 1)

    def forward(self, static, dp, cp, dp_times, cp_times):

        # Embedding for dp and cp
        embedded_dp_fw = self.embed_dp(dp)
        embedded_dp_bw = torch.flip(embedded_dp_fw, [1])
        embedded_cp_fw = self.embed_cp(cp)
        embedded_cp_bw = torch.flip(embedded_cp_fw, [1])


        # GRU Exponential Decay
        dp_times = torch.round(dp_times, decimals=3)
        cp_times = torch.round(cp_times, decimals=3)

        dp_time_diffs_fw = abs_time_to_delta(dp_times)
        dp_time_diffs_bw = torch.flip(dp_time_diffs_fw, [1])
        cp_time_diffs_fw = abs_time_to_delta(cp_times)
        cp_time_diffs_bw = torch.flip(cp_time_diffs_fw, [1])

        rnn_hn_dp = torch.cat((self.gru_dp_fw(embedded_dp_fw, dp_time_diffs_fw), self.gru_dp_bw(embedded_dp_bw, dp_time_diffs_bw)), dim=-1) if self.is_bidirectional else self.gru_dp_fw(embedded_dp_fw, dp_time_diffs_fw)
        rnn_hn_cp = torch.cat((self.gru_cp_fw(embedded_cp_fw, cp_time_diffs_fw), self.gru_cp_bw(embedded_cp_bw, cp_time_diffs_bw)), dim=-1) if self.is_bidirectional else self.gru_cp_fw(embedded_cp_fw, cp_time_diffs_fw)


        # Dropout
        rnn_hn_dp = self.dropout(rnn_hn_dp)
        rnn_hn_cp = self.dropout(rnn_hn_cp)


        # FC layers
        dp_value = self.fc_dp(rnn_hn_dp)
        cp_value = self.fc_cp(rnn_hn_cp)
        combined = torch.cat((static, dp_value, cp_value), dim=-1)
        out = self.fc_final(combined)

        return out
