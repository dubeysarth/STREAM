from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.nn import ChebConv


class MLPEmbedding(nn.Module):
    def __init__(self, dim_in: int, hidden_dims: list[int], dim_out: int, dropout: float) -> None:
        super().__init__()
        dims = [dim_in] + hidden_dims + [dim_out]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers[:-1]) if layers else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalGConvLSTMCell(nn.Module):
    """A compact local fallback for graph-convolutional LSTM decoding."""

    def __init__(self, in_channels: int, hidden_channels: int, k: int = 2) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.k = k
        self.conv_x_i = ChebConv(in_channels, hidden_channels, K=k)
        self.conv_h_i = ChebConv(hidden_channels, hidden_channels, K=k)
        self.conv_x_f = ChebConv(in_channels, hidden_channels, K=k)
        self.conv_h_f = ChebConv(hidden_channels, hidden_channels, K=k)
        self.conv_x_g = ChebConv(in_channels, hidden_channels, K=k)
        self.conv_h_g = ChebConv(hidden_channels, hidden_channels, K=k)
        self.conv_x_o = ChebConv(in_channels, hidden_channels, K=k)
        self.conv_h_o = ChebConv(hidden_channels, hidden_channels, K=k)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        h_prev: torch.Tensor | None = None,
        c_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_channels, device=x.device, dtype=x.dtype)
        if c_prev is None:
            c_prev = torch.zeros_like(h_prev)
        i = torch.sigmoid(
            self.conv_x_i(x, edge_index, edge_weight=edge_weight)
            + self.conv_h_i(h_prev, edge_index, edge_weight=edge_weight)
        )
        f = torch.sigmoid(
            self.conv_x_f(x, edge_index, edge_weight=edge_weight)
            + self.conv_h_f(h_prev, edge_index, edge_weight=edge_weight)
        )
        g = torch.tanh(
            self.conv_x_g(x, edge_index, edge_weight=edge_weight)
            + self.conv_h_g(h_prev, edge_index, edge_weight=edge_weight)
        )
        o = torch.sigmoid(
            self.conv_x_o(x, edge_index, edge_weight=edge_weight)
            + self.conv_h_o(h_prev, edge_index, edge_weight=edge_weight)
        )
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


class LumpedSeq2One(nn.Module):
    """Embedding -> encoder LSTM -> decoder LSTM -> next-day discharge."""

    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 32,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.dynamic_embedding = MLPEmbedding(dynamic_dim, [64], embedding_dim, dropout)
        self.static_embedding = MLPEmbedding(static_dim, [64], embedding_dim, dropout)
        self.encoder = nn.LSTM(input_size=embedding_dim * 2, hidden_size=hidden_dim, num_layers=1)
        self.decoder = nn.LSTM(input_size=embedding_dim * 2, hidden_size=hidden_dim, num_layers=1)
        self.discharge_head = nn.Linear(hidden_dim, 1)
        self.swi_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, dynamic_seq: torch.Tensor, static_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        # dynamic_seq: (batch, seq, dynamic_dim)
        batch, seq_len, _ = dynamic_seq.shape
        dyn = self.dynamic_embedding(dynamic_seq.reshape(batch * seq_len, -1)).reshape(batch, seq_len, -1)
        stat = self.static_embedding(static_vec).unsqueeze(1).repeat(1, seq_len, 1)
        enc_in = torch.cat([dyn, stat], dim=-1).transpose(0, 1)
        _, (h, c) = self.encoder(enc_in)
        dec_token = torch.cat([dyn[:, -1, :], self.static_embedding(static_vec)], dim=-1).unsqueeze(0)
        dec_out, _ = self.decoder(dec_token, (h, c))
        hidden = dec_out.squeeze(0)
        return {
            "discharge": self.discharge_head(hidden).squeeze(-1),
            "swi": self.swi_head(hidden).squeeze(-1),
        }


class HybridSeqGConvSeq2One(nn.Module):
    """Embedding/encoder LSTM followed by a 2-layer graph-conv decoder."""

    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 32,
        dropout: float = 0.25,
        k: int = 2,
    ) -> None:
        super().__init__()
        self.dynamic_embedding = MLPEmbedding(dynamic_dim, [64], embedding_dim, dropout)
        self.static_embedding = MLPEmbedding(static_dim, [64], embedding_dim, dropout)
        self.encoder = nn.LSTM(input_size=embedding_dim * 2, hidden_size=hidden_dim, num_layers=1)
        self.decoder1 = LocalGConvLSTMCell(embedding_dim * 2, hidden_dim, k=k)
        self.decoder2 = LocalGConvLSTMCell(hidden_dim, hidden_dim, k=k)
        self.discharge_head = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.swi_head = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        dynamic_seq: torch.Tensor,
        static_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # dynamic_seq: (seq, nodes, dynamic_dim)
        seq_len, num_nodes, _ = dynamic_seq.shape
        dyn = self.dynamic_embedding(dynamic_seq.reshape(seq_len * num_nodes, -1)).reshape(seq_len, num_nodes, -1)
        stat = self.static_embedding(static_feat)
        encoder_input = torch.cat([dyn, stat.unsqueeze(0).repeat(seq_len, 1, 1)], dim=-1)
        _, (h, c) = self.encoder(encoder_input)
        x_dec = encoder_input[-1]
        h1, c1 = self.decoder1(x_dec, edge_index, edge_weight, h.squeeze(0), c.squeeze(0))
        h2, _ = self.decoder2(h1, edge_index, edge_weight)
        decoder_state = torch.cat([h2, stat], dim=-1)
        return {
            "discharge": self.discharge_head(decoder_state).squeeze(-1),
            "swi": self.swi_head(decoder_state).squeeze(-1),
        }
