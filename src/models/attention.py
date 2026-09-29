"""Bahdanau (additive) attention with padding mask."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query, enc_out, mask):
        # query: (B, dec_dim) | enc_out: (B, S, enc_dim) | mask: (B, S) True = real token
        energy = self.v(torch.tanh(self.W_enc(enc_out) + self.W_dec(query).unsqueeze(1))).squeeze(-1)
        energy = energy.masked_fill(~mask, -1e9)
        weights = F.softmax(energy, dim=-1)                      # (B, S)
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)   # (B, enc_dim)
        return context, weights
