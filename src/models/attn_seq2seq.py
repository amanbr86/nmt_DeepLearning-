"""Model B: Seq2Seq + Bahdanau attention. Same encoder/sizes as Model A; only the decoder differs."""
import torch
import torch.nn as nn

from src.models.attention import BahdanauAttention
from src.models.seq2seq import Seq2Seq


class AttnDecoder(nn.Module):
    def __init__(self, vocab, emb, hid, n_layers, dropout, pad_id=0):
        super().__init__()
        enc_dim = 2 * hid
        self.emb = nn.Embedding(vocab, emb, padding_idx=pad_id)
        self.attention = BahdanauAttention(enc_dim, hid, hid)
        self.rnn = nn.LSTM(emb + enc_dim, hid, n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hid + enc_dim + emb, vocab)
        self.drop = nn.Dropout(dropout)

    def forward(self, tok, state, enc_out, mask):
        emb = self.drop(self.emb(tok))                                    # (B, E)
        context, attn = self.attention(state[0][-1], enc_out, mask)       # query = previous top hidden
        out, state = self.rnn(torch.cat([emb, context], dim=-1).unsqueeze(1), state)
        logits = self.fc_out(self.drop(torch.cat([out.squeeze(1), context, emb], dim=-1)))
        return logits, state, attn


class AttnSeq2Seq(Seq2Seq):
    name = "attention"

    def _build_decoder(self, *args):
        return AttnDecoder(*args)
