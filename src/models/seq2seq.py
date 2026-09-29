"""Model A: basic Seq2Seq (BiLSTM encoder -> LSTM decoder, NO attention).
The decoder only sees the encoder's final (h, c) states."""
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, vocab, emb, hid, n_layers, dropout, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=pad_id)
        self.rnn = nn.LSTM(emb, hid, n_layers, batch_first=True, bidirectional=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.fc_h, self.fc_c = nn.Linear(2 * hid, hid), nn.Linear(2 * hid, hid)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _merge(state, fc):
        # (n_layers*2, B, H) -> (n_layers, B, 2H) -> project to (n_layers, B, H)
        n = state.size(0) // 2
        state = state.view(n, 2, state.size(1), state.size(2))
        return torch.tanh(fc(torch.cat([state[:, 0], state[:, 1]], dim=-1)))

    def forward(self, src, src_len):
        x = self.drop(self.emb(src))
        packed = pack_padded_sequence(x, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        return out, (self._merge(h, self.fc_h), self._merge(c, self.fc_c))   # out: (B, S, 2H)


class Decoder(nn.Module):
    def __init__(self, vocab, emb, hid, n_layers, dropout, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=pad_id)
        self.rnn = nn.LSTM(emb, hid, n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hid, vocab)
        self.drop = nn.Dropout(dropout)

    def forward(self, tok, state, enc_out=None, mask=None):
        x = self.drop(self.emb(tok)).unsqueeze(1)             # (B, 1, E)
        out, state = self.rnn(x, state)
        return self.fc_out(self.drop(out.squeeze(1))), state, None   # no attention weights


class Seq2Seq(nn.Module):
    name = "seq2seq"

    def __init__(self, src_vocab, trg_vocab, emb_dim, hidden_dim, n_layers, dropout, pad_id=0):
        super().__init__()
        self.pad_id, self.trg_vocab = pad_id, trg_vocab
        self.encoder = Encoder(src_vocab, emb_dim, hidden_dim, n_layers, dropout, pad_id)
        self.decoder = self._build_decoder(trg_vocab, emb_dim, hidden_dim, n_layers, dropout, pad_id)

    def _build_decoder(self, *args):
        return Decoder(*args)

    # ---- generic interface used by training, greedy and beam search ----
    def encode(self, src, src_len):
        enc_out, state = self.encoder(src, src_len)
        return enc_out, (src != self.pad_id), state

    def decode_step(self, tok, state, enc_out, mask):
        return self.decoder(tok, state, enc_out, mask)          # logits, state, attn|None

    def forward(self, src, src_len, trg, tf_ratio=0.5):
        enc_out, mask, state = self.encode(src, src_len)
        tok, outs = trg[:, 0], []
        for t in range(1, trg.size(1)):
            logits, state, _ = self.decode_step(tok, state, enc_out, mask)
            outs.append(logits)
            tok = trg[:, t] if random.random() < tf_ratio else logits.argmax(-1)
        return torch.stack(outs, dim=1)                         # (B, T-1, V)
