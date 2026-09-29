"""Decoding (greedy / beam) + the Translator class used by the API, Gradio and analysis scripts."""
import os
import re

import torch
import torch.nn.functional as F

from src.data.clean import normalize_en
from src.data.tokenizer import BOS, EOS, Tokenizers
from src.models import load_model

MAX_SRC = 100


@torch.no_grad()
def greedy_decode_batch(model, src, src_len, max_len=None, return_attn=False):
    """Returns list of id-lists (without <sos>/<eos>) and optionally attention (B, steps, S)."""
    model.eval()
    dev = src.device
    max_len = max_len or int(src.size(1) * 1.5) + 10
    enc_out, mask, state = model.encode(src, src_len)
    B = src.size(0)
    tok = torch.full((B,), BOS, dtype=torch.long, device=dev)
    outs = torch.full((B, max_len), EOS, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    attns = []
    for t in range(max_len):
        logits, state, attn = model.decode_step(tok, state, enc_out, mask)
        tok = logits.argmax(-1)
        outs[:, t] = tok.masked_fill(done, EOS)
        done |= tok == EOS
        if return_attn and attn is not None:
            attns.append(attn)
        if done.all():
            break
    seqs = []
    for row in outs.tolist():
        seqs.append(row[: row.index(EOS)] if EOS in row else row)
    if return_attn:
        return seqs, (torch.stack(attns, dim=1) if attns else None)
    return seqs


@torch.no_grad()
def beam_search(model, src, src_len, beam=5, max_len=None, alpha=0.6):
    """Single-sentence beam search (batch size 1)."""
    model.eval()
    dev = src.device
    max_len = max_len or int(src.size(1) * 1.5) + 10
    enc_out, mask, state = model.encode(src, src_len)
    hyps, finished = [([BOS], 0.0, state)], []
    for _ in range(max_len):
        cands = []
        for toks, score, st in hyps:
            logits, new_st, _ = model.decode_step(torch.tensor([toks[-1]], device=dev), st, enc_out, mask)
            top_lp, top_i = F.log_softmax(logits, -1)[0].topk(beam)
            cands += [(toks + [i], score + lp, new_st) for lp, i in zip(top_lp.tolist(), top_i.tolist())]
        cands.sort(key=lambda c: c[1], reverse=True)
        hyps = []
        for toks, score, st in cands[:beam]:
            if toks[-1] == EOS:
                finished.append((toks, score))
            else:
                hyps.append((toks, score, st))
        if not hyps or len(finished) >= beam:
            break
    if not finished:
        finished = [(h[0], h[1]) for h in hyps]
    best = max(finished, key=lambda f: f[1] / (len(f[0]) ** alpha))[0]
    return [t for t in best if t not in (BOS, EOS)]


_SENT_SPLIT = re.compile(r"(?<=[.!?።፧])\s+")


def split_sentences(text):
    return [s for s in _SENT_SPLIT.split(text.strip()) if s]


class Translator:
    def __init__(self, model_name="attention", ckpt_dir="checkpoints", device=None, beam=1):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = Tokenizers(os.path.join(ckpt_dir, "tokenizer"))
        self.model, self.ckpt = load_model(os.path.join(ckpt_dir, f"{model_name}_best.pt"), self.device)
        self.beam, self.model_name = beam, model_name

    def _encode(self, sentence):
        ids = self.tok.encode_src(normalize_en(sentence), MAX_SRC)       # SAME preprocessing as training
        src = torch.tensor([ids], device=self.device)
        return src, torch.tensor([len(ids)], device=self.device)

    def translate_sentence(self, sentence):
        src, src_len = self._encode(sentence)
        if self.beam > 1:
            ids = beam_search(self.model, src, src_len, beam=self.beam)
        else:
            ids = greedy_decode_batch(self.model, src, src_len)[0]
        return self.tok.decode_trg(ids)

    def translate(self, text):
        """Accepts one or several sentences."""
        return " ".join(self.translate_sentence(s) for s in split_sentences(text))

    def translate_with_attention(self, sentence):
        """Greedy decode returning (src_pieces, trg_pieces, attention[T, S])."""
        src, src_len = self._encode(sentence)
        seqs, attn = greedy_decode_batch(self.model, src, src_len, return_attn=True)
        if attn is None:
            raise ValueError("This model has no attention.")
        ids = seqs[0]
        rows = min(len(ids) + 1, attn.size(1))
        src_pieces = [self.tok.src_sp.id_to_piece(i) for i in src[0].tolist()]
        trg_pieces = [self.tok.trg_sp.id_to_piece(i) for i in ids] + ["<eos>"]
        return src_pieces, trg_pieces[:rows], attn[0, :rows].cpu().numpy()
