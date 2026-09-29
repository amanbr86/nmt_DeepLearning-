import torch

from src.models.attn_seq2seq import AttnSeq2Seq
from src.models.seq2seq import Seq2Seq

MODELS = {"seq2seq": Seq2Seq, "attention": AttnSeq2Seq}


def build_model(name, src_vocab, trg_vocab, emb_dim, hidden_dim, n_layers, dropout, pad_id=0):
    return MODELS[name](src_vocab, trg_vocab, emb_dim, hidden_dim, n_layers, dropout, pad_id)


def load_model(path, device="cpu"):
    """Rebuild a model from a checkpoint saved by train.py."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt
