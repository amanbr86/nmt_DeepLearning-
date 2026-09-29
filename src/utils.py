import json
import os
import random

import numpy as np
import torch
import yaml


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    """Average token-level cross-entropy with full teacher forcing."""
    model.eval()
    total, n_tokens = 0.0, 0
    for src, src_len, trg in loader:
        src, src_len, trg = src.to(device), src_len.to(device), trg.to(device)
        out = model(src, src_len, trg, tf_ratio=1.0)
        gold = trg[:, 1:].reshape(-1)
        loss = criterion(out.reshape(-1, out.size(-1)), gold)
        k = (gold != criterion.ignore_index).sum().item()
        total += loss.item() * k
        n_tokens += k
    return total / max(n_tokens, 1)
