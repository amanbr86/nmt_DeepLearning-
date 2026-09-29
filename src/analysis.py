"""Error analysis helpers:  python -m src.analysis
 1. loss curves for both models
 2. automatic error flags (repetition, missing/extra words, rare source words, long sentences)
 3. a 50-row manual review sheet for word order / morphology / named entities (hand-labelled)"""
import argparse
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.dataset import read_lines
from src.utils import load_config, load_json


def plot_history(rdir, models):
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in models:
        p = f"{rdir}/history_{m}.json"
        if not os.path.exists(p):
            continue
        h = load_json(p)
        ax.plot(range(1, len(h["train_loss"]) + 1), h["train_loss"], "--", label=f"{m} train")
        ax.plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], label=f"{m} val")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{rdir}/loss_curves.png", dpi=150); plt.close()


def has_repetition(text):
    w = text.split()
    if any(a == b for a, b in zip(w, w[1:])):
        return True
    bigrams = Counter(zip(w, w[1:]))
    return any(c >= 2 for c in bigrams.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="+", default=["seq2seq", "attention"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    rdir, pdir = cfg["paths"]["results_dir"], cfg["paths"]["processed_dir"]
    plot_history(rdir, args.models)

    counts = Counter(w for l in read_lines(f"{pdir}/train.en") for w in l.split())
    rows = []
    for m in args.models:
        df = pd.read_csv(f"{rdir}/predictions_{m}.csv").fillna("")
        hw, rw = df.hypothesis.str.split().str.len(), df.reference.str.split().str.len()
        flags = pd.DataFrame({
            "repetition": df.hypothesis.map(has_repetition),
            "missing_words(hyp<0.6*ref)": hw < 0.6 * rw,
            "extra_words(hyp>1.5*ref)": hw > 1.5 * rw,
            "rare_source_word(train freq<=2)": df.source.map(lambda s: any(counts[w] <= 2 for w in s.split())),
            "long_sentence(>20 words)": df.src_words > 20,
            "empty_output": df.hypothesis.str.strip() == ""})
        rows.append((flags.mean() * 100).round(1).rename(m))
        df = df.assign(**flags)
        df.to_csv(f"{rdir}/error_flags_{m}.csv", index=False)
    summary = pd.concat(rows, axis=1)
    summary.to_csv(f"{rdir}/error_flag_rates.csv")
    print("% of test sentences flagged:\n", summary)

    # manual review sheet (same 50 sentences for every model)
    base = pd.read_csv(f"{rdir}/predictions_{args.models[0]}.csv").fillna("").sample(min(50, len(pd.read_csv(f"{rdir}/predictions_{args.models[0]}.csv"))), random_state=7)
    sheet = base[["source", "reference"]].copy()
    for m in args.models:
        sheet[m] = pd.read_csv(f"{rdir}/predictions_{m}.csv").fillna("").loc[sheet.index, "hypothesis"]
    for col in ["word_order", "missing_extra", "repetition", "morphology", "named_entity", "rare_word", "long_sentence"]:
        for m in args.models:
            sheet[f"{col}__{m}"] = ""      # fill with 1 / 0 by hand
    sheet.to_csv(f"{rdir}/manual_review_sheet.csv", index=False)
    print(f"Fill in {rdir}/manual_review_sheet.csv by hand, then sum each column for the report.")


if __name__ == "__main__":
    main()
