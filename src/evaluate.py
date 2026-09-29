"""Evaluate models on the SAME test set:  python -m src.evaluate --models seq2seq attention"""
import argparse
import math
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sacrebleu.metrics import BLEU, CHRF
from torch.nn.utils.rnn import pad_sequence

from src.data.dataset import ParallelDataset, make_loader, read_lines
from src.data.tokenizer import PAD, Tokenizers
from src.inference import beam_search, greedy_decode_batch
from src.models import load_model
from src.utils import count_params, evaluate_loss, get_device, load_config, load_json, save_json

_PUNCT = re.compile(r'([።፣፤፥፦፧፨!?,.;:"()\'])')


def am_tok(text):
    """Split punctuation from words so BLEU isn't hurt by '።' glued to a word."""
    return re.sub(r"\s+", " ", _PUNCT.sub(r" \1 ", text)).strip()


def scores(hyps, refs):
    bleu = BLEU(tokenize="none").corpus_score([am_tok(h) for h in hyps], [[am_tok(r) for r in refs]]).score
    chrf = CHRF().corpus_score(hyps, [refs]).score
    chrfpp = CHRF(word_order=2).corpus_score(hyps, [refs]).score
    return bleu, chrf, chrfpp


def decode_test(model, tok, src_lines, device, batch_size=128, beam=1, max_src=100):
    """Translate all test sentences (length-sorted batches; original order restored)."""
    ids = [tok.encode_src(s, max_src) for s in src_lines]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    hyps = [None] * len(ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        src = pad_sequence([torch.tensor(ids[i]) for i in idx], batch_first=True, padding_value=PAD).to(device)
        src_len = torch.tensor([len(ids[i]) for i in idx]).to(device)
        if beam > 1:
            seqs = [beam_search(model, src[j:j + 1, :src_len[j]], src_len[j:j + 1], beam) for j in range(len(idx))]
        else:
            seqs = greedy_decode_batch(model, src, src_len)
        for i, s in zip(idx, seqs):
            hyps[i] = tok.decode_trg(s)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return hyps, time.time() - t0


def single_sentence_latency(model, tok, src_lines, device, n=100):
    lat = []
    for s in src_lines[:n]:
        ids = tok.encode_src(s)
        src, src_len = torch.tensor([ids], device=device), torch.tensor([len(ids)], device=device)
        t0 = time.time()
        greedy_decode_batch(model, src, src_len)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat.append(time.time() - t0)
    return sum(lat) / len(lat)


def length_bucket(n):
    return "1-10" if n <= 10 else "11-20" if n <= 20 else "21-30" if n <= 30 else "31+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["seq2seq", "attention"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--max_test", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    pdir, ckdir, rdir = cfg["paths"]["processed_dir"], cfg["paths"]["ckpt_dir"], cfg["paths"]["results_dir"]
    tok = Tokenizers(cfg["paths"]["tokenizer_dir"])
    src_lines, ref_lines = read_lines(f"{pdir}/test.en"), read_lines(f"{pdir}/test.am")
    if args.max_test:
        src_lines, ref_lines = src_lines[:args.max_test], ref_lines[:args.max_test]
    test_ds = ParallelDataset(f"{pdir}/test.en", f"{pdir}/test.am", tok, cfg["tokenizer"]["max_subwords"],
                              drop_long=False, max_pairs=args.max_test)
    test_dl = make_loader(test_ds, cfg["train"]["batch_size"], False)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    src_words = [len(s.split()) for s in src_lines]

    rows, all_preds = [], {}
    for name in args.models:
        model, _ = load_model(f"{ckdir}/{name}_best.pt", device)
        test_loss = evaluate_loss(model, test_dl, criterion, device)
        hyps, dec_time = decode_test(model, tok, src_lines, device, beam=args.beam)
        bleu, chrf, chrfpp = scores(hyps, ref_lines)
        hist_path = f"{rdir}/history_{name}.json"
        train_min = load_json(hist_path)["total_time_sec"] / 60 if os.path.exists(hist_path) else float("nan")
        rows.append({
            "model": name, "BLEU": round(bleu, 2), "chrF": round(chrf, 2), "chrF++": round(chrfpp, 2),
            "test_loss": round(test_loss, 3), "perplexity": round(math.exp(test_loss), 1),
            "train_time_min": round(train_min, 1),
            "infer_sec_per_sentence(batched)": round(dec_time / len(src_lines), 5),
            "infer_sec_per_sentence(single)": round(single_sentence_latency(model, tok, src_lines, device), 5),
            "parameters": count_params(model)})
        all_preds[name] = hyps
        pd.DataFrame({"source": src_lines, "reference": ref_lines, "hypothesis": hyps,
                      "src_words": src_words}).to_csv(f"{rdir}/predictions_{name}.csv", index=False)
        print(rows[-1])

    table = pd.DataFrame(rows)
    table.to_csv(f"{rdir}/comparison_table.csv", index=False)
    save_json(rows, f"{rdir}/metrics.json")
    print("\n" + table.to_string(index=False))
    winner = table.sort_values("chrF", ascending=False).iloc[0]["model"]
    print(f"\nBest model by chrF: {winner}")

    # ---- metrics by source-length bucket ----
    buckets = pd.Series([length_bucket(n) for n in src_words])
    brow = []
    for name, hyps in all_preds.items():
        for b in ["1-10", "11-20", "21-30", "31+"]:
            idx = [i for i in range(len(hyps)) if buckets[i] == b]
            if len(idx) < 5:
                continue
            bl, ch, _ = scores([hyps[i] for i in idx], [ref_lines[i] for i in idx])
            brow.append({"model": name, "bucket": b, "n": len(idx), "BLEU": round(bl, 2), "chrF": round(ch, 2)})
    bdf = pd.DataFrame(brow)
    bdf.to_csv(f"{rdir}/metrics_by_length.csv", index=False)
    if not bdf.empty:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        for k, metric in enumerate(["BLEU", "chrF"]):
            for name, g in bdf.groupby("model"):
                ax[k].plot(g["bucket"], g[metric], marker="o", label=name)
            ax[k].set_title(f"{metric} by source length (words)"); ax[k].set_xlabel("bucket"); ax[k].legend()
        plt.tight_layout(); plt.savefig(f"{rdir}/metrics_by_length.png", dpi=150); plt.close()

    # ---- examples table: Source -> Reference -> Seq2Seq -> Attention ----
    if {"seq2seq", "attention"} <= set(all_preds):
        df = pd.DataFrame({"source": src_lines, "reference": ref_lines, "seq2seq": all_preds["seq2seq"],
                           "attention": all_preds["attention"], "src_words": src_words})
        pick = pd.concat([
            df[df.src_words <= 8].sample(min(6, (df.src_words <= 8).sum()), random_state=1),
            df[df.src_words.between(9, 20)].sample(min(8, df.src_words.between(9, 20).sum()), random_state=1),
            df[df.src_words > 20].sample(min(6, (df.src_words > 20).sum()), random_state=1)])
        pick.to_csv(f"{rdir}/examples.csv", index=False)
        print(f"saved {len(pick)} examples -> {rdir}/examples.csv")


if __name__ == "__main__":
    main()
