"""Train either model:  python -m src.train --model seq2seq   |   python -m src.train --model attention"""
import argparse
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.dataset import ParallelDataset, make_loader
from src.data.tokenizer import PAD, Tokenizers
from src.models import build_model
from src.utils import (count_params, ensure_dir, evaluate_loss, get_device, load_config,
                       load_json, save_json, set_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["seq2seq", "attention"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max_pairs", type=int, default=None, help="use only N training pairs (debug)")
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/<model>_last.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg, mcfg = cfg["train"], cfg["model"]
    epochs = args.epochs or tcfg["epochs"]
    set_seed(cfg["seed"])
    device = get_device()
    pdir, ckdir, rdir = cfg["paths"]["processed_dir"], ensure_dir(cfg["paths"]["ckpt_dir"]), ensure_dir(cfg["paths"]["results_dir"])

    tok = Tokenizers(cfg["paths"]["tokenizer_dir"])
    ml = cfg["tokenizer"]["max_subwords"]
    train_ds = ParallelDataset(f"{pdir}/train.en", f"{pdir}/train.am", tok, ml, True, args.max_pairs)
    val_ds = ParallelDataset(f"{pdir}/val.en", f"{pdir}/val.am", tok, ml, True)
    train_dl = make_loader(train_ds, tcfg["batch_size"], True, tcfg["num_workers"])
    val_dl = make_loader(val_ds, tcfg["batch_size"], False, tcfg["num_workers"])
    print(f"device={device} | train={len(train_ds):,} val={len(val_ds):,} pairs")

    model_args = dict(name=args.model, src_vocab=tok.src_vocab, trg_vocab=tok.trg_vocab, pad_id=PAD,
                      emb_dim=mcfg["emb_dim"], hidden_dim=mcfg["hidden_dim"],
                      n_layers=mcfg["n_layers"], dropout=mcfg["dropout"])
    model = build_model(**model_args).to(device)
    n_params = count_params(model)
    print(f"model={args.model} | parameters={n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)

    best_path, last_path = f"{ckdir}/{args.model}_best.pt", f"{ckdir}/{args.model}_last.pt"
    hist_path = f"{rdir}/history_{args.model}.json"
    history = {"train_loss": [], "val_loss": [], "epoch_time": [], "lr": []}
    best_val, bad, start_epoch = float("inf"), 0, 0
    if args.resume and os.path.exists(last_path):
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(st["state_dict"]); optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        history, best_val, bad, start_epoch = st["history"], st["best_val"], st["bad"], st["epoch"]
        print(f"resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        t0, total, n_tok = time.time(), 0.0, 0
        for src, src_len, trg in tqdm(train_dl, desc=f"epoch {epoch + 1}/{epochs}", leave=False):
            src, src_len, trg = src.to(device), src_len.to(device), trg.to(device)
            optimizer.zero_grad()
            out = model(src, src_len, trg, tf_ratio=tcfg["teacher_forcing"])
            gold = trg[:, 1:].reshape(-1)
            loss = criterion(out.reshape(-1, out.size(-1)), gold)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["clip"])
            optimizer.step()
            k = (gold != PAD).sum().item()
            total, n_tok = total + loss.item() * k, n_tok + k
        train_loss = total / n_tok
        val_loss = evaluate_loss(model, val_dl, criterion, device)
        scheduler.step(val_loss)
        dt = time.time() - t0
        for k_, v_ in zip(history, [train_loss, val_loss, dt, optimizer.param_groups[0]["lr"]]):
            history[k_].append(v_)
        history.update(total_time_sec=sum(history["epoch_time"]), n_params=n_params, model_args=model_args)
        print(f"epoch {epoch + 1:02d} | train {train_loss:.3f} | val {val_loss:.3f} "
              f"(ppl {torch.exp(torch.tensor(val_loss)):.1f}) | {dt / 60:.1f} min")

        if val_loss < best_val:
            best_val, bad = val_loss, 0
            torch.save({"model_args": model_args, "state_dict": model.state_dict(),
                        "epoch": epoch + 1, "val_loss": val_loss}, best_path)
            print("  saved best checkpoint")
        else:
            bad += 1
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "history": history, "best_val": best_val,
                    "bad": bad, "epoch": epoch + 1}, last_path)
        save_json(history, hist_path)
        if bad >= tcfg["patience"]:
            print("early stopping")
            break
    print(f"done. total training time: {history['total_time_sec'] / 60:.1f} min | best val loss {best_val:.3f}")


if __name__ == "__main__":
    main()
