"""Attention heatmaps:  python -m src.visualize_attention --idx 3 10 25
                         python -m src.visualize_attention --text "I am going to the university."
Needs an Ethiopic-capable font (Noto Sans Ethiopic / Abyssinica SIL / Nyala), see README."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from src.data.dataset import read_lines
from src.inference import Translator
from src.utils import ensure_dir, load_config

FONTS = ["Noto Sans Ethiopic", "Noto Serif Ethiopic", "Abyssinica SIL", "Nyala", "Kefa", "Ethiopia Jiret"]


def set_amharic_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for f in FONTS:
        if f in available:
            plt.rcParams["font.family"] = f
            return f
    print("WARNING: no Ethiopic font found -> Amharic labels will be boxes. "
          "Install one: `sudo apt-get install -y fonts-noto-core` (then restart Python).")
    return None


def clean_piece(p):
    return p.replace("▁", "·") if p not in ("<eos>", "<sos>") else p


def plot_attention(src_pieces, trg_pieces, attn, path, title=""):
    fig, ax = plt.subplots(figsize=(max(5, .55 * len(src_pieces)), max(4, .45 * len(trg_pieces))))
    im = ax.imshow(attn, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(src_pieces))); ax.set_xticklabels([clean_piece(p) for p in src_pieces], rotation=90)
    ax.set_yticks(range(len(trg_pieces))); ax.set_yticklabels([clean_piece(p) for p in trg_pieces])
    ax.set_xlabel("English source (subwords)"); ax.set_ylabel("Amharic output (subwords)")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=.03)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--idx", type=int, nargs="*", default=[0, 1, 2, 3], help="test-set line numbers")
    ap.add_argument("--text", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_amharic_font()
    out = ensure_dir(os.path.join(cfg["paths"]["results_dir"], "attention_plots"))
    tr = Translator("attention", cfg["paths"]["ckpt_dir"])
    items = [(args.text, "custom")] if args.text else [
        (read_lines(f"{cfg['paths']['processed_dir']}/test.en")[i], f"test{i}") for i in args.idx]
    for text, tag in items:
        s, t, a = tr.translate_with_attention(text)
        path = os.path.join(out, f"attn_{tag}.png")
        plot_attention(s, t, a, path, title=text[:90])
        print("saved", path)


if __name__ == "__main__":
    main()
