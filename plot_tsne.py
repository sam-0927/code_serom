"""
plot_tsne.py

Reads per-character embeddings saved by text_extract.py and draws t-SNE plots.
Automatically iterates every leaf subfolder of --emb_dir.

Filename convention expected:
  {stem}_{char}_{idx}_emb.npy   (content_emb)
  {stem}_{char}_{idx}_fc.npy    (ctc_proj output)

Usage:
  # both emb & fc for all subfolders (default)
  python plot_tsne.py --emb_dir text_emb_dir

  # only emb type for all subfolders
  python plot_tsne.py --emb_dir text_emb_dir --type emb
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ─────────────────────────────────────────────────────────────────────────────

def parse_char(stem: str, emb_type: str) -> str | None:
    """
    stem: e.g. '1089-134691-0009_h_0_emb'
    Returns the character label ('h', 'ap', ...) or None if unparseable.
    """
    suffix = f"_{emb_type}"
    if not stem.endswith(suffix):
        return None
    core  = stem[: -len(suffix)]
    parts = core.rsplit("_", 2)        # [file_stem, char, idx]
    if len(parts) < 3:
        return None
    char = parts[1]
    if re.fullmatch(r"[a-z]|sp|ap", char):
        return char
    return None


def generate_plot(emb_dir: Path, emb_type: str, out_path: Path,
                  perplexity: float | None, seed: int):
    files = sorted(emb_dir.glob(f"*_{emb_type}.npy"))
    if not files:
        print(f"  [skip] no '*_{emb_type}.npy' in {emb_dir}")
        return

    embeddings, labels = [], []
    for f in files:
        char = parse_char(f.stem, emb_type)
        if char is None:
            continue
        embeddings.append(np.load(str(f)))
        labels.append(char)

    if len(embeddings) < 2:
        print(f"  [skip] too few valid embeddings ({len(embeddings)}) in {emb_dir}")
        return

    X = np.stack(embeddings)
    n = len(labels)

    perp = perplexity if perplexity is not None else n * 0.1
    perp = float(np.clip(perp, 5, n - 1))
    print(f"  type={emb_type}  n={n}  perplexity={perp:.1f}  → {out_path}")

    coords = TSNE(n_components=2, perplexity=perp, random_state=seed).fit_transform(X)

    unique_labels = sorted(set(labels))
    cmap          = matplotlib.colormaps["tab20"].resampled(len(unique_labels))
    label2color   = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl in unique_labels:
        mask = np.array([l == lbl for l in labels])
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=label2color[lbl], label=lbl, s=10, alpha=0.7)

    ax.legend(markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=8, title="char")
    ax.set_title(f"t-SNE  type={emb_type}  perplexity={perp:.0f}  n={n}\n{emb_dir}")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="t-SNE plot of per-character embeddings from text_extract.py"
    )
    p.add_argument("--emb_dir",    required=True,
                   help="Root embedding directory (text_emb_dir)")
    p.add_argument("--type",       choices=["emb", "fc", "all"], default="all",
                   help="Embedding type: 'emb', 'fc', or 'all' (default: all)")
    p.add_argument("--tsne_dir",   default="tsne",
                   help="Output directory for plots (default: tsne/)")
    p.add_argument("--perplexity", type=float, default=None,
                   help="t-SNE perplexity (default: 10%% of sample count)")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args     = parse_args()
    emb_root = Path(args.emb_dir)
    tsne_dir = Path(args.tsne_dir)
    tsne_dir.mkdir(parents=True, exist_ok=True)

    emb_types = ("emb", "fc") if args.type == "all" else (args.type,)

    # collect all leaf directories that contain .npy files
    leaf_dirs = sorted({f.parent for f in emb_root.rglob("*.npy")})
    if not leaf_dirs:
        print(f"No .npy files found under {emb_root}")
        return

    print(f"Found {len(leaf_dirs)} folder(s) under {emb_root}")

    for leaf in leaf_dirs:
        # build filename: use relative path from emb_root for reverb+snr info
        rel     = leaf.relative_to(emb_root)
        rel_str = "_".join(rel.parts) if rel.parts else leaf.name
        print(f"\n[{rel_str}]")
        for emb_type in emb_types:
            out_path = tsne_dir / f"tsne_{emb_type}_{rel_str}.png"
            try:
                generate_plot(leaf, emb_type, out_path, args.perplexity, args.seed)
            except Exception as e:
                print(f"  [error] {emb_type} {rel_str}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
