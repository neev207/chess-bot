"""Train the learned evaluation and compare it with the hand-written material evaluation.

Outputs:
  models/learned_pst.npy           the learned 384 weights
  results/eval_training.json       metrics table (material vs learned vs coin-flip) + loss history
  results/training_curve.png       train/validation loss per epoch
  results/learned_pst.png          what the model learned (bonus/penalty per square, per piece)

Usage:  python scripts/train_eval.py --data data/positions.npz
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _plot import ACCENT, ACCENT2, DIV_HIGH, DIV_LOW, INK, MUTED, new_fig, plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from chessbot.constants import material_weights  # noqa: E402
from chessbot.training import evaluate_model, split_by_game, train  # noqa: E402

PIECES = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]


def plot_curve(history, path):
    fig, ax = new_fig()
    ep = [h["epoch"] for h in history]
    tr = [h["train_logloss"] for h in history]
    va = [h["val_logloss"] for h in history]
    ax.plot(ep, tr, color=ACCENT2, linewidth=2)
    ax.plot(ep, va, color=ACCENT, linewidth=2)
    ax.text(ep[-1], tr[-1], "  train", color=INK, va="center", fontsize=9)
    ax.text(ep[-1], va[-1], "  validation", color=INK, va="center", fontsize=9)
    ax.set_xlabel("epoch", color=MUTED)
    ax.set_ylabel("log-loss (lower is better)", color=MUTED)
    ax.set_title("Learning the evaluation function", loc="left", color=INK, fontsize=12)
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pst(w, path):
    """Heatmaps of (learned - material) per piece: warm = penalty, cool = bonus, white = no change."""
    cmap = LinearSegmentedColormap.from_list("div", [DIV_LOW, "#ffffff", DIV_HIGH])
    delta = (w - material_weights()).reshape(6, 8, 8)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6), dpi=150)
    fig.patch.set_facecolor("white")
    for p, ax in enumerate(axes.ravel()):
        lim = max(1.0, float(np.abs(delta[p]).max()))
        ax.imshow(delta[p], origin="lower", cmap=cmap, vmin=-lim, vmax=lim)  # rank 1 at the bottom
        ax.set_title(f"{PIECES[p]}  (±{lim:.0f} cp)", color=INK, fontsize=10, loc="left")
        ax.set_xticks(range(8))
        ax.set_xticklabels(list("abcdefgh"), color=MUTED, fontsize=7)
        ax.set_yticks(range(8))
        ax.set_yticklabels([str(i) for i in range(1, 9)], color=MUTED, fontsize=7)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle("Learned positional bonus per square, White's point of view "
                 "(blue = bonus, orange = penalty)", color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/positions.npz")
    ap.add_argument("--out", default="models/learned_pst.npy")
    ap.add_argument("--results", default="results")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--l2", type=float, default=2e-7)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.data)
    X, y, gid = d["X"], d["y"], d["gid"]
    tr, va = split_by_game(gid, args.val_frac, args.seed)
    print(f"{len(y)} positions from {len(np.unique(gid))} games  ->  train {len(tr)}, validation {len(va)}")
    X_tr, y_tr, X_va, y_va = X[tr], y[tr], X[va], y[va]

    w0 = material_weights()
    w, history = train(X_tr, y_tr, X_va, y_va, w0, epochs=args.epochs, batch=args.batch,
                       lr=args.lr, l2=args.l2, seed=args.seed)

    coin = float(-(y_va * math.log(0.5) + (1 - y_va) * math.log(0.5)).mean())
    table = {
        "coin_flip (always 50%)": {"logloss": coin, "sign_accuracy": 0.5},
        "material_only": evaluate_model(X_va, y_va, w0),
        "learned": evaluate_model(X_va, y_va, w),
    }
    print("\nValidation metrics (positions from games the model never saw):")
    print(f"{'model':26s} {'log-loss':>9s} {'sign-acc':>9s}")
    for name, m in table.items():
        print(f"{name:26s} {m['logloss']:9.4f} {m['sign_accuracy']*100:8.1f}%")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, w)
    res = pathlib.Path(args.results)
    res.mkdir(parents=True, exist_ok=True)
    with open(res / "eval_training.json", "w") as f:
        json.dump({"args": vars(args), "n_train": int(len(tr)), "n_val": int(len(va)),
                   "validation_metrics": table, "history": history}, f, indent=2)
    plot_curve(history, res / "training_curve.png")
    plot_pst(w, res / "learned_pst.png")
    print(f"\nSaved {out}, {res/'eval_training.json'}, and two plots in {res}/")


if __name__ == "__main__":
    main()
