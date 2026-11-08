"""EXPERIMENT 3 - How much playing strength does each extra ply of search buy?

Plays depth 2 vs 1, depth 3 vs 2, ... and chains the Elo differences into a "ladder".
Gives you a strength-vs-depth curve (diminishing returns are expected) and, via the
engine's measured speed, a way to talk about compute vs strength.

Usage:  python scripts/exp_depth_ladder.py --eval models/learned_pst.npy --max-depth 4 --pairs 20 --workers 4
"""
import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _plot import ACCENT, INK, MUTED, new_fig, plt  # noqa: E402
from chessbot.match import run_match  # noqa: E402
from chessbot.stats import format_summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="material")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    ladder = [{"depth": 1, "elo": 0.0, "lo": 0.0, "hi": 0.0}]
    steps = []
    for d in range(2, args.max_depth + 1):
        a = f"eval={args.eval},depth={d}"
        b = f"eval={args.eval},depth={d-1}"
        print(f"\n=== depth {d} vs depth {d-1} ===")
        r = run_match(a, b, pairs=args.pairs, workers=args.workers, seed=args.seed)
        print(format_summary(f"d{d}", f"d{d-1}", r))
        steps.append({"depth": d, **r})
        prev = ladder[-1]
        half_lo, half_hi = r["elo"] - r["elo_lo"], r["elo_hi"] - r["elo"]
        # independent matches: add the uncertainties in quadrature
        err_lo = math.hypot(prev["elo"] - prev["lo"], half_lo)
        err_hi = math.hypot(prev["hi"] - prev["elo"], half_hi)
        elo = prev["elo"] + r["elo"]
        ladder.append({"depth": d, "elo": elo, "lo": elo - err_lo, "hi": elo + err_hi})

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "depth_ladder.json", "w") as f:
        json.dump({"eval": args.eval, "steps": steps, "ladder": ladder}, f, indent=2)

    fig, ax = new_fig(7, 4)
    xs = [p["depth"] for p in ladder]
    ys = [p["elo"] for p in ladder]
    lo = [p["elo"] - p["lo"] for p in ladder]
    hi = [p["hi"] - p["elo"] for p in ladder]
    ax.errorbar(xs, ys, yerr=[lo, hi], color=ACCENT, linewidth=2, marker="o", markersize=7,
                capsize=3, markerfacecolor=ACCENT, markeredgecolor="white")
    ax.set_xticks(xs)
    ax.set_xlabel("search depth (plies)", color=MUTED)
    ax.set_ylabel("Elo relative to depth 1", color=MUTED)
    ax.set_title("Playing strength vs search depth (error bars: 95% CI)", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "depth_ladder.png")
    print(f"\nSaved {out/'depth_ladder.json'} and {out/'depth_ladder.png'}")


if __name__ == "__main__":
    main()
