"""EXPERIMENT 1 - How much does each search technique reduce the work?

For each configuration we search the same set of positions to the same depth and count
"nodes" (positions visited). Fewer nodes for the SAME answer = a better algorithm.

Configurations (each one adds a technique to the previous):
  minimax              no pruning at all (the textbook baseline)
  alpha-beta           pruning, but moves tried in default order
  + MVV-LVA            captures tried first (most valuable victim, least valuable attacker)
  + killers/history    quiet moves that caused cutoffs earlier are tried first
  + transposition tbl  Zobrist-hashed table of already-searched positions

Quiescence search is OFF here (it changes the tree, so it is evaluated by playing matches).

Usage:  python scripts/exp_search_ablation.py --depths 3 4
"""
import argparse
import csv
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import chess  # noqa: E402

from _plot import ACCENT, INK, MUTED, new_fig, plt  # noqa: E402
from chessbot.evaluation import load_eval  # noqa: E402
from chessbot.search import Searcher  # noqa: E402

DEFAULT_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r2q1rk1/ppp2ppp/2n1bn2/3pp3/1b1P4/2NBPN2/PPP2PPP/R1BQ1RK1 w - - 0 9",
]

OFF = dict(use_pruning=False, use_mvv=False, use_killers=False, use_tt=False, use_quiescence=False)
CONFIGS = [
    ("minimax", dict(OFF)),
    ("alpha-beta", dict(OFF, use_pruning=True)),
    ("+ MVV-LVA", dict(OFF, use_pruning=True, use_mvv=True)),
    ("+ killers/history", dict(OFF, use_pruning=True, use_mvv=True, use_killers=True)),
    ("+ transposition table", dict(OFF, use_pruning=True, use_mvv=True, use_killers=True, use_tt=True)),
]


def load_positions(path):
    fens = list(DEFAULT_FENS)
    p = pathlib.Path(path)
    if p.exists():
        extra = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        fens += extra[:20]
        print(f"Using {len(DEFAULT_FENS)} standard positions + {min(20, len(extra))} from {path}")
    else:
        print(f"Using {len(DEFAULT_FENS)} standard positions ({path} not found)")
    boards = []
    for fen in fens:
        try:
            boards.append(chess.Board(fen))
        except ValueError:
            print("  skipping invalid FEN:", fen)
    return boards


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--eval", default="material")
    ap.add_argument("--positions", default="data/test_positions.fen")
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-minimax-depth", type=int, default=3,
                    help="plain minimax explodes; skip it above this depth")
    args = ap.parse_args()

    evaluator = load_eval(args.eval)
    boards = load_positions(args.positions)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for depth in args.depths:
        for name, flags in CONFIGS:
            if name == "minimax" and depth > args.max_minimax_depth:
                continue
            for i, board in enumerate(boards):
                s = Searcher(evaluator, **flags)
                r = s.search(board, max_depth=depth)
                rows.append({"depth": depth, "config": name, "position": i, "nodes": r.nodes,
                             "seconds": r.seconds, "score": r.score})
            print(f"depth {depth}  {name:24s} done", flush=True)

    with open(out / "search_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Summary: geometric mean of nodes (node counts are heavy-tailed), plus the ratio to alpha-beta.
    summary = {}
    for depth in args.depths:
        summary[depth] = {}
        base = {r["position"]: r["nodes"] for r in rows if r["depth"] == depth and r["config"] == "alpha-beta"}
        ref_score = {r["position"]: r["score"] for r in rows if r["depth"] == depth and r["config"] == "alpha-beta"}
        for name, _ in CONFIGS:
            rs = [r for r in rows if r["depth"] == depth and r["config"] == name]
            if not rs:
                continue
            ratio = geomean([r["nodes"] / base[r["position"]] for r in rs])
            mismatches = sum(1 for r in rs if r["score"] != ref_score[r["position"]])
            summary[depth][name] = {
                "geomean_nodes": geomean([r["nodes"] for r in rs]),
                "mean_seconds": sum(r["seconds"] for r in rs) / len(rs),
                "nodes_vs_alpha_beta": ratio,
                "score_differs_from_alpha_beta": mismatches,
            }
    with open(out / "search_ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = []
    for depth in args.depths:
        lines.append(f"\nDepth {depth}  ({len(boards)} positions)")
        lines.append(f"{'configuration':24s} {'nodes (geo-mean)':>17s} {'vs alpha-beta':>14s} {'seconds':>9s} {'score differs':>14s}")
        for name, v in summary[depth].items():
            lines.append(f"{name:24s} {v['geomean_nodes']:17,.0f} {v['nodes_vs_alpha_beta']:13.2f}x {v['mean_seconds']:9.2f} "
                         f"{v['score_differs_from_alpha_beta']:14d}")
    text = "\n".join(lines)
    print(text)
    (out / "search_ablation.txt").write_text(text + "\n")

    # Plot: horizontal bars of geometric-mean nodes for the deepest depth that has every config
    depth = max(d for d in args.depths if len(summary[d]) == len(CONFIGS)) if any(
        len(summary[d]) == len(CONFIGS) for d in args.depths) else max(args.depths)
    names = list(summary[depth].keys())
    vals = [summary[depth][n]["geomean_nodes"] for n in names]
    fig, ax = new_fig(8, 3.8)
    ypos = list(range(len(names)))[::-1]
    ax.barh(ypos, vals, color=ACCENT, height=0.55)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, color=INK)
    ax.set_xscale("log")
    for y_, v in zip(ypos, vals):
        ax.text(v * 1.08, y_, f"{v:,.0f}", va="center", color=INK, fontsize=9)
    ax.set_xlim(right=max(vals) * 6)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e4e8ec")
    ax.set_xlabel(f"positions visited, depth {depth} (geometric mean over positions, log scale)", color=MUTED)
    ax.set_title("Search work saved by each technique (shorter = better)", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "search_ablation.png")
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()
