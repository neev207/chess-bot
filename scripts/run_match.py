"""EXPERIMENT 2 - Is engine A stronger than engine B?  (any two configurations)

Engines are described by a spec string (see chessbot/engine.py). Examples:

  # Does the LEARNED evaluation beat the hand-written material evaluation at equal search depth?
  python scripts/run_match.py --a "eval=models/learned_pst.npy,depth=3" --b "eval=material,depth=3" --pairs 50 --workers 4

  # Does quiescence search help?
  python scripts/run_match.py --a "eval=material,depth=3,quiescence=1" --b "eval=material,depth=3,quiescence=0" --pairs 50

  # Equal TIME instead of equal depth (fairer, because a learned eval costs more per position)
  python scripts/run_match.py --a "eval=models/learned_pst.npy,depth=12,time=0.3" --b "eval=material,depth=12,time=0.3" --pairs 50

Each opening is played twice with colours swapped. Result: score, Elo difference, 95% CI.
"""
import argparse
import json
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chessbot.match import run_match  # noqa: E402
from chessbot.stats import format_summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--pairs", type=int, default=25, help="number of openings; games = 2 x pairs")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default=None, help="name for the results file")
    args = ap.parse_args()

    t0 = time.time()
    r = run_match(args.a, args.b, pairs=args.pairs, workers=args.workers, seed=args.seed,
                  opening_plies=args.opening_plies, max_plies=args.max_plies)
    r["minutes"] = (time.time() - t0) / 60
    print("\n" + format_summary("A", "B", r))
    print(f"A = {args.a}\nB = {args.b}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or re.sub(r"[^A-Za-z0-9]+", "_", f"{args.a}_vs_{args.b}")[:80]
    with open(out / f"match_{tag}.json", "w") as f:
        json.dump({"a": args.a, "b": args.b, "seed": args.seed, "pairs": args.pairs, **r}, f, indent=2)
    print(f"Saved {out / ('match_' + tag + '.json')}")


if __name__ == "__main__":
    main()
