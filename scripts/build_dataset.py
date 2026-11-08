"""PGN file(s) -> data/positions.npz  (features + labels) and data/test_positions.fen.

Usage:
  python scripts/build_dataset.py data/lichess_sample.pgn --max-games 20000 --min-elo 1600
  python scripts/build_dataset.py data/selfplay.pgn
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chessbot.data import build_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn", help=".pgn or .pgn.zst file")
    ap.add_argument("--out", default="data/positions.npz")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--min-elo", type=int, default=0, help="both players must be at least this rated")
    ap.add_argument("--per-game", type=int, default=6, help="positions sampled per game")
    ap.add_argument("--skip-plies", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fens", default="data/test_positions.fen", help="where to save sample FENs for benchmarks")
    args = ap.parse_args()

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    info = build_dataset(args.pgn, args.out, max_games=args.max_games, min_elo=args.min_elo,
                         per_game=args.per_game, skip_plies=args.skip_plies, seed=args.seed,
                         fen_dump_path=args.fens)
    print(f"Saved {info['positions']} positions from {info['games']} games to {args.out}")


if __name__ == "__main__":
    main()
