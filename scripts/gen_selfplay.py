"""Generate training games WITHOUT downloading anything: the engine plays itself.

Variety comes from random opening moves and a small chance (--eps) of a random move.
This is a convenient fallback. Real human games (Lichess) make a better training set:
see GUIDE.md, Day 3.

Usage:  python scripts/gen_selfplay.py --games 400 --workers 4 --depth 2 --out data/selfplay.pgn
"""
import argparse
import pathlib
import random
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import chess  # noqa: E402
import chess.pgn  # noqa: E402

from chessbot.engine import build_engine  # noqa: E402


def one_game(args):
    seed, depth, eps, opening_plies, max_plies = args
    rng = random.Random(seed)
    engine = build_engine({"eval": "material", "depth": depth})
    board = chess.Board()
    for _ in range(opening_plies):
        legal = list(board.legal_moves)
        if not legal:  # (astronomically rare) game already over inside the random opening
            break
        board.push(rng.choice(legal))
    while board.outcome(claim_draw=True) is None and len(board.move_stack) < max_plies:
        if rng.random() < eps:
            move = rng.choice(list(board.legal_moves))
        else:
            move = engine.choose_move(board)
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    result = outcome.result() if outcome else "1/2-1/2"  # too long -> call it a draw
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "selfplay"
    game.headers["Result"] = result
    return str(game)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--eps", type=float, default=0.08, help="probability of a random move")
    ap.add_argument("--opening-plies", type=int, default=6)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/selfplay.pgn")
    args = ap.parse_args()

    tasks = [(args.seed * 100_003 + i, args.depth, args.eps, args.opening_plies, args.max_plies)
             for i in range(args.games)]
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f, ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, pgn in enumerate(ex.map(one_game, tasks), 1):
            f.write(pgn + "\n\n")
            if i % 20 == 0 or i == args.games:
                print(f"  {i}/{args.games} games", flush=True)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
