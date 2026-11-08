"""Minimal UCI (Universal Chess Interface) front-end, so the engine can be used in chess GUIs
(Arena, Cute Chess, ...) or with lichess-bot.

Run:  python -m chessbot.uci --eval models/learned_pst.npy --depth 4
"""
import argparse
import sys

import chess

from .engine import build_engine


def _parse_position(tokens):
    if tokens[0] == "startpos":
        board, idx = chess.Board(), 1
    else:  # "fen <6 fields>"
        board, idx = chess.Board(" ".join(tokens[1:7])), 7
    if idx < len(tokens) and tokens[idx] == "moves":
        for u in tokens[idx + 1:]:
            board.push_uci(u)
    return board


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="material")
    ap.add_argument("--depth", type=int, default=4)
    args = ap.parse_args(argv)
    engine = build_engine({"eval": args.eval, "depth": args.depth})
    board = chess.Board()

    for line in sys.stdin:
        tokens = line.split()
        if not tokens:
            continue
        cmd = tokens[0]
        if cmd == "uci":
            print("id name ChessBot-Research")
            print("id author student")
            print("uciok")
        elif cmd == "isready":
            print("readyok")
        elif cmd == "ucinewgame":
            board = chess.Board()
            engine.new_game()
        elif cmd == "position":
            board = _parse_position(tokens[1:])
        elif cmd == "go":
            opts = {}
            for i, tok in enumerate(tokens):
                if tok in ("depth", "movetime", "wtime", "btime", "winc", "binc") and i + 1 < len(tokens):
                    opts[tok] = int(tokens[i + 1])
            depth, time_limit = engine.depth, None
            if "depth" in opts:
                depth = opts["depth"]
            elif "movetime" in opts:
                depth, time_limit = 12, opts["movetime"] / 1000.0
            elif "wtime" in opts or "btime" in opts:
                mine, inc = ("wtime", "winc") if board.turn == chess.WHITE else ("btime", "binc")
                remaining = opts.get(mine, 60000) / 1000.0
                time_limit = max(0.05, remaining / 30.0 + opts.get(inc, 0) / 2000.0)
                depth = 12
            res = engine.searcher.search(board, depth, time_limit)
            print(f"info depth {res.depth} score cp {res.score} nodes {res.nodes} time {int(res.seconds * 1000)}")
            print(f"bestmove {res.move.uci() if res.move else '0000'}")
        elif cmd == "quit":
            break
        sys.stdout.flush()


if __name__ == "__main__":
    main()
