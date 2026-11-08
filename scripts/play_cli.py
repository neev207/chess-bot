"""Play against your engine in the terminal.  Moves can be SAN (Nf3, e4, O-O) or UCI (g1f3).

Usage:  python scripts/play_cli.py --eval models/learned_pst.npy --depth 4 --color white
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import chess  # noqa: E402

from chessbot.engine import build_engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="material")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--color", choices=["white", "black"], default="white")
    args = ap.parse_args()

    engine = build_engine({"eval": args.eval, "depth": args.depth})
    human = chess.WHITE if args.color == "white" else chess.BLACK
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        print("\n" + str(board) + "\n")
        if board.turn == human:
            text = input("Your move (or 'quit'): ").strip()
            if text == "quit":
                return
            try:
                try:
                    move = board.parse_san(text)
                except ValueError:
                    move = chess.Move.from_uci(text)
                    if move not in board.legal_moves:
                        raise ValueError
            except ValueError:
                print("Illegal or unreadable move, try again.")
                continue
            board.push(move)
        else:
            res = engine.analyse(board)
            print(f"Engine plays {board.san(res.move)}   (eval {res.score/100:+.2f}, {res.nodes} nodes, {res.seconds:.1f}s)")
            board.push(res.move)
    print("\n" + str(board))
    print("Game over:", board.result(claim_draw=True))


if __name__ == "__main__":
    main()
