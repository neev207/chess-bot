"""Play engines against each other.

Fairness measures (these are what make the result trustworthy):
  * every opening is played TWICE with colours swapped,
  * openings are random (a few random plies), reproducible via a seed,
  * games are spread across CPU cores with multiprocessing.
"""
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import chess

from .engine import build_engine
from .stats import summarize


def random_opening(seed: int, plies: int = 4) -> List[str]:
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    return [m.uci() for m in board.move_stack]


def play_game(white, black, opening_ucis, max_plies: int = 300) -> Tuple[float, int]:
    """Returns (White's score, number of plies). Reaching max_plies is adjudicated as a draw."""
    board = chess.Board()
    for u in opening_ucis:
        board.push_uci(u)
    white.new_game()
    black.new_game()
    while True:
        outcome = board.outcome(claim_draw=True)  # mate, stalemate, 3-fold, 50-move, ...
        if outcome is not None:
            if outcome.winner is None:
                return 0.5, len(board.move_stack)
            return (1.0 if outcome.winner == chess.WHITE else 0.0), len(board.move_stack)
        if len(board.move_stack) >= max_plies:
            return 0.5, len(board.move_stack)
        engine = white if board.turn == chess.WHITE else black
        board.push(engine.choose_move(board))


def _play_pair(args):
    spec_a, spec_b, seed, opening_plies, max_plies = args
    a, b = build_engine(spec_a), build_engine(spec_b)
    opening = random_opening(seed, opening_plies)
    white_score_1, plies1 = play_game(a, b, opening, max_plies)   # A is White
    white_score_2, plies2 = play_game(b, a, opening, max_plies)   # B is White
    return [white_score_1, 1.0 - white_score_2], plies1 + plies2


def run_match(spec_a, spec_b, pairs: int = 20, workers: int = 1, seed: int = 0,
              opening_plies: int = 4, max_plies: int = 300, verbose: bool = True) -> dict:
    """Play `pairs` x 2 games. Returns the dict from stats.summarize (from A's point of view)."""
    tasks = [(spec_a, spec_b, seed * 100_003 + i, opening_plies, max_plies) for i in range(pairs)]
    scores: List[float] = []
    total_plies = 0

    def record(res, done):
        nonlocal total_plies
        s, plies = res
        scores.extend(s)
        total_plies += plies
        if verbose:
            running = sum(scores) / len(scores)
            print(f"  pair {done}/{pairs}  running score for A: {running*100:.1f}%", file=sys.stderr, flush=True)

    if workers <= 1:
        for i, t in enumerate(tasks, 1):
            record(_play_pair(t), i)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, res in enumerate(ex.map(_play_pair, tasks), 1):
                record(res, i)

    result = summarize(scores)
    result["avg_game_plies"] = total_plies / len(scores)
    return result
