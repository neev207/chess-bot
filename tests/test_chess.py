"""Tests for the engine itself. Run from the project root:  python -m pytest -q"""
import pathlib
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
chess = pytest.importorskip("chess", reason="python-chess is not installed (pip install python-chess)")

from chessbot.engine import build_engine  # noqa: E402
from chessbot.evaluation import TableEval, feature_vector, material_eval  # noqa: E402
from chessbot.match import play_game, random_opening  # noqa: E402
from chessbot.search import Searcher  # noqa: E402

MIDDLEGAME = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


def test_start_position_is_balanced():
    assert material_eval().evaluate(chess.Board()) == 0
    w = np.random.RandomState(0).randint(-50, 50, 384)
    assert TableEval(w).evaluate(chess.Board()) == 0  # colour-symmetric by construction


def test_feature_vector_matches_eval():
    rng = random.Random(1)
    w = np.random.RandomState(2).randint(-50, 50, 384)
    ev = TableEval(w)
    board = chess.Board()
    for _ in range(40):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        assert ev.evaluate(board) == int(feature_vector(board).astype(np.int64) @ w)


def test_finds_mate_in_one():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    res = Searcher(material_eval()).search(board, max_depth=3)
    assert res.move == chess.Move.from_uci("a1a8")
    assert res.score > 900_000


def test_wins_free_queen():
    board = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    res = Searcher(material_eval()).search(board, max_depth=2)
    assert res.move == chess.Move.from_uci("d1d5")


def test_does_not_modify_callers_board():
    board = chess.Board(MIDDLEGAME)
    before = board.fen()
    Searcher(material_eval()).search(board, max_depth=2)
    assert board.fen() == before


def test_alpha_beta_gives_same_value_as_minimax():
    """Pruning and move ordering must change SPEED, never the answer."""
    board = chess.Board(MIDDLEGAME)
    flags = dict(use_quiescence=False)
    mini = Searcher(material_eval(), use_pruning=False, use_mvv=False, use_killers=False,
                    use_tt=False, **flags).search(board, max_depth=3)
    full = Searcher(material_eval(), **flags).search(board, max_depth=3)
    plain_ab = Searcher(material_eval(), use_mvv=False, use_killers=False, use_tt=False,
                        **flags).search(board, max_depth=3)
    assert plain_ab.score == mini.score
    assert full.score == mini.score
    assert full.nodes < mini.nodes


def test_time_limit_is_respected_and_returns_legal_move():
    board = chess.Board(MIDDLEGAME)
    eng = build_engine("eval=material,depth=20,time=0.3")
    res = eng.analyse(board)
    assert res.move in board.legal_moves
    assert res.seconds < 2.0


def test_full_game_finishes_with_a_result():
    a = build_engine("eval=material,depth=1")
    b = build_engine("eval=material,depth=1")
    score, plies = play_game(a, b, random_opening(seed=3, plies=4), max_plies=60)
    assert score in (0.0, 0.5, 1.0) and plies > 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
