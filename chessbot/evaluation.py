"""Static evaluation functions.

An evaluation function looks at ONE position (no searching) and returns a score in
centipawns (1 pawn = 100) from WHITE's point of view. Positive = good for White.

We use a *linear piece-square* model:

    score = sum over White pieces of  W[piece_type][square]
          - sum over Black pieces of  W[piece_type][mirrored square]

W holds 6 piece types x 64 squares = 384 numbers. Black pieces use the vertically
mirrored square, so both colours share the same table (the model is symmetric by design).

* Setting every square of a piece type to its material value -> "material only" eval.
* Learning W from game results (scripts/train_eval.py) -> data-driven eval.
"""
import os

import chess
import numpy as np

from .constants import MATERIAL, N_FEATURES, material_weights  # noqa: F401  (re-exported)


def feature_vector(board: chess.Board) -> np.ndarray:
    """384-long vector x such that  eval_white = x . W  (counts: White +1, Black -1)."""
    x = np.zeros(N_FEATURES, dtype=np.int8)
    for sq, piece in board.piece_map().items():
        base = (piece.piece_type - 1) * 64
        if piece.color == chess.WHITE:
            x[base + sq] += 1
        else:
            x[base + (sq ^ 56)] -= 1  # sq ^ 56 flips the board vertically
    return x


class TableEval:
    """Linear piece-square evaluation (see module docstring). Returns White-POV centipawns."""

    def __init__(self, weights, name: str = "table"):
        w = np.asarray(weights, dtype=np.float64).reshape(6, 64)
        # plain Python ints in nested lists are faster than numpy in the hot loop
        self.table = [[int(round(v)) for v in row] for row in w]
        self.name = name

    def evaluate(self, board: chess.Board) -> int:
        t = self.table
        score = 0
        for sq, piece in board.piece_map().items():
            if piece.color == chess.WHITE:
                score += t[piece.piece_type - 1][sq]
            else:
                score -= t[piece.piece_type - 1][sq ^ 56]
        return score


def material_eval() -> TableEval:
    return TableEval(material_weights(), name="material")


def load_eval(spec: str) -> TableEval:
    """'material' -> material-only eval; otherwise a path to a .npy weight file (384 numbers)."""
    if spec == "material":
        return material_eval()
    if not os.path.exists(spec):
        raise FileNotFoundError(
            f"Evaluation weights '{spec}' not found. Train them first: python scripts/train_eval.py"
        )
    name = os.path.splitext(os.path.basename(spec))[0]
    return TableEval(np.load(spec), name=name)
