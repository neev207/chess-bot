"""Constants shared by evaluation and training (numpy only, no chess import)."""
import numpy as np

# Centipawn material values, indexed by (piece_type - 1): Pawn, Knight, Bishop, Rook, Queen, King
MATERIAL = [100, 320, 330, 500, 900, 0]
N_FEATURES = 6 * 64


def material_weights() -> np.ndarray:
    """384 weights where every square of a piece type holds that piece's material value."""
    return np.repeat(np.array(MATERIAL, dtype=np.float64), 64)
