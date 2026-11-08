"""Turn PGN game files (e.g. from Lichess) into a labelled training set.

Each training example is:  (384-number feature vector of a position,  game result 1 / 0.5 / 0).

Only "quiet" positions are kept (not in check, last move was not a capture): a static
evaluation cannot judge a position in the middle of a capture sequence, so those would
just add label noise. The first few plies are skipped too (opening theory is not
informative about who wins).
"""
import io
import random
from typing import Iterator, List, Optional

import chess
import chess.pgn
import numpy as np

from .evaluation import feature_vector

RESULT_MAP = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def open_text(path: str):
    """Open .pgn or .pgn.zst (needs `pip install zstandard`) as a text stream."""
    if str(path).endswith(".zst"):
        import zstandard
        raw = open(path, "rb")
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def iter_games(path: str, max_games: Optional[int] = None, min_elo: int = 0) -> Iterator[chess.pgn.Game]:
    n = 0
    with open_text(path) as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            headers = game.headers
            if headers.get("Result") not in RESULT_MAP:
                continue
            if min_elo:
                try:
                    if min(int(headers["WhiteElo"]), int(headers["BlackElo"])) < min_elo:
                        continue
                except (KeyError, ValueError):
                    continue
            yield game
            n += 1
            if max_games and n >= max_games:
                break


def quiet_position_fens(game: chess.pgn.Game, skip_plies: int = 8) -> List[str]:
    board = game.board()
    fens = []
    for ply, move in enumerate(game.mainline_moves(), start=1):
        was_capture = board.is_capture(move)
        board.push(move)
        if ply >= skip_plies and not was_capture and not board.is_check():
            fens.append(board.fen())
    return fens


def build_dataset(pgn_path: str, out_path: str, max_games: Optional[int] = None, min_elo: int = 0,
                  per_game: int = 6, skip_plies: int = 8, seed: int = 0,
                  fen_dump_path: Optional[str] = None, fen_dump_n: int = 40) -> dict:
    rng = random.Random(seed)
    X, y, gid = [], [], []
    dumped: List[str] = []
    n_games = 0
    for game in iter_games(pgn_path, max_games=max_games, min_elo=min_elo):
        result = RESULT_MAP[game.headers["Result"]]
        fens = quiet_position_fens(game, skip_plies)
        if not fens:
            continue
        chosen = rng.sample(fens, min(per_game, len(fens)))
        for fen in chosen:
            X.append(feature_vector(chess.Board(fen)))
            y.append(result)
            gid.append(n_games)
        if len(dumped) < fen_dump_n and fens:
            dumped.append(rng.choice(fens))
        n_games += 1
        if n_games % 500 == 0:
            print(f"  processed {n_games} games, {len(y)} positions", flush=True)
    if not y:
        raise RuntimeError("No positions extracted. Is the PGN empty or filtered too strictly?")
    np.savez_compressed(out_path, X=np.array(X, dtype=np.int8), y=np.array(y, dtype=np.float32),
                        gid=np.array(gid, dtype=np.int32))
    if fen_dump_path:
        with open(fen_dump_path, "w") as f:
            f.write("\n".join(dumped) + "\n")
    return {"games": n_games, "positions": len(y)}
