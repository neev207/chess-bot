"""Engine = evaluation function + search settings, buildable from a short text spec.

Spec string examples (comma separated key=value):
    "eval=material,depth=3"
    "eval=models/learned_pst.npy,depth=4,quiescence=0"
    "eval=models/learned_pst.npy,depth=10,time=0.5"     # 0.5 s per move, up to depth 10

Keys: eval, depth, time, pruning, mvv, killers, tt, quiescence, name   (flags: 1/0)
"""
from typing import Optional

import chess

from .evaluation import load_eval
from .search import Searcher, SearchResult


def parse_spec(spec) -> dict:
    if isinstance(spec, dict):
        return dict(spec)
    out = {}
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        key, _, value = item.partition("=")
        out[key.strip()] = value.strip()
    return out


def _flag(v) -> bool:
    return str(v).lower() in ("1", "true", "yes", "on")


class Engine:
    def __init__(self, evaluator, depth: int = 3, time_limit: Optional[float] = None,
                 name: Optional[str] = None, **search_flags):
        self.evaluator = evaluator
        self.depth = depth
        self.time_limit = time_limit
        self.name = name or f"{evaluator.name}_d{depth}"
        self.searcher = Searcher(evaluator, **search_flags)

    def new_game(self):
        self.searcher.clear()

    def analyse(self, board: chess.Board) -> SearchResult:
        return self.searcher.search(board, self.depth, self.time_limit)

    def choose_move(self, board: chess.Board) -> chess.Move:
        return self.analyse(board).move


def build_engine(spec) -> Engine:
    cfg = parse_spec(spec)
    evaluator = load_eval(cfg.get("eval", "material"))
    depth = int(cfg.get("depth", 3))
    tl = cfg.get("time")
    time_limit = float(tl) if tl not in (None, "", "none") else None
    flags = dict(
        use_pruning=_flag(cfg.get("pruning", 1)),
        use_mvv=_flag(cfg.get("mvv", 1)),
        use_killers=_flag(cfg.get("killers", 1)),
        use_tt=_flag(cfg.get("tt", 1)),
        use_quiescence=_flag(cfg.get("quiescence", 1)),
    )
    return Engine(evaluator, depth, time_limit, cfg.get("name"), **flags)
