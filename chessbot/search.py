"""Alpha-beta search with switchable features, so each one can be measured separately.

Features (each can be turned on/off -> ablation experiments):
  use_pruning    alpha-beta cutoffs (off = plain minimax; the slow baseline)
  use_mvv        move ordering: captures first, "most valuable victim / least valuable attacker"
  use_killers    move ordering: killer moves + history heuristic for quiet moves
  use_tt         transposition table (Zobrist hashing) + trying the stored best move first
  use_quiescence at depth 0, keep searching captures until the position is "quiet"

The search is "negamax": one function used for both sides, scores always from the point
of view of the side to move.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import chess
import chess.polyglot

INF = 10_000_000
MATE = 1_000_000
MATE_BOUND = MATE - 1000
MAX_PLY = 128
EXACT, LOWER, UPPER = 0, 1, 2

ORDER_VAL = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 20,
}


class TimeUp(Exception):
    pass


@dataclass
class SearchResult:
    move: Optional[chess.Move]
    score: int          # centipawns, side-to-move point of view (|score| >= ~999000 means mate)
    depth: int          # deepest fully completed iteration
    nodes: int
    seconds: float
    nodes_by_depth: Dict[int, int] = field(default_factory=dict)  # cumulative nodes after each depth


def _to_tt(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


def _from_tt(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


class Searcher:
    def __init__(self, evaluator, use_pruning=True, use_mvv=True, use_killers=True,
                 use_tt=True, use_quiescence=True, max_qdepth=8, tt_max_entries=1_000_000):
        self.evaluator = evaluator
        self.use_pruning = use_pruning
        self.use_mvv = use_mvv
        self.use_killers = use_killers
        self.use_tt = use_tt and use_pruning  # TT bounds only make sense with alpha-beta
        self.use_quiescence = use_quiescence
        self.max_qdepth = max_qdepth
        self.tt_max = tt_max_entries
        self.clear()

    # ------------------------------------------------------------------ public API
    def clear(self):
        """Forget everything (call between independent games / benchmark positions)."""
        self.tt = {}
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.history = {}
        self.nodes = 0
        self._start = 0.0
        self._time_limit = None

    def search(self, board: chess.Board, max_depth: int = 4, time_limit: Optional[float] = None) -> SearchResult:
        """Iterative deepening: search depth 1, then 2, ... up to max_depth (or until time runs out).

        The caller's board is never modified (we search on a copy)."""
        board = board.copy()
        self._start = time.perf_counter()
        self.nodes = 0
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.history = {}

        legal = list(board.legal_moves)
        if not legal:
            return SearchResult(None, 0, 0, 0, 0.0, {})

        best_move, best_score, done_depth = legal[0], 0, 0
        nodes_by_depth: Dict[int, int] = {}
        prev_best = None
        for depth in range(1, max_depth + 1):
            # depth 1 always completes, so we always have *some* move to play
            self._time_limit = time_limit if depth > 1 else None
            try:
                score, move = self._search_root(board, depth, prev_best)
            except TimeUp:
                break
            best_move, best_score, done_depth = move, score, depth
            prev_best = move
            nodes_by_depth[depth] = self.nodes
            if abs(score) >= MATE_BOUND:  # forced mate found; deeper search cannot improve it
                break
        return SearchResult(best_move, best_score, done_depth, self.nodes,
                            time.perf_counter() - self._start, nodes_by_depth)

    # ------------------------------------------------------------------ internals
    def _tick(self):
        self.nodes += 1
        if self._time_limit is not None and (self.nodes & 511) == 0:
            if time.perf_counter() - self._start > self._time_limit:
                raise TimeUp

    def _eval(self, board: chess.Board) -> int:
        e = self.evaluator.evaluate(board)  # White point of view
        return e if board.turn == chess.WHITE else -e

    @staticmethod
    def _mvv_lva(board: chess.Board, move: chess.Move) -> int:
        """Most Valuable Victim - Least Valuable Attacker score (bigger = try earlier)."""
        s = 0
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square) or chess.PAWN  # en passant: square is empty
            attacker = board.piece_type_at(move.from_square)
            s += 100 * ORDER_VAL[victim] - ORDER_VAL[attacker]
        if move.promotion:
            s += 100 * ORDER_VAL[move.promotion]
        return s

    def _order(self, board, moves, tt_move, ply):
        if not (self.use_mvv or self.use_killers or tt_move is not None):
            return moves  # no ordering: keep python-chess's default generation order
        killers = self.killers[ply] if (self.use_killers and ply < MAX_PLY) else ()
        keyed = []
        for m in moves:
            if tt_move is not None and m == tt_move:
                s = 10_000_000
            elif self.use_mvv and (board.is_capture(m) or m.promotion):
                s = 1_000_000 + self._mvv_lva(board, m)
            elif self.use_killers:
                s = 900_000 if m in killers else self.history.get((m.from_square, m.to_square), 0)
            else:
                s = 0
            keyed.append((s, m))
        keyed.sort(key=lambda t: -t[0])  # stable sort: ties keep their original order
        return [m for _, m in keyed]

    def _search_root(self, board, depth, prev_best):
        moves = list(board.legal_moves)
        moves = self._order(board, moves, prev_best if self.use_tt else None, 0)
        alpha, beta = -INF, INF
        best_score, best_move = -INF, moves[0]
        for m in moves:
            board.push(m)
            if self.use_pruning:
                score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            else:
                score = -self._negamax(board, depth - 1, -INF, INF, 1)
            board.pop()
            if score > best_score:
                best_score, best_move = score, m
            if score > alpha:
                alpha = score
        return best_score, best_move

    def _negamax(self, board, depth, alpha, beta, ply):
        self._tick()

        # Draws (checked first, they are cheap)
        if board.halfmove_clock >= 100 or board.is_insufficient_material() or board.is_repetition(2):
            return 0

        # Horizon reached
        if depth <= 0:
            if self.use_quiescence:
                return self._quiesce(board, alpha, beta, ply, self.max_qdepth)
            return self._eval(board)

        # Transposition table probe
        alpha_orig = alpha
        tt_move = None
        key = None
        if self.use_tt:
            key = chess.polyglot.zobrist_hash(board)
            entry = self.tt.get(key)
            if entry is not None:
                e_depth, e_flag, e_score, e_move = entry
                tt_move = e_move
                if e_depth >= depth:
                    e_score = _from_tt(e_score, ply)
                    if e_flag == EXACT:
                        return e_score
                    if e_flag == LOWER and e_score > alpha:
                        alpha = e_score
                    elif e_flag == UPPER and e_score < beta:
                        beta = e_score
                    if alpha >= beta:
                        return e_score

        moves = list(board.legal_moves)
        if not moves:
            return -(MATE - ply) if board.is_check() else 0  # checkmate : stalemate
        moves = self._order(board, moves, tt_move, ply)

        best, best_move = -INF, None
        for m in moves:
            board.push(m)
            if self.use_pruning:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                score = -self._negamax(board, depth - 1, -INF, INF, ply + 1)
            board.pop()
            if score > best:
                best, best_move = score, m
            if self.use_pruning:
                if score > alpha:
                    alpha = score
                if alpha >= beta:  # beta cutoff: opponent will never allow this line
                    if self.use_killers and not board.is_capture(m) and not m.promotion and ply < MAX_PLY:
                        k = self.killers[ply]
                        if k[0] != m:
                            k[1] = k[0]
                            k[0] = m
                        hk = (m.from_square, m.to_square)
                        self.history[hk] = self.history.get(hk, 0) + depth * depth
                    break

        if self.use_tt:
            if best <= alpha_orig:
                flag = UPPER
            elif best >= beta:
                flag = LOWER
            else:
                flag = EXACT
            old = self.tt.get(key)
            if old is None or old[0] <= depth:
                if len(self.tt) >= self.tt_max:
                    self.tt.clear()
                self.tt[key] = (depth, flag, _to_tt(best, ply), best_move)
        return best

    def _quiesce(self, board, alpha, beta, ply, qdepth):
        """Search only captures/promotions so we don't evaluate in the middle of an exchange."""
        self._tick()
        if board.is_check():
            if qdepth <= 0:
                return self._eval(board)
            moves = list(board.legal_moves)
            if not moves:
                return -(MATE - ply)
            best = -INF
        else:
            stand_pat = self._eval(board)  # "I can always just stop capturing"
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            if qdepth <= 0:
                return stand_pat
            best = stand_pat
            moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]

        moves.sort(key=lambda m: -self._mvv_lva(board, m))
        for m in moves:
            board.push(m)
            score = -self._quiesce(board, -beta, -alpha, ply + 1, qdepth - 1)
            board.pop()
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return best
