"""Match statistics: turn game scores into an Elo difference with a 95% confidence interval.

No chess import here on purpose, so it is easy to test.
"""
import math
from typing import Dict, List


def elo_from_score(score: float) -> float:
    """Expected score s (0..1) -> Elo difference.  s=0.5 -> 0,  s=0.64 -> ~+100."""
    s = min(max(score, 1e-3), 1 - 1e-3)  # clip so 100% / 0% scores don't give infinity
    return -400.0 * math.log10(1.0 / s - 1.0)


def summarize(scores: List[float]) -> Dict[str, float]:
    """scores: one number per game from engine A's view (1 win, 0.5 draw, 0 loss)."""
    n = len(scores)
    if n == 0:
        raise ValueError("no games")
    wins = sum(1 for s in scores if s == 1)
    draws = sum(1 for s in scores if s == 0.5)
    losses = n - wins - draws
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    return {
        "games": n, "wins": wins, "draws": draws, "losses": losses,
        "score": mean,
        "elo": elo_from_score(mean),
        "elo_lo": elo_from_score(lo),
        "elo_hi": elo_from_score(hi),
    }


def format_summary(name_a: str, name_b: str, r: Dict[str, float]) -> str:
    return (f"{name_a} vs {name_b}: +{r['wins']} ={r['draws']} -{r['losses']} "
            f"({r['games']} games)  score {r['score']*100:.1f}%  "
            f"Elo diff {r['elo']:+.0f}  [95% CI {r['elo_lo']:+.0f}, {r['elo_hi']:+.0f}]")
