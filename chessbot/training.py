"""Learn evaluation weights from game results ("Texel tuning").

Idea: a good evaluation score should predict who eventually wins. We model

    P(White scores) = 1 / (1 + 10^(-score/400))        score = x . w   (centipawns)

and fit w by minimising cross-entropy against the real game results (1 / 0.5 / 0).
It is logistic regression with a chess-specific link function. We start from the
material-only weights and regularise towards them, which keeps the model sane when
data is limited.

This file only needs numpy (no chess import), so it is easy to test.
"""
import math
from typing import Callable, Dict, List, Tuple

import numpy as np

C = math.log(10.0) / 400.0  # converts centipawns to logistic "logit" units


def win_prob(score_cp: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-C * score_cp))


def split_by_game(game_ids: np.ndarray, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split by GAME, not by position: positions from one game are highly correlated, so putting
    some in train and some in validation would leak information and give fake-good validation scores."""
    rng = np.random.RandomState(seed)
    unique = np.unique(game_ids)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * val_frac))
    val_games = set(unique[:n_val].tolist())
    is_val = np.array([g in val_games for g in game_ids.tolist()])
    return np.where(~is_val)[0], np.where(is_val)[0]


def evaluate_model(X: np.ndarray, y: np.ndarray, w: np.ndarray, batch: int = 20000) -> Dict[str, float]:
    """Log-loss, mean squared error, and 'sign accuracy' (does the eval favour the real winner?)."""
    n = len(y)
    total_ll, total_se, correct, decisive = 0.0, 0.0, 0, 0
    for i in range(0, n, batch):
        xb = X[i:i + batch].astype(np.float32)
        yb = y[i:i + batch]
        s = xb @ w.astype(np.float32)
        p = np.clip(win_prob(s), 1e-7, 1 - 1e-7)
        total_ll += float(-(yb * np.log(p) + (1 - yb) * np.log(1 - p)).sum())
        total_se += float(((p - yb) ** 2).sum())
        mask = yb != 0.5
        decisive += int(mask.sum())
        correct += int((((s > 0) == (yb > 0.5)) & mask).sum())
    return {
        "logloss": total_ll / n,
        "mse": total_se / n,
        "sign_accuracy": correct / max(decisive, 1),
    }


def train(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
          w0: np.ndarray, epochs: int = 30, batch: int = 1024, lr: float = 1.0, l2: float = 2e-7,
          seed: int = 0, log: Callable[[str], None] = print) -> Tuple[np.ndarray, List[dict]]:
    """Minibatch Adam on the Texel loss. Returns (best weights by validation loss, history).

    lr is in centipawns per step (Adam normalises gradient size), decayed linearly to 10%.
    l2 pulls weights towards w0 (the material-only starting point)."""
    rng = np.random.RandomState(seed)
    w = w0.astype(np.float64).copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    n = len(y_train)
    steps_total = epochs * max(1, math.ceil(n / batch))
    history: List[dict] = []

    # a fixed sample of the training set to monitor training loss cheaply
    mon = rng.choice(n, size=min(n, 20000), replace=False)
    best_w, best_val = w.copy(), float("inf")

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = X_train[idx].astype(np.float32)
            yb = y_train[idx]
            s = xb @ w.astype(np.float32)
            p = win_prob(s)
            grad = (xb.T @ (p - yb)) * (C / len(idx)) + l2 * (w - w0)
            t += 1
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad * grad
            m_hat = m / (1 - b1 ** t)
            v_hat = v / (1 - b2 ** t)
            lr_t = lr * (1.0 - 0.9 * t / steps_total)
            w = w - lr_t * m_hat / (np.sqrt(v_hat) + eps)
        tr = evaluate_model(X_train[mon], y_train[mon], w)
        va = evaluate_model(X_val, y_val, w)
        history.append({"epoch": epoch, "train_logloss": tr["logloss"], "val_logloss": va["logloss"],
                        "val_sign_accuracy": va["sign_accuracy"]})
        log(f"epoch {epoch:3d}  train loss {tr['logloss']:.4f}  val loss {va['logloss']:.4f}  "
            f"val sign-acc {va['sign_accuracy']*100:.1f}%")
        if va["logloss"] < best_val:
            best_val, best_w = va["logloss"], w.copy()
    return best_w, history
