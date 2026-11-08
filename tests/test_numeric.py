"""Tests that need only numpy (no python-chess)."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chessbot.stats import elo_from_score, summarize  # noqa: E402
from chessbot.training import C, evaluate_model, split_by_game, train, win_prob  # noqa: E402


def test_elo_from_score():
    assert abs(elo_from_score(0.5)) < 1e-9
    assert abs(elo_from_score(0.64) - 100) < 5
    assert elo_from_score(0.7) > elo_from_score(0.6) > 0 > elo_from_score(0.4)


def test_summarize_counts_and_ci():
    r = summarize([1, 1, 0.5, 0, 1, 0.5, 0, 1])
    assert (r["wins"], r["draws"], r["losses"], r["games"]) == (4, 2, 2, 8)
    assert r["elo_lo"] < r["elo"] < r["elo_hi"]


def test_win_prob_is_elo_formula():
    assert abs(win_prob(np.array([0.0]))[0] - 0.5) < 1e-12
    assert abs(win_prob(np.array([400.0]))[0] - 10 / 11) < 1e-9  # +400 cp  ->  ~90.9%


def test_split_by_game_has_no_leakage():
    gid = np.repeat(np.arange(50), 6)
    tr, va = split_by_game(gid, 0.2, seed=1)
    assert set(gid[tr]).isdisjoint(set(gid[va]))
    assert len(tr) + len(va) == len(gid)


def test_training_recovers_signal():
    """Synthetic problem: results are generated from known weights; training must beat the start."""
    rng = np.random.RandomState(0)
    n_feat, n = 20, 30000
    w_true = rng.normal(0, 100, n_feat)
    X = rng.randint(-1, 2, size=(n, n_feat)).astype(np.int8)
    y = (rng.rand(n) < win_prob(X.astype(np.float64) @ w_true)).astype(np.float32)
    X_tr, y_tr, X_va, y_va = X[:25000], y[:25000], X[25000:], y[25000:]
    w0 = np.zeros(n_feat)
    before = evaluate_model(X_va, y_va, w0)
    w, hist = train(X_tr, y_tr, X_va, y_va, w0, epochs=15, batch=512, lr=2.0, l2=0.0,
                    log=lambda s: None)
    after = evaluate_model(X_va, y_va, w)
    assert after["logloss"] < before["logloss"] - 0.05
    assert after["sign_accuracy"] > before["sign_accuracy"]
    assert np.corrcoef(w, w_true)[0, 1] > 0.9


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
