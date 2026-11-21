"""Non-LLM baselines: always-favorite, always-home, logistic regression."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import STATS


def always_home(df):
    return ["HOME"] * len(df)


def always_favorite(df):
    """Pick'em games (spread 0) have no favorite and are left unbet."""
    return [("HOME" if s > 0 else "AWAY" if s < 0 else None) for s in df.spread_line]


def features(df):
    """Home-minus-away differences of the same stats the LLM sees, plus the spread."""
    X = {f"d_{s}": df[f"home_{s}"] - df[f"away_{s}"] for s in STATS}
    for s in ("wins", "losses", "ats_wins", "ats_losses"):
        X[f"d_{s}_rate"] = (df[f"home_{s}"] / df.home_games_played
                            - df[f"away_{s}"] / df.away_games_played)
    X["d_rest"] = df.home_rest - df.away_rest
    X["div_game"] = df.div_game
    X["spread_line"] = df.spread_line
    return np.column_stack(list(X.values())).astype(float), list(X)


def fit_logreg(train, val, Cs=(0.001, 0.01, 0.1, 1, 10)):
    """Fit on train (pushes dropped), choose regularisation on val by log loss."""
    tr, va = train[~train.push], val[~val.push]
    Xtr, names = features(tr)
    Xva, _ = features(va)
    best = None
    for C in Cs:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        m.fit(Xtr, tr.home_cover)
        p = np.clip(m.predict_proba(Xva)[:, 1], 1e-6, 1 - 1e-6)
        ll = -np.mean(va.home_cover * np.log(p) + (1 - va.home_cover) * np.log(1 - p))
        if best is None or ll < best[0]:
            best = (ll, C, m)
    return best[2], best[1], names


def logreg_picks(model, df):
    p_home = model.predict_proba(features(df)[0])[:, 1]
    picks = np.where(p_home >= 0.5, "HOME", "AWAY").tolist()
    confs = np.maximum(p_home, 1 - p_home).tolist()
    return picks, confs
