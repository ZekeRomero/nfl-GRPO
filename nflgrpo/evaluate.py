"""Metrics shared by the baselines and the LLM: win rate against the spread,
profit at -110, calibration, bootstrap confidence intervals, shortcut and
metamorphic-consistency checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

BREAKEVEN = 110 / 210  # 52.38% at -110
WIN_PAYOUT = 100 / 110


def score(df: pd.DataFrame, picks, confs=None, n_boot=5000, seed=0) -> dict:
    """picks: 'HOME'/'AWAY'/None per row of df (None = no valid answer, counted
    as no bet and reported in `invalid`). confs: probability the pick covers."""
    picks = np.array([p if p in ("HOME", "AWAY") else "" for p in picks])
    valid = picks != ""
    bet = valid & ~df.push.values           # pushes refund the stake
    y = df.label.values
    won = (picks == y)[bet].astype(float)
    n = int(bet.sum())
    out = {"n_games": len(df), "invalid": int((~valid).sum()),
           "pushes": int((valid & df.push.values).sum()), "n_bets": n}
    if n == 0:
        return out
    rng = np.random.default_rng(seed)
    boot = rng.choice(won, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    wr = won.mean()
    out.update({
        "win_rate": wr, "win_rate_lo": lo, "win_rate_hi": hi,
        "units": won.sum() * WIN_PAYOUT - (n - won.sum()),
        "roi": wr * WIN_PAYOUT - (1 - wr),
        "roi_lo": lo * WIN_PAYOUT - (1 - lo), "roi_hi": hi * WIN_PAYOUT - (1 - hi),
        "beats_breakeven": bool(lo > BREAKEVEN),
        # shortcut diagnostics (research question 4)
        "home_pick_rate": (picks[valid] == "HOME").mean(),
        "favorite_pick_rate": _favorite_rate(df[valid], picks[valid]),
    })
    if confs is not None:
        c = np.array([np.nan if v is None else v for v in confs], dtype=float)[bet]
        ok = ~np.isnan(c)
        if ok.any():
            out["brier"] = float(np.mean((c[ok] - won[ok]) ** 2))
            out["ece"] = ece(c[ok], won[ok])
            out["mean_conf"] = float(c[ok].mean())
    return out


def _favorite_rate(df, picks):
    fav = np.where(df.spread_line.values > 0, "HOME",
                   np.where(df.spread_line.values < 0, "AWAY", ""))
    has_fav = fav != ""
    return float((picks[has_fav] == fav[has_fav]).mean()) if has_fav.any() else np.nan


def ece(conf, won, bins=10):
    """Expected calibration error: |stated confidence - actual win rate|,
    averaged over confidence bins weighted by how many picks fall in each."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(conf[idx == b].mean() - won[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def consistency(picks_by_variant: dict) -> dict:
    """picks_by_variant: {variant_name: [pick per game]}, must include 'base'.
    Returns, per variant, the share of games where the pick matches base."""
    base = np.array([p or "" for p in picks_by_variant["base"]])
    return {k: float((np.array([p or "" for p in v]) == base).mean())
            for k, v in picks_by_variant.items() if k != "base"}


def table(results: dict) -> pd.DataFrame:
    t = pd.DataFrame(results).T
    pct = [c for c in t.columns if c.startswith(("win_rate", "roi")) or c.endswith("_rate")]
    t[pct] = t[pct].astype(float) * 100
    return t
