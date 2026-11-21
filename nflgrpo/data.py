"""Build the pre-game dataset: one row per NFL regular-season game with both
teams' season-to-date stats (prior games only) and the closing point spread.

Sources (nflverse, free, no API key):
  games.csv             results + spread_line for every game since 1999
  stats_team_week_Y.csv weekly team box-score stats

Spread convention (nflverse): spread_line > 0 means the HOME team is favored
by that many points. result = home_score - away_score.
Home covers  <=> result > spread_line.   Push <=> result == spread_line.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
TEAM_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_team/stats_team_week_{season}.csv"
)
# games.csv keeps historical abbreviations; team stats use current ones.
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# stat name -> (how it is shown in the prompt, decimals)
STAT_LABELS = {
    "ppg": ("points/game", 1),
    "papg": ("points allowed/game", 1),
    "pass_ypg": ("pass yds/game", 0),
    "rush_ypg": ("rush yds/game", 0),
    "pass_ypg_allowed": ("pass yds allowed/game", 0),
    "rush_ypg_allowed": ("rush yds allowed/game", 0),
    "to_margin_pg": ("turnover margin/game", 2),
    "sacks_pg": ("sacks/game", 1),
    "sacks_allowed_pg": ("sacks allowed/game", 1),
    "off_epa_pg": ("offensive EPA/game", 1),
}
STATS = list(STAT_LABELS)


def _fetch(url: str, cache: Path) -> pd.DataFrame:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            cache.write_bytes(r.read())
    return pd.read_csv(io.BytesIO(cache.read_bytes()), low_memory=False)


def load_games(cache_dir: Path) -> pd.DataFrame:
    g = _fetch(GAMES_URL, cache_dir / "games.csv")
    g = g[(g.game_type == "REG") & g.result.notna() & g.spread_line.notna()].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_FIX)
    return g


def load_team_stats(seasons, cache_dir: Path) -> pd.DataFrame:
    frames = [
        _fetch(TEAM_STATS_URL.format(season=s), cache_dir / f"stats_team_week_{s}.csv")
        for s in seasons
    ]
    t = pd.concat(frames, ignore_index=True)
    t = t[t.season_type == "REG"].copy()
    t["giveaways"] = (
        t.passing_interceptions + t.sack_fumbles_lost
        + t.rushing_fumbles_lost + t.receiving_fumbles_lost
    )
    t["takeaways"] = t.def_interceptions + t.fumble_recovery_opp
    t["off_epa"] = t.passing_epa.fillna(0) + t.rushing_epa.fillna(0)
    keep = ["season", "week", "team", "passing_yards", "rushing_yards", "giveaways",
            "takeaways", "def_sacks", "sacks_suffered", "off_epa"]
    return t[keep]


def team_game_table(games: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (team, game) with that game's raw numbers."""
    def side(team, opp, sign):
        d = pd.DataFrame({
            "game_id": games.game_id, "season": games.season, "week": games.week,
            "team": games[f"{team}_team"], "opp": games[f"{opp}_team"],
            "pf": games[f"{team}_score"], "pa": games[f"{opp}_score"],
        })
        ats = sign * (games.result - games.spread_line)  # >0: this team covered
        d["win"] = (d.pf > d.pa).astype(float)
        d["loss"] = (d.pf < d.pa).astype(float)
        d["ats_win"] = (ats > 0).astype(float)
        d["ats_loss"] = (ats < 0).astype(float)
        return d

    tg = pd.concat([side("home", "away", 1), side("away", "home", -1)], ignore_index=True)
    tg = tg.merge(stats, on=["season", "week", "team"], how="left", validate="1:1")
    allowed = stats.rename(columns={
        "team": "opp", "passing_yards": "pass_allowed", "rushing_yards": "rush_allowed",
    })[["season", "week", "opp", "pass_allowed", "rush_allowed"]]
    tg = tg.merge(allowed, on=["season", "week", "opp"], how="left", validate="1:1")
    tg["to_margin"] = tg.takeaways - tg.giveaways
    return tg.sort_values(["season", "team", "week"]).reset_index(drop=True)


def season_to_date(tg: pd.DataFrame) -> pd.DataFrame:
    """Per (team, game): stats from that team's EARLIER games this season only.

    The shift(1) below is the leakage guard: a game's own numbers never enter
    its own features. tests in scripts/check_leakage.py verify this.
    """
    grp = tg.groupby(["season", "team"], sort=False)
    out = tg[["game_id", "team"]].copy()
    out["games_played"] = grp.cumcount()

    def prior_mean(col):
        return grp[col].transform(lambda s: s.shift(1).expanding().mean())

    def prior_sum(col):
        return grp[col].transform(lambda s: s.shift(1).expanding().sum()).fillna(0)

    for name, col in [
        ("ppg", "pf"), ("papg", "pa"), ("pass_ypg", "passing_yards"),
        ("rush_ypg", "rushing_yards"), ("pass_ypg_allowed", "pass_allowed"),
        ("rush_ypg_allowed", "rush_allowed"), ("to_margin_pg", "to_margin"),
        ("sacks_pg", "def_sacks"), ("sacks_allowed_pg", "sacks_suffered"),
        ("off_epa_pg", "off_epa"),
    ]:
        out[name] = prior_mean(col)
    for name, col in [("wins", "win"), ("losses", "loss"),
                      ("ats_wins", "ats_win"), ("ats_losses", "ats_loss")]:
        out[name] = prior_sum(col).astype(int)
    out["ties"] = out.games_played - out.wins - out.losses
    return out


def build_dataset(first_season=2010, last_season=2025, min_games=3,
                  cache_dir="data/raw") -> pd.DataFrame:
    """One row per game. Drops games where either team has played fewer than
    `min_games` games this season (too little season-to-date signal)."""
    cache_dir = Path(cache_dir)
    games = load_games(cache_dir)
    games = games[games.season.between(first_season, last_season)]
    stats = load_team_stats(sorted(games.season.unique()), cache_dir)
    std = season_to_date(team_game_table(games, stats))

    cols = [c for c in std.columns if c not in ("game_id", "team")]
    df = games[["game_id", "season", "week", "gameday", "home_team", "away_team",
                "home_score", "away_score", "result", "spread_line", "total_line",
                "home_rest", "away_rest", "div_game"]].copy()
    for side in ("home", "away"):
        s = std.rename(columns={c: f"{side}_{c}" for c in cols})
        df = df.merge(s, left_on=["game_id", f"{side}_team"],
                      right_on=["game_id", "team"], how="left").drop(columns="team")

    df = df[(df.home_games_played >= min_games) & (df.away_games_played >= min_games)]
    df["margin_vs_spread"] = df.result - df.spread_line
    df["push"] = df.margin_vs_spread == 0
    df["home_cover"] = (df.margin_vs_spread > 0).astype(int)
    df["label"] = df.home_cover.map({1: "HOME", 0: "AWAY"}).where(~df.push, "PUSH")
    return df.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def split(df: pd.DataFrame, train_end=2021, val=(2022,), test=(2023, 2024, 2025)):
    return (df[df.season <= train_end], df[df.season.isin(val)], df[df.season.isin(test)])
