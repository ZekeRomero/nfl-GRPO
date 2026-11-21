"""Sanity checks: no feature uses the game's own result, and rewards behave.

  python scripts/check_leakage.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nflgrpo.data import load_games, load_team_stats, season_to_date, team_game_table  # noqa: E402
from nflgrpo.rewards import brier_reward, format_reward, parse_output, win_reward  # noqa: E402

cache = ROOT / "data" / "raw"
games = load_games(cache)
games = games[games.season.between(2022, 2023)]
stats = load_team_stats([2022, 2023], cache)
tg = team_game_table(games, stats)
base = season_to_date(tg)

# 1. Scramble every team's LAST game of each season. Nothing before it may change.
tg2 = tg.copy()
last = tg2.groupby(["season", "team"]).tail(1).index
for c in ("pf", "pa", "passing_yards", "rushing_yards", "win", "loss", "ats_win", "off_epa"):
    tg2.loc[last, c] = 999
changed = season_to_date(tg2)
num = base.select_dtypes("number").columns
diff = (base[num].fillna(-1).values != changed[num].fillna(-1).values).any(axis=1)
assert not diff.any(), "features changed when only the final game's result changed"

# 2. A game's own row: features must equal the mean of strictly earlier games.
t = tg[(tg.season == 2023) & (tg.team == "KC")].reset_index(drop=True)
b = base.loc[tg[(tg.season == 2023) & (tg.team == "KC")].index].reset_index(drop=True)
for i in (1, 5, 10):
    assert np.isclose(b.ppg[i], t.pf[:i].mean()) and b.games_played[i] == i
assert tg.passing_yards.notna().all(), "team stats failed to merge for some games"
print("leakage checks passed")

# 3. Rewards
good = "<reasoning>\nHome offense is far better and well rested.\n</reasoning>\n<pick>HOME</pick>\n<confidence>0.6</confidence>"
assert parse_output(good) == ("HOME", 0.6, True)
assert parse_output("HOME")[0] is None
c = [good, good.replace("HOME", "AWAY"), "I like the home team", good]
lab = ["HOME", "HOME", "HOME", "PUSH"]
assert format_reward(None, c) == [0.25, 0.25, 0.0, 0.25]
assert win_reward(None, c, lab) == [1.0, 0.0, 0.0, 0.5]
assert np.allclose(brier_reward(None, c, lab), [0.84, 0.64, 0.0, 0.75])
print("reward checks passed")
