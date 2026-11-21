"""Score the non-LLM baselines on the validation and test seasons.

  python scripts/run_baselines.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nflgrpo import baselines as B  # noqa: E402
from nflgrpo.data import split  # noqa: E402
from nflgrpo.evaluate import BREAKEVEN, score, table  # noqa: E402

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "games_features.csv")
train, val, test = split(df)
model, C, names = B.fit_logreg(train, val)
print(f"logistic regression: C={C}, {len(names)} features, trained on {len(train)} games")
print(f"break-even win rate at -110: {BREAKEVEN:.2%}\n")

cols = ["n_bets", "win_rate", "win_rate_lo", "win_rate_hi", "units", "roi",
        "beats_breakeven", "home_pick_rate", "favorite_pick_rate", "brier", "ece"]
pd.set_option("display.width", 200)
for name, part in (("validation 2022", val), ("test 2023-2025", test)):
    lp, lc = B.logreg_picks(model, part)
    res = {
        "always_home": score(part, B.always_home(part)),
        "always_favorite": score(part, B.always_favorite(part)),
        "logistic_regression": score(part, lp, lc),
    }
    t = table(res)
    print(name)
    print(t[[c for c in cols if c in t]].round(2).to_string(), "\n")
    t.to_csv(ROOT / "data" / f"baselines_{name.split()[0]}.csv")
