"""Download nflverse data and write the pre-game prompt dataset.

  python scripts/build_dataset.py            # 2010-2025, writes data/*.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nflgrpo.data import build_dataset, split  # noqa: E402
from nflgrpo.prompts import messages  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--first", type=int, default=2010)
ap.add_argument("--last", type=int, default=2025)
ap.add_argument("--min-games", type=int, default=3)
ap.add_argument("--out", default=str(ROOT / "data"))
a = ap.parse_args()

out = Path(a.out)
df = build_dataset(a.first, a.last, a.min_games, cache_dir=out / "raw")
df.to_csv(out / "games_features.csv", index=False)

for name, part in zip(("train", "val", "test"), split(df)):
    with open(out / f"{name}.jsonl", "w") as f:
        for _, r in part.iterrows():
            f.write(json.dumps({"game_id": r.game_id, "prompt": messages(r),
                                "label": r.label}) + "\n")
    cover = part[~part.push].home_cover.mean()
    print(f"{name:5s} seasons {part.season.min()}-{part.season.max()}  "
          f"{len(part):4d} games  {int(part.push.sum()):3d} pushes  home covers {cover:.1%}")
