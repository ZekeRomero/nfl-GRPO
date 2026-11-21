# Training a small LLM with GRPO to make NFL betting decisions

Zeke Romero, Nathan Khazam, Dukhwan Kim

Can GRPO teach a small LLM useful reasoning when the reward is extremely noisy?
We fine-tune a small Qwen model with LoRA + GRPO to pick NFL games against the
spread, and compare it with the untrained model and simple baselines on seasons
it never trained on.

## Quick start

```bash
pip install -r requirements.txt
python scripts/build_dataset.py     # downloads nflverse data, writes data/*.jsonl
python scripts/check_leakage.py     # verifies features only use earlier games
python scripts/run_baselines.py     # always-home, always-favorite, logistic regression
```

Training runs on a free Colab T4: open `notebooks/train_grpo.ipynb`, set
`REPO_URL`, run all. Run it once with `REWARD = "win"` and once with `"brier"`.

## Layout

| File | What it does |
| --- | --- |
| `nflgrpo/data.py` | Downloads nflverse games + weekly team stats, builds season-to-date features |
| `nflgrpo/prompts.py` | Row -> chat prompt; metamorphic prompt variants |
| `nflgrpo/rewards.py` | Output parser and the two GRPO reward sets (`win`, `brier`) |
| `nflgrpo/baselines.py` | Always-home, always-favorite, logistic regression |
| `nflgrpo/evaluate.py` | Win rate, profit at -110, Brier, ECE, bootstrap CIs, shortcut and consistency checks |
| `notebooks/train_grpo.ipynb` | Untrained eval -> GRPO training -> trained eval, on Colab |

## Design decisions

- **Split:** train 2010-2021, tune on 2022, test on 2023-2025. Regular season only.
- **No leakage:** every stat is the mean of that team's *earlier* games in the same
  season. Games where either team has played fewer than 3 games are dropped
  (`--min-games`).
- **Spread:** nflverse `spread_line` (positive = home favored). Home covers when
  `home_score - away_score > spread_line`. Pushes are excluded from win rate and
  profit.
- **Anonymised prompts:** teams appear as "Team A at Team B" with no season year,
  so the model cannot answer by recalling the real game. The `real_names_season`
  variant measures how much it would have gained from recall.
- **Rewards:** wrong and unparseable answers both score 0, so the model is never
  paid for refusing to answer. `win` = 1 if the pick covered. `brier` =
  1 - (confidence - outcome)^2.
- **Metamorphic tests:** renaming the teams, swapping the labels, or reordering
  the stat blocks must not change the pick. `evaluate.consistency` reports how
  often it does.
- **Break-even:** 52.38% at -110. A result "beats the market" only if the lower
  end of its 95% bootstrap interval is above that.
