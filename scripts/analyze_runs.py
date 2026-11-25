"""Analyse saved GRPO runs: strict re-scoring, paired tests, calibration,
reasoning audit, and figures.

  python scripts/analyze_runs.py RUNS_DIR OUT_DIR [SUFFIX]

RUNS_DIR must contain win_300steps_<SUFFIX>/ and brier_300steps_<SUFFIX>/ (default v2) as written by
notebooks/train_grpo.ipynb.

Why "strict": the prompt shows the format as
    <pick>HOME</pick> or <pick>AWAY</pick>
and many answers copy that line verbatim. The training-time parser takes the
LAST pick tag, so a copied template was scored as a real pick. Strict scoring
counts an answer only if it has exactly one pick tag naming exactly one side.
"""
import json
import re
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nflgrpo.evaluate import BREAKEVEN, WIN_PAYOUT, ece  # noqa: E402
from nflgrpo.prompts import user_prompt  # noqa: E402

RUNS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
SUFFIX = sys.argv[3] if len(sys.argv) > 3 else "v2"
W, B = RUNS / f"win_300steps_{SUFFIX}", RUNS / f"brier_300steps_{SUFFIX}"
feat = pd.read_csv(ROOT / "data" / "games_features.csv").set_index("game_id")

PICK = re.compile(r"<pick>([^<]{0,60})</pick>", re.I)
SIDE = re.compile(r"\b(HOME|AWAY)\b", re.I)


def strict_pick(text):
    tags = PICK.findall(text)
    sides = {s.upper() for t in tags for s in SIDE.findall(t)}
    if not tags:
        return None, "no pick tag"
    if sides == {"HOME", "AWAY"}:
        return None, "copied both options"
    if len(tags) == 1 and len(sides) == 1:
        return sides.pop(), "single pick"
    return None, "other"


def load(path):
    d = pd.read_json(path, lines=True).set_index("game_id")
    sp = d.output.map(strict_pick)
    d["strict"], d["kind"] = sp.str[0], sp.str[1]
    d = d.join(feat[["spread_line", "push", "season", "week"]])
    d["fav"] = np.where(d.spread_line > 0, "HOME", np.where(d.spread_line < 0, "AWAY", ""))
    return d


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / (1 + z * z / n), (c + h) / (1 + z * z / n)


def summarise(d, col):
    ok = d[col].notna() & ~d.push
    won = (d.loc[ok, col] == d.loc[ok, "label"]).astype(float)
    n, k = int(ok.sum()), int(won.sum())
    lo, hi = wilson(k, n)
    conf = d.loc[ok, "conf"].astype(float)
    cm = conf.notna()
    has_fav = ok & (d.fav != "")
    return {
        "answers_used": int(d[col].notna().sum()), "bets": n,
        "win_rate": 100 * k / n if n else np.nan, "ci_lo": 100 * lo, "ci_hi": 100 * hi,
        "units": k * WIN_PAYOUT - (n - k),
        "home_pick_pct": 100 * (d.loc[d[col].notna(), col] == "HOME").mean(),
        "favorite_pick_pct": 100 * (d.loc[has_fav, col] == d.loc[has_fav, "fav"]).mean(),
        "mean_conf": conf[cm].mean(),
        "brier": ((conf[cm] - won[cm]) ** 2).mean(),
        "ece": ece(conf[cm].values, won[cm].values) if cm.any() else np.nan,
    }


def mcnemar(a, b, d):
    """Exact two-sided McNemar test on games both models bet (non-push)."""
    ok = a.notna() & b.notna() & ~d.push
    wa, wb = (a[ok] == d.label[ok]), (b[ok] == d.label[ok])
    n01, n10 = int((~wa & wb).sum()), int((wa & ~wb).sum())
    n = n01 + n10
    p = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    return {"games": int(ok.sum()), "only_b_right": n01, "only_a_right": n10, "p_value": p}


runs = {
    ("untrained", "val"): load(W / "outputs_untrained_val.jsonl"),
    ("untrained", "test"): load(W / "outputs_untrained_test.jsonl"),
    ("win", "val"): load(W / "outputs_grpo_win_val.jsonl"),
    ("win", "test"): load(W / "outputs_grpo_win_test.jsonl"),
    ("brier", "val"): load(B / "outputs_grpo_brier_val.jsonl"),
    ("brier", "test"): load(B / "outputs_grpo_brier_test.jsonl"),
}

# 1. answer kinds
kinds = pd.DataFrame({f"{m}_{s}": d.kind.value_counts() for (m, s), d in runs.items()}).fillna(0).astype(int)
kinds.to_csv(OUT / "answer_kinds.csv")
print("ANSWER KINDS\n", kinds.to_string(), "\n")

# 2. metrics, lenient (as scored during the run) and strict
rows = {}
for (m, s), d in runs.items():
    rows[(m, s, "as_run")] = summarise(d, "pick")
    rows[(m, s, "strict")] = summarise(d, "strict")
met = pd.DataFrame(rows).T
met.index.names = ["model", "split", "scoring"]
met.round(3).to_csv(OUT / "metrics.csv")
pd.set_option("display.width", 220)
print("METRICS\n", met.round(2).to_string(), "\n")

# 3. baselines restricted to nothing (all games) for reference
for s in ("val", "test"):
    d = runs[("brier", s)]
    nb = ~d.push
    for name, picks in (("always_home", pd.Series("HOME", d.index)),
                        ("always_away", pd.Series("AWAY", d.index)),
                        ("always_favorite", d.fav.replace("", np.nan))):
        ok = nb & picks.notna()
        k, n = int((picks[ok] == d.label[ok]).sum()), int(ok.sum())
        print(f"baseline {s:4s} {name:16s} {100*k/n:5.1f}%  n={n}")
print()

# 4. paired tests
tests = {}
for s in ("val", "test"):
    u, w, b = runs[("untrained", s)], runs[("win", s)], runs[("brier", s)]
    for col in ("pick", "strict"):
        tests[(s, col, "untrained_vs_win")] = mcnemar(u[col], w[col], u)
        tests[(s, col, "untrained_vs_brier")] = mcnemar(u[col], b[col], u)
        tests[(s, col, "win_vs_brier")] = mcnemar(w[col], b[col], u)
tests = pd.DataFrame(tests).T
tests.to_csv(OUT / "paired_tests.csv")
print("PAIRED McNEMAR (a_vs_b)\n", tests.to_string(), "\n")

# 5. does the Brier model just pick the favorite? win rate when it agrees/disagrees
for s in ("val", "test"):
    d = runs[("brier", s)]
    ok = ~d.push & (d.fav != "")
    agree = d.strict[ok] == d.fav[ok]
    won = d.strict[ok] == d.label[ok]
    print(f"brier {s}: with favorite {agree.sum()} bets {100*won[agree].mean():.1f}% | "
          f"against favorite {(~agree).sum()} bets {100*won[~agree].mean():.1f}%")
    # by spread size
    big = d.spread_line[ok].abs() >= 3.5
    print(f"   favorite-pick rate: spread>=3.5 {100*agree[big].mean():.0f}%  spread<3.5 {100*agree[~big].mean():.0f}%")
print()

# 6. confidence: distribution and whether it is informative
for (m, s), d in runs.items():
    if s != "test":
        continue
    ok = d.strict.notna() & ~d.push & d.conf.notna()
    won = (d.strict[ok] == d.label[ok]).astype(float)
    c = d.conf[ok].astype(float)
    r = np.corrcoef(c, won)[0, 1] if c.std() > 0 else np.nan
    top = c.round(2).value_counts().head(4).to_dict()
    print(f"conf {m:9s} n={ok.sum()} corr(conf, won)={r:+.3f} most common={top}")
print()

# 7. reasoning audit
NUM = re.compile(r"(?<![\w.])[-+]?\d+\.\d+|\b\d{2,3}\b")
TEAM = re.compile(r"team a|team b|home team|away team", re.I)
VERB = re.compile(r"cover|more likely to win|will win", re.I)


def claimed_side(text):
    """Side the reasoning says will cover: the team named closest before the
    last 'cover'/'will win' in the same sentence. None if it cannot be read."""
    side = None
    for sent in re.split(r"[.\n]", text):
        v = None
        for v in VERB.finditer(sent):
            pass
        if v is None or re.search(r"\bnot\b|unlikely|n't", sent[:v.end()], re.I):
            continue
        teams = TEAM.findall(sent[:v.start()])
        if teams:
            side = "AWAY" if teams[-1].lower() in ("team a", "away team") else "HOME"
    return side
audit = {}
for (m, s), d in runs.items():
    if s != "test":
        continue
    grounded, total, implied_ok, implied_n, uniq = 0, 0, 0, 0, set()
    for gid, r in d.iterrows():
        text = r.output.split("<pick>")[0]
        uniq.add(re.sub(r"\s+", " ", text.strip()))
        prompt = user_prompt(feat.loc[gid])
        pnums = set(NUM.findall(prompt))
        nums = NUM.findall(text)
        total += len(nums)
        grounded += sum(n.lstrip("+") in {p.lstrip("+") for p in pnums} for n in nums)
        side = claimed_side(text)
        if side and r.strict:
            implied_n += 1
            implied_ok += side == r.strict
    audit[m] = {
        "answers": len(d), "distinct_reasoning_texts": len(uniq),
        "numbers_per_answer": total / len(d),
        "pct_numbers_found_in_prompt": 100 * grounded / total if total else np.nan,
        "answers_with_explicit_claim_and_single_pick": implied_n,
        "pct_pick_matches_claim": 100 * implied_ok / implied_n if implied_n else np.nan,
        "mean_words": d.output.str.split().str.len().mean(),
    }
audit = pd.DataFrame(audit).T
audit.round(2).to_csv(OUT / "reasoning_audit.csv")
print("REASONING AUDIT (test)\n", audit.round(2).to_string(), "\n")

# 8. metamorphic picks: validity and agreement
for d, tag in ((W, "before"), (W, "after"), (B, "after")):
    mp = pd.read_csv(d / f"metamorphic_picks_{tag}.csv", index_col=0)
    base = mp.base
    line = {}
    for c in mp.columns[1:]:
        both = base.notna() & mp[c].notna()
        line[c] = f"valid {100*mp[c].notna().mean():.0f}% / agree-when-valid {100*(mp[c][both]==base[both]).mean():.0f}%"
    print(d.name, tag, line)
print()

# 9. figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, GREY, INK, SUB = "#257de0", "#f47230", "#00b894", "#9c9c9c", "#191919", "#666666"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#cccccc", "axes.labelcolor": SUB, "xtick.color": SUB,
                     "ytick.color": SUB, "text.color": INK, "axes.grid": True, "grid.color": "#eeeeee",
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150})

lw, lb = pd.read_csv(W / "train_log.csv").dropna(subset=["reward"]), pd.read_csv(B / "train_log.csv").dropna(subset=["reward"])


def roll(s):
    return s.rolling(6, min_periods=1).mean()


fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
panels = [("rewards/format_reward/mean", "Format reward (max 0.25)", None),
          (None, "Outcome reward", None),
          ("rewards/home_pick_monitor/mean", "Share of sampled answers picking HOME", 0.5)]
for ax, (col, title, ref) in zip(axes, panels):
    for log, color, name, oc in ((lw, BLUE, "Win reward run", "rewards/win_reward/mean"),
                                 (lb, ORANGE, "Brier reward run", "rewards/brier_reward/mean")):
        y = log[col or oc]
        ax.plot(log.step, y, color=color, alpha=0.25, linewidth=1)
        ax.plot(log.step, roll(y), color=color, linewidth=2, label=name)
    if ref is not None:
        ax.axhline(ref, color=GREY, linewidth=1, linestyle="--")
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("Training step")
axes[0].legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle("Training curves (thin: logged every 5 steps; thick: 30-step rolling mean)", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig_training_curves.png", bbox_inches="tight")

# win rate chart (strict scoring), test + val
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, s, title in zip(axes, ("val", "test"), ("Validation season 2022", "Test seasons 2023-2025")):
    d = runs[("brier", s)]
    nb = ~d.push
    items = []
    for name, picks in (("Always favorite", d.fav.replace("", np.nan)), ("Always home", pd.Series("HOME", d.index))):
        ok = nb & picks.notna()
        k, n = int((picks[ok] == d.label[ok]).sum()), int(ok.sum())
        items.append((name, 100 * k / n, *[100 * v for v in wilson(k, n)], n, GREY))
    for m, label, color in (("untrained", "Untrained", AQUA), ("win", "GRPO win reward", BLUE), ("brier", "GRPO Brier reward", ORANGE)):
        r = met.loc[(m, s, "strict")]
        items.append((label, r.win_rate, r.ci_lo, r.ci_hi, int(r.bets), color))
    for i, (name, p, lo, hi, n, color) in enumerate(items):
        ax.errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt="o", color=color, markersize=8, linewidth=2, capsize=3)
        ax.text(60.5, i, f"{p:.1f}%  (n={n})", va="center", fontsize=9, color=INK)
    ax.axvline(100 * BREAKEVEN, color=INK, linewidth=1, linestyle="--")
    ax.text(100 * BREAKEVEN + 0.3, len(items) - 0.45, "break-even 52.4%", fontsize=8, color=SUB)
    ax.axvline(50, color=GREY, linewidth=1)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([i[0] for i in items], color=INK)
    ax.set_xlim(30, 72)
    ax.set_ylim(-0.6, len(items) - 0.2)
    ax.set_xlabel("Win rate against the spread, with 95% interval")
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", visible=False)
fig.suptitle("Win rate against the spread (answers with exactly one pick only)", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig_win_rates.png", bbox_inches="tight")

# calibration: one panel per model so overlapping points stay readable
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
for ax, (m, label, color) in zip(axes, (("untrained", "Untrained", AQUA), ("win", "GRPO win reward", BLUE), ("brier", "GRPO Brier reward", ORANGE))):
    d = pd.concat([runs[(m, "val")], runs[(m, "test")]])
    ok = d.strict.notna() & ~d.push & d.conf.notna()
    g = pd.DataFrame({"c": (d.conf[ok].astype(float) * 10).round() / 10, "w": (d.strict[ok] == d.label[ok]).astype(float)})
    t = g.groupby("c").w.agg(["mean", "count"])
    t = t[t["count"] >= 10]
    ax.plot([0.4, 1], [0.4, 1], color=GREY, linewidth=1, linestyle="--")
    ax.plot(t.index, t["mean"], color=color, linewidth=2)
    ax.scatter(t.index, t["mean"], s=np.clip(t["count"] / 2, 30, 260), color=color, edgecolor="white", linewidth=1.5, zorder=3)
    for c, r in t.iterrows():
        ax.text(c, r["mean"] + 0.045, f"n={int(r['count'])}", ha="center", fontsize=8, color=SUB)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel("Stated confidence (rounded to 0.1)")
    ax.set_title(f"{label}  ({int(ok.sum())} single-pick bets)", loc="left", fontsize=10, color=INK)
axes[0].set_ylabel("Share of those picks that covered")
axes[0].text(0.83, 0.88, "perfect", fontsize=8, color=SUB, rotation=40)
fig.suptitle("Calibration on validation + test: stated confidence against actual win rate", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig_calibration.png", bbox_inches="tight")
print("wrote figures and CSVs to", OUT)
