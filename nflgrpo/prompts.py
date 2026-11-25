"""Turn a dataset row into a chat prompt, plus metamorphic variants.

By default team names and the season year are hidden ("Team A at Team B").
Reason: a pretrained LLM may simply remember who won "KC at BUF, week 11,
2023". Anonymising forces it to use the stats, and makes the 2023-2025 test
seasons a fair test even if they are inside the model's training data.
"""
from __future__ import annotations

from .data import STAT_LABELS

# Keep the whole prompt under 512 tokens: Unsloth's GRPO generation path crashes
# (CUDA illegal memory access) on longer prompts. The notebook asserts this.
SYSTEM_PROMPT = """You are an NFL betting analyst. Given a game, the point spread and each team's season-to-date stats, decide which side covers the spread.

Answer in under 100 words, in exactly this format:
<reasoning>brief analysis</reasoning>
<pick>HOME</pick> or <pick>AWAY</pick>
<confidence>probability your pick covers, 0 to 1</confidence>"""

REMINDER = "Which side covers the spread? Be brief; end with <pick> and <confidence>."


def _fmt(v, nd):
    return f"{v:+.{nd}f}" if nd == 2 else f"{v:.{nd}f}"


def _team_block(row, side, name):
    rec = f"{row[f'{side}_wins']}-{row[f'{side}_losses']}"
    if row[f"{side}_ties"]:
        rec += f"-{row[f'{side}_ties']}"
    ats = f"{row[f'{side}_ats_wins']}-{row[f'{side}_ats_losses']}"
    lines = [f"{name} ({side}): record {rec}, against the spread {ats}"]
    for stat, (label, nd) in STAT_LABELS.items():
        lines.append(f"  {label}: {_fmt(row[f'{side}_{stat}'], nd)}")
    return "\n".join(lines)


def spread_text(spread, home, away):
    if spread > 0:
        return f"{home} -{spread:g} (home team favored by {spread:g})"
    if spread < 0:
        return f"{away} -{-spread:g} (away team favored by {-spread:g})"
    return "Pick'em (no favorite)"


def user_prompt(row, anonymize=True, names=("Team A", "Team B"),
                home_first=False, show_season=False) -> str:
    """names = (away_name, home_name) when anonymize=True."""
    away, home = names if anonymize else (row["away_team"], row["home_team"])
    when = f"Week {row['week']}" + (f", {row['season']} season" if show_season else "")
    blocks = [_team_block(row, "away", away), _team_block(row, "home", home)]
    if home_first:
        blocks.reverse()
    return "\n".join([
        f"{when}. {away} (away) at {home} (home).",
        f"Spread: {spread_text(row['spread_line'], home, away)}",
        f"Rest days: {away} {int(row['away_rest'])}, {home} {int(row['home_rest'])}",
        f"Division game: {'yes' if row['div_game'] else 'no'}",
        "",
        "Season-to-date stats (games before this one only):",
        *blocks,
        "",
        REMINDER,
    ])


def messages(row, **kw):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(row, **kw)}]


# Metamorphic variants: the correct pick is identical for every one of these,
# so a model that reasons from the stats should give the same HOME/AWAY answer.
VARIANTS = {
    "base": {},
    "renamed": {"names": ("Team X", "Team Y")},           # labels changed
    "names_swapped": {"names": ("Team B", "Team A")},      # labels swapped
    "home_first": {"home_first": True},                   # stat block order flipped
    "real_names": {"anonymize": False},                   # real team abbreviations
    "real_names_season": {"anonymize": False, "show_season": True},  # memorisation probe
}


def variant_messages(row):
    return {k: messages(row, **kw) for k, kw in VARIANTS.items()}
