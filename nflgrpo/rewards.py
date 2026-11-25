"""Output parsing and GRPO reward functions.

TRL calls each reward function as f(prompts, completions, **dataset_columns)
and expects one float per completion. `label` is a dataset column:
"HOME", "AWAY" or "PUSH".

Design note: an unparseable answer must never score better than a wrong one,
otherwise the model learns to stop answering. So wrong = 0 and invalid = 0,
and the format bonus is only paid for a fully valid answer.
"""
from __future__ import annotations

import re

_PICK = re.compile(r"<pick>([^<]{0,60})</pick>", re.I)
_SIDE = re.compile(r"\b(HOME|AWAY)\b", re.I)
_CONF = re.compile(r"<confidence>\s*(\d*\.?\d+)\s*(%?)\s*</confidence>", re.I)
_REASON = re.compile(r"<reasoning>(.+?)</reasoning>", re.S | re.I)


def _pick(text):
    """The side picked, or None. Every <pick> tag in the answer must name the
    same single side. An answer that copies the prompt's two options
    ("<pick>HOME</pick> or <pick>AWAY</pick>") names both, so it is not a pick."""
    tags = _PICK.findall(text)
    if not tags:
        return None, 0
    sides = {m.upper() for t in tags for m in _SIDE.findall(t)}
    return (sides.pop() if len(sides) == 1 else None), len(tags)


def _conf(text):
    found = _CONF.findall(text)
    if not found:
        return None, 0
    num, pct = found[-1]
    v = float(num) / (100 if pct else 1)
    return (v if 0.0 <= v <= 1.0 else None), len(found)


def parse_output(text: str):
    """Returns (pick, confidence, valid). pick/confidence are None if missing.
    valid = exactly one pick tag, one confidence tag, and a reasoning block."""
    pick, n_pick = _pick(text)
    conf, n_conf = _conf(text)
    reason = _REASON.search(text)
    valid = (pick is not None and conf is not None and n_pick == 1 and n_conf == 1
             and reason is not None and len(reason.group(1).strip()) >= 20)
    return pick, conf, valid


def _text(completion):
    return completion[0]["content"] if isinstance(completion, list) else completion


def format_reward(prompts, completions, **kw):
    """Graded, so there is something to learn from even when most answers are
    malformed: 0.10 for a usable pick, 0.10 for a usable confidence, and the
    full 0.25 only when the whole answer is well formed."""
    out = []
    for c in completions:
        pick, conf, valid = parse_output(_text(c))
        out.append(0.25 if valid else 0.10 * (pick is not None) + 0.10 * (conf is not None))
    return out


def win_reward(prompts, completions, label, **kw):
    """Version A: 1 if the pick covered, else 0. Pushes pay 0.5 to every valid pick."""
    out = []
    for c, y in zip(completions, label):
        pick, _, _ = parse_output(_text(c))
        if pick is None:
            out.append(0.0)
        elif y == "PUSH":
            out.append(0.5)
        else:
            out.append(1.0 if pick == y else 0.0)
    return out


def brier_reward(prompts, completions, label, **kw):
    """Version B: 1 - (confidence - outcome)^2, a proper scoring rule in [0, 1].
    Always answering 0.5 earns 0.75; the only way to beat that in expectation
    is confidence that actually tracks how often the pick wins."""
    out = []
    for c, y in zip(completions, label):
        pick, conf, _ = parse_output(_text(c))
        if pick is None or conf is None:
            out.append(0.0)
        elif y == "PUSH":
            out.append(0.75)
        else:
            out.append(1.0 - (conf - float(pick == y)) ** 2)
    return out


def home_pick_monitor(prompts, completions, **kw):
    """Not a reward (used with weight 0). Logs the share of sampled answers that
    pick HOME at each step, so a collapse to one side shows up in the training log."""
    return [1.0 if parse_output(_text(c))[0] == "HOME" else 0.0 for c in completions]


REWARD_SETS = {
    "win": [format_reward, win_reward],
    "brier": [format_reward, brier_reward],
}
