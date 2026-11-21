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

_PICK = re.compile(r"<pick>\s*(HOME|AWAY)\s*</pick>", re.I)
_CONF = re.compile(r"<confidence>\s*([01](?:\.\d+)?)\s*</confidence>", re.I)
_REASON = re.compile(r"<reasoning>(.+?)</reasoning>", re.S | re.I)


def parse_output(text: str):
    """Returns (pick, confidence, valid). pick/confidence are None if missing.
    valid = exactly one pick, one confidence in [0,1], non-empty reasoning."""
    picks, confs = _PICK.findall(text), _CONF.findall(text)
    pick = picks[-1].upper() if picks else None
    conf = float(confs[-1]) if confs else None
    if conf is not None and not 0.0 <= conf <= 1.0:
        conf = None
    reason = _REASON.search(text)
    valid = (len(picks) == 1 and len(confs) == 1 and conf is not None
             and reason is not None and len(reason.group(1).strip()) >= 20)
    return pick, conf, valid


def _text(completion):
    return completion[0]["content"] if isinstance(completion, list) else completion


def format_reward(prompts, completions, **kw):
    """Small bonus for a fully well-formed answer."""
    return [0.25 if parse_output(_text(c))[2] else 0.0 for c in completions]


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


REWARD_SETS = {
    "win": [format_reward, win_reward],
    "brier": [format_reward, brier_reward],
}
