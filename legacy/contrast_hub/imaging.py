"""Imaging-side reading of a corpus contrast into a hub signature.

Input per contrast (key_coverage.py's contrasts file): the ns-pond key, the side names from analysis_features
(`contrast_positive` / `contrast_negative`: a cell's level name, then the names of the conditions it carries, joined
by " | "), the paper, tasks and families. Per paper: the abbreviation dictionary (abbrev.py) and the record's condition
glossary (glossary.py: names, descriptions, kinds).

Order, first that yields a signature (each returns (signature, how, sign)):
0. a high-confidence Codex reading of the contrast (codex_normalize.py), which saw the names, the paper's condition
   descriptions and abbreviations;
1. the side names: each with the paper's abbreviations spelled out (never an expansion that names an effect, score
   or measure), normalized in the task's family, with the paper and family lexicons where the atoms find nothing.
   A side's atoms are its level name's; when that has none, those its cells share (hub.combine). A name without
   atoms falls back to its glossary condition's kind and description head, keeping no atom on an incidental axis;
2. the ns-pond key's tokens, with the incidental axes left out (when the names gave nothing, the key's atoms came
   from descriptions or stimulus content, where side, speed, accuracy and level are rarely the manipulation). A
   baseline marked `:reversed` means the reported direction is baseline > target: sign -1;
3. task level, for a real task id (not a family name): the task itself against a rest-like baseline is
   `task:<id>|@task>@rest`, and with no other side `task:<id>|@task` (a simple effect, never established). A task
   against a matched control is left unnormalized: the control is a condition to be read.
"""
import re

import hub as H

DESC_END = re.compile(r"[:;(]|\. |\b(relative to|compared (to|with)|versus|vs\.?|than|in contrast|following|preceded by|after|before)\b", re.I)
#: axes whose atoms in a description or stimulus list are rarely what a contrast manipulates
INCIDENTAL = {"side", "speed", "accuracy", "level", "engagement", "agent", "modality", "delay"}
#: abbreviation expansions that name a measure or effect, not a condition
NOT_CONDITION_LF = re.compile(r"\b(effect|score|index|rate|time|times|task|test|scale|questionnaire|inventory|ratio|latency|amplitude)s?$", re.I)


def desc_head(d):
    d = (d or "").strip()
    parts = d.split(":", 1) if ":" in d[:60] else [d]
    out = []
    for p in parts:
        m = DESC_END.search(p.strip())
        out.append((p.strip()[:m.start()] if m else p.strip())[:160])
    return out


def expand(name, abbr):
    if not abbr:
        return name

    def rep(m):
        w = m.group(0)
        lf = abbr.get(w) or abbr.get(w.upper())
        return lf if lf and not NOT_CONDITION_LF.search(lf) else w
    return re.sub(r"[A-Za-z][A-Za-z0-9+\-]{0,9}", rep, name)


class Paper:
    def __init__(self, abbr, gloss):
        self.abbr = abbr or {}
        self.cond = {}
        for t in (gloss or {}).get("tasks", []):
            for c in t["conditions"]:
                self.cond.setdefault(H.fold(c["name"]), c)


def name_side(name, task, fam, paper, lex):
    s = H.side_from_text(expand(name, paper.abbr), task, fam, lex)
    c = paper.cond.get(H.fold(name))
    if s["kind"] == "base" and c and c["kind"] == "control_state" and not H.LOWLEVEL.search(c["description"][:160]):
        return H.side(kind="base", base="control")        # named "baseline" but described as a matched control
    if s["atoms"] or s["kind"] in ("base", "not_a_condition"):
        return s
    if c:
        b = H.base_kind(c["kind"], c["description"][:160] or name)
        if b:
            return H.side(kind="base", base=b)
        for h in desc_head(c["description"]):
            d = H.side_from_text(expand(h, paper.abbr), task, fam, None)
            at = {k: v for k, v in d["atoms"].items() if k not in INCIDENTAL}
            if at:
                return H.side(at)
    return s


def names_side(field, task, fam, paper, lex):
    names = [n.strip() for n in field.split(" | ") if n.strip()]
    if not names:
        return None
    sides = [name_side(n, task, fam, paper, lex) for n in names]
    if sides[0]["atoms"] or len(sides) == 1:
        return sides[0]
    return H.combine(sides[1:]) if any(s["atoms"] for s in sides[1:]) else H.combine(sides)


def key_sides(key, lex):
    fam, tgt, base = key.split("|", 2)
    sign = 1
    if base.endswith(":reversed"):
        base, sign = base[: -len(":reversed")], -1
    T = H.side_from_tokens([x for x in tgt.split(";") if x and x != "-"], fam, lex, INCIDENTAL)
    if base == "-":
        B = None
    else:
        body = base[3:] if base.startswith("vs:") else base.replace("+", ";")
        B = H.side_from_tokens([x for x in body.split(";") if x], fam, lex, INCIDENTAL)
    return fam, T, B, sign


def from_override(o, fam, task, family_ids):
    """a Codex contrast reading (codex_normalize.py) -> (signature, how, sign)"""
    if o["type"] == "not_conditions":
        return None, "not_a_condition", 1
    if o["type"] == "interaction":
        return None, "interaction", 1
    if o["type"] == "task_vs_base":
        if task and task not in family_ids:
            return (f"task:{task}", (("@task", H.TASK, H.REST),)), "override_task", 1
        return None, "task_family_id", 1
    T = H.side(dict(x.split(":", 1) for x in o["target"]))
    if o["baseline_special"] == "@base":
        B = H.side(kind="base", base="rest")
    elif o["baseline"]:
        B = H.side(dict(x.split(":", 1) for x in o["baseline"]))
    else:
        B = None if o["type"] in ("parametric",) or not o["baseline"] and o["type"] != "difference" else H.side()
    if not T["atoms"] and B is not None and B["atoms"]:
        return (fam, tuple(sorted((ax, H.ABSENT, v) for ax, v in B["atoms"].items()))), "override", 1
    if B is not None and not B["atoms"] and not H.is_base(B):
        B = None
    sig = H.signature(fam, T, B)
    return (sig, "override", 1) if sig else (None, "override_empty", 1)


def read(a, paper, lex, override=None, family_ids=frozenset()):
    task = (a["tasks"].split(",") or [None])[0] or None
    fam = (a["families"].split(",") or [None])[0] or H.TASK_FAMILY.get(task)
    if a["keys"] and not a["keys"][0].startswith("task:"):
        fam = a["keys"][0].split("|", 1)[0]
    if not fam:
        return None, "no_family", 1
    if override is not None and override["confidence"] == "high":
        dropped_base = (override.get("source") == "ft" and a["neg"] and not override["baseline"]
                        and not override["baseline_special"] and override["type"] not in ("not_conditions", "task_vs_base"))
        if not dropped_base:            # the fine-tuned model sometimes drops a baseline the paper names: rules instead
            return from_override(override, fam, task, family_ids)
    T = names_side(a["pos"], task, fam, paper, lex) if a["pos"] else None
    B = names_side(a["neg"], task, fam, paper, lex) if a["neg"] else None
    flip_named = 1
    if T is None and B is not None:     # only the baseline side is named: its deactivation, read with sign -1
        T, B, flip_named = B, None, -1
    if T is not None:
        if T["kind"] == "not_a_condition" or (B and B["kind"] == "not_a_condition"):
            return None, "not_a_condition", 1
        T2, B2, sign = H.orient(T, B)
        sig = H.signature(fam, T2, B2)
        if sig:
            return sig, "names", sign * flip_named
    if a["keys"] and not a["keys"][0].startswith("task:"):
        kfam, KT, KB, ksign = key_sides(a["keys"][0], lex)
        if KT is not None:
            KT2, KB2, s2 = H.orient(KT, KB)
            sig = H.signature(kfam, KT2, KB2)
            if sig:
                return sig, "key", ksign * s2
    if T is None:
        return None, "no_sides", 1
    real_task = task and task not in family_ids
    t_plain = not T["atoms"] and T["kind"] == "task_state"
    if t_plain and B is None:
        return ((f"task:{task}", (("@task", H.TASK, None),)), "task_simple", 1) if real_task else (None, "task_family_id", 1)
    if t_plain and H.is_base(B):
        if B["base"] != "rest":
            return None, "task_vs_control", 1
        return ((f"task:{task}", (("@task", H.TASK, H.REST),)), "task_vs_rest", 1) if real_task else (None, "task_family_id", 1)
    if H.is_base(T) and B is not None and not B["atoms"] and B["kind"] == "task_state":
        if T["base"] != "rest":
            return None, "task_vs_control", 1
        return ((f"task:{task}", (("@task", H.TASK, H.REST),)), "task_vs_rest", -1) if real_task else (None, "task_family_id", 1)
    if override is not None:
        return None, "override_low", 1
    return None, "unnormalized", 1
