"""Contrast hub: one condition/contrast vocabulary that imaging contrasts and claim sentences both map onto.

A hub contrast is a *signature* inside a task family: the condition axes on which the target and baseline differ,
with each side's value, e.g. `conflict_and_interference|congruency:incongruent>congruent` or
`perceptual_judgment|stimulus:faces>@base` (a baseline condition kind: rest, fixation, control). Phases and atoms
both sides share (the task's material) are not part of it. Condition atoms are ns-pond's (`conditions_9e96082b.py`,
vendored read-only from ns-pond old/main 9e96082b) plus this hub's `lexicon.json` (strings the atoms miss, mapped by
the growth loop). Each signature has one canonical direction (the one more corpus papers use); a contrast in the
other direction maps with sign -1.

Sources (provenance on every entry):
- ns_pond: the corpus contrasts' keys (`family|target tokens|baseline`, records.py:295), re-read as signatures;
- cogat: Cognitive Atlas task contrasts (contrast_labels/cogat_tasks.json), split with normalize_contrast, in the
  family of the inventory task whose crosswalk names that Atlas task; their concept links are kept;
- inventory: each task's condition_schema (the conditions, axis:value, a family's tasks can have);
- claims: signatures claim sentences name that neither of the above has (`add-claims`).

A signature is `established` when two or more papers or an Atlas contrast attest it.

usage: python hub.py build CONTRASTS.json OUT_DIR [LEXICON.json|-] [ABBREV.json|-] [GLOSSARY.json|-] [PAPER_LEXICON.json|-] [OVERRIDES.json|-] [DIRECTIONS.json|-] [ATTEST.json|-] [JUDGE.json|-] [TASKS.json|-]   (CONTRASTS.json from key_coverage.py; TASKS.json from task_assign.py)
       python hub.py add-claims HUB.json CLAIMS_MATCH.json             (claim-born entries, from match_claims.py)
(module: Hub, side_from_text, signature, used by match_claims.py)
"""
import collections
import json
import os
import re
import sys
from pathlib import Path

from conditions_9e96082b import normalize_condition_ctx

HERE = Path(__file__).parent
INVENTORY = json.load(open(HERE / "task_inventory_v3_9e96082b.json"))
TASK_FAMILY = {t["id"]: t["family"] for t in INVENTORY["tasks"]}
BASE_KINDS = {"rest", "fixation", "control_state"}
REST, CONTROL, TASK = "@rest", "@control", "@task"
BASES = {REST, CONTROL}
ABSENT = ""        # the axis has no value on this side (printed ∅); a one-sided component's baseline is None
#: a control_state condition is a low-level baseline (rest-like) when its name says so; otherwise it is a matched
#: control condition (butterflies for spiders, reversed speech), which is a condition, not a baseline
LOWLEVEL = re.compile(r"\b(rest|fixat\w*|baseline|passive|null|low[- ]level|sensorimotor|blank|cross(hair)?|implicit)\b", re.I)


def fold(s):
    return re.sub(r"[^a-z0-9+]+", " ", str(s).lower()).strip()


class Lexicon:
    """folded condition string -> atoms. Two scopes: family-wide entries (the growth loop's strings seen across
    papers, never an abbreviation) and paper entries (names decoded from one paper's own conditions, descriptions and
    abbreviations, which hold only in that paper). `for_paper` gives the view one paper's contrasts are read with."""

    def __init__(self, path=None, paper_path=None, paper=None, _m=None, _p=None):
        self.m = _m if _m is not None else (json.load(open(path)) if path and Path(path).exists() else {})
        self.p = _p if _p is not None else (json.load(open(paper_path)) if paper_path and Path(paper_path).exists() else {})
        self.paper = paper

    def for_paper(self, paper):
        return Lexicon(paper=paper, _m=self.m, _p=self.p)

    def get(self, fam, s):
        f = fold(s)
        if self.paper is not None:
            hit = (self.p.get(self.paper) or {}).get(f)
            if hit:
                return hit
        return (self.m.get(fam) or {}).get(f) or (self.m.get("*") or {}).get(f)


def base_kind(kind, name=""):
    """'rest' (rest, fixation, a low-level control), 'control' (a matched control condition) or None"""
    if kind in ("rest", "fixation"):
        return "rest"
    if kind == "control_state":
        return "rest" if LOWLEVEL.search(name or "") else "control"
    return None


#: conflict-like families, where "matching" means congruent; elsewhere it is a task (figure matching, delayed match)
CONFLICT = {"conflict_and_interference", "attentional_cueing", "response_inhibition"}
#: words that ns-pond reads as load (simple, easy, hard, complex, difficult) name difficulty, not memory load
DIFFICULTY = re.compile(r"\b(simple|easy|hard|complex|difficult)\b", re.I)
LOAD_WORDS = re.compile(r"\bload\b|back\b|\bdigits?\b|\bitems?\b|\bset size\b|\bspan\b", re.I)
#: hub-level atoms for words ns-pond's atoms miss, applied only where they find nothing on that axis
EXTRA = [(re.compile(r"\bemotional\b", re.I), ("valence", "emotional")), (re.compile(r"\berotic\b", re.I), ("valence", "positive")),
         (re.compile(r"^(outcomes?|feedback)$", re.I), ("outcome", "any"))]


def fix_atoms(atoms, text, fam):
    """the hub's corrections to ns-pond's atoms: 'matching' outside conflict families is not congruency; simple /
    easy / hard / complex without a load word are difficulty; a few words its atoms miss"""
    t = text or ""
    if atoms.get("congruency") == "congruent" and fam not in CONFLICT and re.search(r"\bmatch", t, re.I):
        atoms = {k: v for k, v in atoms.items() if k != "congruency"}
    if atoms.get("load") in ("high", "low") and DIFFICULTY.search(t) and not LOAD_WORDS.search(t):
        atoms = {**{k: v for k, v in atoms.items() if k != "load"}, "difficulty": atoms["load"]}
    for rx, (ax, v) in EXTRA:
        if ax not in atoms and rx.search(t.strip()):
            atoms = {**atoms, ax: v}
    return atoms


def side(atoms=None, kind="task_state", residue="", base=None):
    return {"atoms": atoms or {}, "kind": kind, "residue": residue, "base": base}


def side_from_text(text, task=None, fam=None, lex=None):
    """a condition name -> side: atoms {axis: value}, kind (task_state | base | not_a_condition), residue, and base
    ('rest' / 'control') when the condition is a baseline without atoms. A baseline that carries atoms (0-back,
    scrambled faces) is a condition: it is compared on those atoms."""
    c = normalize_condition_ctx(text, task, fam)
    atoms = fix_atoms(dict(a.split(":", 1) for a in c["atoms"]), text, fam)
    if atoms == {"trial_type": "null"}:
        return side(kind="base", base="rest")                # null trials are the implicit baseline
    if atoms:
        return side(atoms)
    b = base_kind(c["kind"], text)
    if b:
        return side(kind="base", base=b)
    hit = lex.get(fam, text) if lex else None
    if hit is not None:                               # the lexicon only fills in where the atoms find nothing
        if hit in (["@base"], ["@rest"]):
            return side(kind="base", base="rest")
        if hit == ["@none"]:
            return side(kind="not_a_condition", residue=fold(text))
        return side(dict(a.split(":", 1) for a in hit))
    return side(residue=c["residue"] or fold(text))


#: basic emotions as valences, for lifting a side whose cells differ in emotion and for the hierarchy parent
EMO_VALENCE = {"fear": "negative", "anger": "negative", "sad": "negative", "disgust": "negative", "pain": "negative",
               "happy": "positive"}


def lift(ax, vals):
    """the coarser atom a set of differing values on one axis shares, or None"""
    if ax == "emotion":
        v = {EMO_VALENCE.get(x) for x in vals}
        if len(v) == 1 and None not in v:
            return "valence", v.pop()
        if None not in v:
            return "valence", "emotional"
    if ax == "valence" and set(vals) <= {"negative", "positive", "emotional"}:
        return "valence", "emotional"
    if ax == "outcome" and set(vals) <= {"gain", "loss", "any"}:
        return "outcome", "any"
    return None


def combine(cells):
    """one side from the cells it pools (a main-effect side lists several): the atoms every cell shares; an axis on
    which the cells differ lifts to its coarser grain (fear | anger -> valence:negative) or drops; never the first
    cell's value"""
    cells = [c for c in cells if c is not None]
    if not cells:
        return None
    with_atoms = [c["atoms"] for c in cells if c["atoms"]]
    if with_atoms:
        axes = set.intersection(*[set(a) for a in with_atoms])
        out = {}
        for ax in sorted(axes):
            vals = {a[ax] for a in with_atoms}
            if len(vals) == 1:
                out[ax] = vals.pop()
            else:
                lf = lift(ax, vals)
                if lf and lf[0] not in out:
                    out[lf[0]] = lf[1]
        if out:
            return side(out)
    if all(c["kind"] == "base" for c in cells):
        return side(kind="base", base="rest" if all(c["base"] == "rest" for c in cells) else "control")
    if any(c["kind"] == "not_a_condition" for c in cells):
        return side(kind="not_a_condition", residue=cells[0]["residue"])
    return side(residue=" | ".join(c["residue"] for c in cells if c["residue"]))


def side_from_tokens(tokens, fam, lex=None, drop_axes=()):
    """a key side ('congruency:incongruent,phase:cue;...' or 'rest+fixation' or '~residue') -> side; each ';' token is
    one cell. Atoms on drop_axes are left out (the incidental axes, when the key is the only reading)."""
    cells = []
    for t in tokens:
        if t.startswith("~"):
            hit = lex.get(fam, t[1:]) if lex else None
            if hit in (["@base"], ["@rest"]):
                cells.append(side(kind="base", base="rest"))
            elif hit and hit != ["@none"]:
                cells.append(side(dict(a.split(":", 1) for a in hit)))
            else:
                cells.append(side(residue=t[1:]))
        elif t in BASE_KINDS or t == "?":
            cells.append(side(kind="base", base="rest" if t != "control_state" else "control") if t != "?" else side())
        else:
            at = {}
            for a in t.split(","):
                ax, v = a.split(":", 1)
                if ax != "phase" and ax not in drop_axes:
                    at.setdefault(ax, v)
            cells.append(side(at) if at else side(residue=t))
    return combine(cells)


def is_base(s):
    return s is not None and s["kind"] == "base" and not s["atoms"]


def orient(T, B):
    """(target, baseline, sign): a baseline-vs-condition contrast (deactivation) is read as condition > baseline with
    sign -1"""
    if is_base(T) and B is not None and B["atoms"]:
        return B, T, -1
    return T, B, 1


def signature(fam, tgt, base):
    """(family, components): a sorted tuple of (axis, target value, baseline value). The baseline value is a value,
    ABSENT ("": the axis has no value on that side), REST / CONTROL (a baseline condition), or None (one-sided: a
    simple effect or main effect with no other side). None when the target has no atoms."""
    if tgt is None or not tgt["atoms"]:
        return None
    if base is None:
        comp = [(ax, v, None) for ax, v in tgt["atoms"].items()]
    elif is_base(base):
        mark = REST if base["base"] == "rest" else CONTROL
        comp = [(ax, v, mark) for ax, v in tgt["atoms"].items()]
    elif not base["atoms"]:
        return None
    else:
        axes = set(tgt["atoms"]) | set(base["atoms"])
        comp = [(ax, tgt["atoms"].get(ax, ABSENT), base["atoms"].get(ax, ABSENT)) for ax in axes
                if tgt["atoms"].get(ax) != base["atoms"].get(ax)]
        if not comp:
            return None
        comp = alias_axes(merge_axes(comp))
        two = [c for c in comp if c[1] != ABSENT and c[2] != ABSENT]
        if two:                     # a component absent on one side is usually shared material read on one side only
            comp = two
    return fam, tuple(sorted(comp))


#: one manipulation read on two axes: the value on the other axis that means the same comparison
ALIAS = {("switch", "repeat"): ("novelty", "old"), ("mentalizing", "mental"): ("belief", "false")}


def alias_axes(comp):
    d = {ax: (a, b) for ax, a, b in comp}
    for (ax1, v1), (ax2, v2) in ALIAS.items():
        if ax1 in d and ax2 in d:
            (a1, b1), (a2, b2) = d[ax1], d[ax2]
            if a1 == v1 and not b1 and b2 and not a2:          # target on ax1, baseline on ax2
                d[ax2] = (v2, b2); del d[ax1]
            elif b1 == v1 and not a1 and a2 and not b2:
                d[ax2] = (a2, v2); del d[ax1]
    return [(ax, a, b) for ax, (a, b) in d.items()]


#: axes that name one dimension at two grains: a basic emotion against a valence ("fearful > neutral faces") is one
#: comparison on the emotion axis; "neutral" read as congruency in a conflict family is the valence of the other side
MERGE = [("emotion", "valence"), ("valence", "congruency")]


def merge_axes(comp):
    d = {ax: (a, b) for ax, a, b in comp}
    for fine, coarse in MERGE:
        if fine in d and coarse in d:
            (fa, fb), (ca, cb) = d[fine], d[coarse]
            if coarse == "congruency" and "neutral" not in (ca, cb):
                continue
            if fa and not fb and cb and not ca:
                d[fine] = (fa, cb); del d[coarse]
            elif fb and not fa and ca and not cb:
                d[fine] = (ca, fb); del d[coarse]
    return [(ax, a, b) for ax, (a, b) in d.items()]


def flip(sig):
    """the reverse comparison, or None for a one-sided signature or one whose reverse would put a baseline first"""
    fam, comp = sig
    if any(b is None or b in BASES or b == TASK and a != TASK for _, a, b in comp):
        return None
    return fam, tuple(sorted((ax, b, a) for ax, a, b in comp))


def sig_str(sig):
    fam, comp = sig
    return fam + "|" + ";".join(f"{ax}:{a or '∅'}" + ("" if b is None else f">{b or '∅'}") for ax, a, b in comp)


#: marked-first value order per axis: the condition an effect is computed for comes first (incongruent > congruent).
#: Fixed, so a signature's direction does not change as papers are added; the majority decides only where this is
#: silent, and that choice is frozen in directions.json
ORDER = {
    "congruency": ["incongruent", "neutral", "congruent"], "response_demand": ["nogo", "stop", "go"],
    "switch": ["switch", "repeat"], "cue_validity": ["invalid", "valid"], "memory_outcome": ["remembered", "forgotten"],
    "cs": ["cs_plus", "cs_minus"], "accuracy": ["error", "correct"], "novelty": ["new", "old"],
    "valence": ["negative", "positive", "emotional", "neutral"],
    "emotion": ["fear", "anger", "sad", "disgust", "pain", "happy", "surprise", "neutral"],
    "outcome": ["gain", "loss", "no_outcome"], "trial_type": ["deviant", "target", "lure", "catch", "standard"],
    "mentalizing": ["mental", "physical"], "referent": ["self", "other"],
    "regulation": ["reappraise", "suppress", "increase", "look"], "fairness": ["unfair", "fair"],
    "risk": ["risky", "safe"], "expectancy": ["unexpected", "expected"], "motion": ["moving", "static"],
    "engagement": ["active", "passive"], "social": ["social", "exclusion", "inclusion"],
    "stimulus": ["faces", "bodies", "words", "food", "drug_cue", "objects", "houses_scenes", "pseudowords", "letters", "scrambled"],
    "stop_outcome": ["success", "failure"], "agent": ["human", "computer"], "delay": ["delayed", "immediate"],
}
HIGH_LOW = {"load", "magnitude", "intensity", "probability", "level", "arousal", "word_frequency"}


def _order(ax, a, b):
    """1 if a comes first on this axis, -1 if b does, 0 if the table is silent"""
    if a == b:
        return 0
    if b == ABSENT or b in BASES:
        return 1
    if a == ABSENT or a in BASES:
        return -1
    if ax in HIGH_LOW:
        rank = lambda v: (float(v) if re.fullmatch(r"\d+(\.\d+)?", v) else {"high": 1e6, "low": -1e6}.get(v))
        ra, rb = rank(a), rank(b)
        if ra is not None and rb is not None and ra != rb:
            return 1 if ra > rb else -1
        return 0
    o = ORDER.get(ax)
    if o and a in o and b in o:
        return 1 if o.index(a) < o.index(b) else -1
    return 0


def canonical(sig, papers, frozen):
    """(canonical signature, sign) by ORDER, then the frozen choice, then the majority of papers"""
    f = flip(sig)
    if f is None:
        return sig, 1
    comps = sorted(sig[1], key=lambda c: (c[1] == ABSENT or c[2] == ABSENT, c))   # two-valued components decide first
    for ax, a, b in comps:
        o = _order(ax, a, b)
        if o:
            return (sig, 1) if o > 0 else (f, -1)
    k, kf = sig_str(sig), sig_str(f)
    if k in frozen or kf in frozen:
        return (sig, 1) if k in frozen else (f, -1)
    n, nf = len(papers.get(sig, ())), len(papers.get(f, ()))
    return (sig, 1) if (n, kf) >= (nf, k) else (f, -1)


FAMILY_PREFIX = "family:"


def family_of(scope):
    """the task family of a scope (a task id, a family name used as a task id, or `family:<name>`)"""
    if scope.startswith(FAMILY_PREFIX):
        return scope[len(FAMILY_PREFIX):]
    if scope.startswith("task:"):
        scope = scope[5:]
    return TASK_FAMILY.get(scope, scope)


def scoped(sig, task):
    """a signature read in a task's family, re-scoped to the task itself: contrasts mean what they mean within a task
    ("neutral" in a Stroop is not "neutral" in a reward task), so an entry is a task's comparison"""
    if sig is None or sig[0].startswith("task:") or not task:
        return sig
    return task, sig[1]


def parents(sig):
    """backoff entries, each (signature, kind), all within the contrast's task except 'family':
    - 'family': the same comparison at family level (`family:<name>|...`), which links sibling tasks (Stroop, flanker
      and Simon incongruent > congruent); reported as related, never as a match;
    - 'hierarchy': a basic emotion compared with neutral, as its valence (fear > neutral -> valence:negative > neutral);
    - 'component': each component of a multi-component signature that has a value on both sides (never one that is
      absent on a side: that is usually the task's shared material);
    - 'axis': `family|axis:*`, the axis alone (a classification, not a match)."""
    fam, comp = sig
    if fam.startswith("task:") or fam.startswith(FAMILY_PREFIX):
        return []
    out = [((FAMILY_PREFIX + family_of(fam), comp), "family")]
    for c in comp:
        ax, a, b = c
        if ax == "emotion" and a in EMO_VALENCE and b is not None:
            b2 = EMO_VALENCE.get(b, b)
            if b2 == EMO_VALENCE[a]:
                continue                                    # fear > disgust: the same valence on both sides
            rest = tuple(sorted([("valence", EMO_VALENCE[a], b2)] + [x for x in comp if x != c]))
            out.append(((fam, rest), "hierarchy"))
    if len(comp) > 1:
        out += [((fam, (c,)), "component") for c in comp if c[1] not in (ABSENT,) and c[2] not in (ABSENT, None)]
    out += [((fam, ((ax, "*", "*"),)), "axis") for ax in sorted({c[0] for c in comp})]
    return out


def _sig_from_json(sj):
    fam, comp = json.loads(sj)
    return fam, tuple(tuple(c) for c in comp)


class Hub:
    def __init__(self, entries, parent_entries=None):
        self.entries = entries                          # sig_str -> entry (exact signatures)
        self.parents = parent_entries or {}
        self.by_sig = {_sig_from_json(e["sig_json"]): k for k, e in entries.items()}
        self.by_family = collections.defaultdict(list)
        for k, e in entries.items():
            self.by_family[_sig_from_json(e["sig_json"])[0]].append(k)   # by scope (task)

    @classmethod
    def load(cls, path):
        h = json.load(open(path))
        return cls(h["contrasts"], h.get("parents"))

    def lookup(self, sig, established_only=False):
        """exact (or flipped) signature -> (entry id, sign) or None"""
        for s, sign in ((sig, 1), (flip(sig), -1)):
            if s is None:
                continue
            k = self.by_sig.get(s)
            if k and (not established_only or self.entries[k]["established"]):
                return k, sign
        return None

    def relaxed(self, sig, established_only=True):
        """the claim names fewer axes than the hub entry (or more): the best-supported established entry in the same
        family whose components contain the claim's, or are contained in them, in either direction"""
        fam, comp = sig; best = None
        for s, sign in ((sig, 1), (flip(sig), -1)):
            if s is None:
                continue
            c = set(s[1])
            for k in self.by_family.get(fam, []):
                e = self.entries[k]
                if established_only and not e["established"]:
                    continue
                ec = set(_sig_from_json(e["sig_json"])[1])
                if {b is None for _, _, b in c} != {b is None for _, _, b in ec}:
                    continue                          # a one-sided entry never stands for a comparison, nor the reverse
                if c <= ec or (ec <= c and 2 * len(ec) >= len(c)):
                    score = (len(c & ec), e["papers"])
                    if best is None or score > best[0]:
                        best = (score, k, sign)
        return (best[1], best[2]) if best else None


def add_claims(hub_path, claims_match_path, min_articles=2):
    """signatures that claims name and the hub lacks become entries (provenance `claims`), established when
    min_articles or more articles name them; entries that claims hit gain the `claims` provenance too"""
    h = json.load(open(hub_path)); E = h["contrasts"]
    arts = collections.defaultdict(set); sigs = {}
    for r in json.load(open(claims_match_path))["claims"]:
        m = r["match"]
        if m["how"] == "no_entry":
            arts[m["sig"]].add(r["article"]); sigs[m["sig"]] = m["sig_json"]
        elif "hub" in m:
            E[m["hub"]]["provenance"] = sorted(set(E[m["hub"]]["provenance"]) | {"claims"})
            E[m["hub"]]["claim_articles"] = E[m["hub"]].get("claim_articles", 0) + 1
    for k, a in arts.items():
        fam = json.loads(sigs[k])[0]
        E[k] = {"family": fam, "sig_json": sigs[k], "papers": 0, "contrasts": 0, "established": len(a) >= min_articles,
                "provenance": ["claims"], "claim_articles": len(a), "tasks": {}, "top_keys": [], "base_kinds": {}, "cogat": []}
    h["summary"]["claim_entries_added"] = len(arts)
    h["summary"]["claim_entries_established"] = sum(len(a) >= min_articles for a in arts.values())
    json.dump(h, open(hub_path, "w"), indent=1)
    print(f"added {len(arts)} claim-born entries ({h['summary']['claim_entries_established']} established)")


def cogat_contrasts(lex=None):
    """Atlas task contrasts -> (family, task id, contrast name, signature or None, concepts)"""
    at = json.load(open(HERE.parent / "contrast_labels" / "cogat_tasks.json"))
    by_atlas = collections.defaultdict(list)
    for t in INVENTORY["tasks"]:
        for c in t["crosswalk"]["cognitive_atlas"]:
            by_atlas[c["id"]].append(t["id"])
    split = re.compile(r"\s*(?:>|\bminus\b|\bvs\.?\b|\bversus\b|\bcompared (?:to|with)\b|\brelative to\b|\s-\s|−)\s*", re.I)
    out = []
    for t in at.values():
        tasks = by_atlas.get(t["cogat_id"], [])
        concepts = collections.defaultdict(list)
        for cc in t.get("concept_contrasts", []):
            for n in cc["contrasts"]:
                concepts[n].append(cc["concept"])
        for c in t["contrasts"]:
            name = c["name"]
            parts = [p for p in re.split(r":\s*", name) if split.search(p)]
            body = parts[0] if parts else name
            sides = split.split(body, maxsplit=1)
            for tid in tasks or [None]:
                fam = TASK_FAMILY.get(tid)
                sig = None
                if fam and len(sides) == 2:
                    T, B, _ = orient(side_from_text(sides[0], tid, fam, lex), side_from_text(sides[1], tid, fam, lex))
                    sig = signature(fam, T, B)
                elif fam:                                  # a single condition or effect named as the contrast
                    sig = signature(fam, side_from_text(name, tid, fam, lex), None)
                out.append({"family": fam, "task": tid, "atlas_task": t["name"], "contrast": name, "id": c["id"],
                            "sig": sig, "concepts": concepts.get(name, [])})
    return out


#: articles of the claim catalog that must name a signature for it to count as shared when one corpus paper has it
MIN_LIT = 2


def well_formed(sig):
    """every component valued on both sides (or a baseline): what literature attestation may establish"""
    return all(a not in (ABSENT, None) and b not in (ABSENT, None) for _, a, b in sig[1])


def cross_axis(sig):
    """the two sides' values sit on different axes and no component is valued on both: a misreading"""
    comp = sig[1]
    return (len(comp) >= 2 and not any(a != ABSENT and b not in (ABSENT, None) for _, a, b in comp)
            and any(a == ABSENT for _, a, b in comp) and any(b == ABSENT for _, a, b in comp))


def generic_sig(sig):
    """a bare level:high > low pools unrelated quantities (alpha power, desirability): never established"""
    return all(c[0] == "level" for c in sig[1])


def sig_json(sig):
    return json.dumps([sig[0], [list(x) for x in sig[1]]])


def build(contrasts_path, out_dir, lex_path=None, abbr_path=None, gloss_path=None, paper_lex_path=None,
          overrides_path=None, directions_path=None, attest_path=None, judge_path=None, tasks_path=None):
    """read every corpus contrast (imaging.read), put each in its exact canonical signature, and record which
    established entry it counts toward:
    - L0: its exact signature is established (2+ papers whose contrasts have exactly it, or an Atlas contrast);
      task-level entries (`task:<id>|@task>@rest`) count only for real task ids;
    - L0a: one corpus paper has it and 2+ articles of the claim catalog name it in their text (attest.py);
    - L0h: its emotion -> valence hierarchy parent is (fear > neutral counts toward negative > neutral);
    - L0j: judge.py ruled it (high confidence) the same comparison as an established entry of its task;
    - F 'sibling task': the same comparison is in 2+ papers of other tasks of the family (related, not a match);
    - L1 'related': a two-sided component of a multi-component signature is (the rarest axis in the family first);
    - L2 'axis': only the axis is (a classification, not a match).
    An exact entry is a task's comparison (`<task id>|axis:...`): it counts only its own contrasts' papers; parents
    are a separate table."""
    import imaging
    lex = Lexicon(lex_path, paper_lex_path)
    overrides = json.load(open(overrides_path)) if overrides_path and Path(overrides_path).exists() else {}
    frozen = set(json.load(open(directions_path))) if directions_path and Path(directions_path).exists() else set()
    attest = json.load(open(attest_path)) if attest_path and Path(attest_path).exists() else {}
    judged = json.load(open(judge_path)) if judge_path and Path(judge_path).exists() else {}
    C = {}
    for cp in contrasts_path.split("+"):              # several contrast files joined by "+" (sample + more papers)
        C.update(json.load(open(cp)))
    ta_ids = set()
    if tasks_path and Path(tasks_path).exists():       # task_assign.py: a task for contrasts labelled only by family
        import task_assign
        task_assign.apply(C, json.load(open(tasks_path)))
        ta_ids = {a["tasks"] for a in C.values() if a.get("task_assigned")}
    abbr = json.load(open(abbr_path)) if abbr_path else {}
    gloss = {}
    for gp in (gloss_path or "").split("+"):           # glossaries joined by "+", like the contrast files
        if gp:
            gloss.update(json.load(open(gp)))
    family_ids = set(INVENTORY["families"]) | {t["family"] for t in INVENTORY["tasks"]}
    paper_obj = {}; per = {}; st = collections.Counter()
    for aid, a in C.items():
        p = a["paper"]
        if p not in paper_obj:
            paper_obj[p] = imaging.Paper(abbr.get(p), gloss.get(p))
        sig, how, sign = imaging.read(a, paper_obj[p], lex.for_paper(p), overrides.get(aid), family_ids)
        sig = scoped(sig, (a["tasks"].split(",") or [None])[0] or None)
        st["how:" + how] += 1
        per[aid] = (sig, sign, a) if sig else None
    papers = collections.defaultdict(set)
    for v in per.values():
        if v:
            papers[v[0]].add(v[2]["paper"])
    for aid, v in list(per.items()):     # a cross-axis reading one paper alone has is usually a misreading: for Codex
        if v and not v[0][0].startswith("task:") and cross_axis(v[0]):
            f = flip(v[0])
            if len(papers[v[0]] | (papers.get(f, set()) if f else set())) < 2:
                per[aid] = None; st["how:cross_axis_single"] += 1; st["how:" + "cross_axis_from"] += 1
    papers = collections.defaultdict(set)
    for v in per.values():
        if v:
            papers[v[0]].add(v[2]["paper"])
    canon = {sig: canonical(sig, papers, frozen) for sig in papers}
    E = {}; P = {}

    def new(sig, kind):
        return {"scope": sig[0], "family": family_of(sig[0]), "sig_json": sig_json(sig), "papers_set": set(), "contrasts": 0, "kind": kind,
                "tasks": collections.Counter(), "keys": collections.Counter(), "base_kinds": collections.Counter(),
                "provenance": set(), "cogat": []}
    for aid, v in list(per.items()):
        if not v:
            continue
        sig, sign, a = v; csig, s2 = canon[sig]
        k = sig_str(csig)
        e = E.setdefault(k, new(csig, "exact"))
        e["papers_set"].add(a["paper"]); e["contrasts"] += 1; e["provenance"].add("ns_pond")
        e["tasks"].update(t for t in a["tasks"].split(",") if t)
        if a["keys"]:
            e["keys"][a["keys"][0]] += 1
        e["base_kinds"].update(b for _, _, b in csig[1] if b in BASES)
        for psig, pk in parents(csig):
            pc, _ = canonical(psig, papers, frozen)
            pe = P.setdefault(sig_str(pc), new(pc, pk)); pe["papers_set"].add(a["paper"]); pe["contrasts"] += 1
        per[aid] = (csig, sign * s2, a)
    atl = cogat_contrasts(lex); st["cogat_contrasts"] = len({c["id"] for c in atl})
    for c in atl:
        if c["sig"] is None or not c["task"]:
            st["cogat_unnormalized"] += 1; continue
        csig, sign = canonical(scoped(c["sig"], c["task"]), papers, frozen)
        e = E.setdefault(sig_str(csig), new(csig, "exact")); e["provenance"].add("cogat")
        e["cogat"].append({"id": c["id"], "name": c["contrast"], "task": c["task"], "concepts": c["concepts"], "sign": sign})

    def done(e, k):
        n = len(e["papers_set"])
        fam = e["scope"]
        if fam.startswith("task:"):
            tid = fam[5:]
            est = n >= 2 and tid not in family_ids and REST in k and TASK in k
        elif fam in family_ids:
            est = False         # a family label used as the task pools unrelated tasks: not a task's comparison
        else:
            est = (n >= 2 or bool(e["cogat"])) and not generic_sig(_sig_from_json(e["sig_json"]))
        sg = _sig_from_json(e["sig_json"]); fl = flip(sg)
        lit = max(attest.get(k, 0), attest.get(sig_str(fl), 0) if fl else 0)
        if fam.startswith("task:"):
            lit = 0
        return {"scope": e["scope"], "family": e["family"], "sig_json": e["sig_json"], "kind": e["kind"], "papers": n, "contrasts": e["contrasts"],
                "literature_articles": lit, "established_by": "corpus" if est else ("literature" if lit >= MIN_LIT and n >= 1 and well_formed(sg) and not generic_sig(sg) and fam not in family_ids else None),
                "established": est or (lit >= MIN_LIT and n >= 1 and well_formed(sg) and not generic_sig(sg) and fam not in family_ids), "provenance": sorted(e["provenance"]), "tasks": dict(e["tasks"].most_common(8)),
                "top_keys": [x for x, _ in e["keys"].most_common(3)], "base_kinds": dict(e["base_kinds"]),
                "cogat": e["cogat"]}
    out = {k: done(e, k) for k, e in E.items()}
    pout = {k: done(e, k) for k, e in P.items()}
    axis_papers = collections.Counter()
    for k, e in pout.items():
        if e["kind"] == "axis":
            axis_papers[(e["scope"], json.loads(e["sig_json"])[1][0][0])] = e["papers"]
    assign = {}; lv = collections.Counter(); unnorm = []; ta = collections.Counter()
    for aid, v in per.items():
        if not v:
            unnorm.append(aid); lv["unnormalized"] += 1; continue
        csig, sign, a = v; k = sig_str(csig); kind = "task" if csig[0].startswith("task:") else "condition"
        level, target, psign = None, None, 1
        if out[k]["established"]:
            level, target = ("L0" if out[k]["established_by"] == "corpus" else "L0a"), k
        elif judged.get(aid, {}).get("decision") == "same" and judged[aid]["confidence"] == "high" and out.get(judged[aid]["entry"], {}).get("established"):
            level, target = "L0j", judged[aid]["entry"]     # judge.py: the same comparison as this entry of the task
            psign = judged[aid]["sign"] * sign              # the judge's direction is relative to the named sides
        else:
            ps = []
            for p, pk in parents(csig):
                pc, psn = canonical(p, papers, frozen)
                ps.append((pk, sig_str(pc), psn))
            hier = [(x, sn) for pk, x, sn in ps if pk == "hierarchy" and pout.get(x, {}).get("established")]
            sib = [(x, sn) for pk, x, sn in ps if pk == "family" and pout.get(x, {}).get("papers", 0) >= 2]
            comp = [(x, sn) for pk, x, sn in ps if pk == "component" and pout.get(x, {}).get("established")]
            axis = [(x, sn) for pk, x, sn in ps if pk == "axis" and pout.get(x, {}).get("established")]
            if hier:
                level, (target, psign) = "L0h", hier[0]
            elif sib:
                level, (target, psign) = "F", sib[0]      # the same comparison is shared by sibling tasks of the family
            elif comp:
                comp.sort(key=lambda x: axis_papers.get((csig[0], json.loads(pout[x[0]]["sig_json"])[1][0][0]), 0))
                level, (target, psign) = "L1", comp[0]
            elif axis:
                level, (target, psign) = "L2", axis[0]
        if kind == "condition" and csig[0] in family_ids:
            kind = "family_labelled"                     # the corpus names only the family as its task
        if C[aid].get("task_assigned"):
            ta[f"{C[aid]['task_assigned']}:{level}"] += 1
        assign[aid] = {"hub": k, "sign": sign, "level": level, "established": target, "kind": kind,
                       "task_assigned": C[aid].get("task_assigned"),
                       "established_sign": sign * (psign if level in ("L0h", "L0j", "F", "L1", "L2") else 1),
                       "judged": judged.get(aid, {}).get("decision")}
        lv[f"{kind}:{level or 'unestablished'}"] += 1
    conds = collections.defaultdict(lambda: collections.defaultdict(set))
    for t in INVENTORY["tasks"]:
        for ax, vals in (t.get("condition_schema") or {}).items():
            if not ax.startswith("_"):
                for x in vals:
                    conds[t["family"]][f"{ax}:{x}"].add("inventory")
    for e in out.values():
        for ax, a, b in json.loads(e["sig_json"])[1]:
            for x in (a, b):
                if x and x not in BASES and x not in ("*", TASK):
                    conds[e["family"]][f"{ax}:{x}"].add("ns_pond")
    conditions = {f: {a: sorted(p) for a, p in sorted(d.items())} for f, d in sorted(conds.items())}
    n = len(C); g = lambda key: lv.get(key, 0)
    summary = {"contrasts": n, "levels": dict(lv), "stats": dict(st),
               "headline_exact_condition_L0": g("condition:L0"), "share_exact_condition_L0": round(g("condition:L0") / n, 3),
               "literature_L0a": g("condition:L0a"), "judged_same_L0j": g("condition:L0j"), "task_vs_rest_L0": g("task:L0"), "hierarchy_L0h": g("condition:L0h"),
               "share_matched_L0_L0h_task": round((g("condition:L0") + g("condition:L0h") + g("task:L0")) / n, 3),
               "share_matched_with_literature": round((g("condition:L0") + g("condition:L0a") + g("condition:L0h") + g("task:L0")) / n, 3),
               "share_matched_with_judge": round((g("condition:L0") + g("condition:L0a") + g("condition:L0h") + g("condition:L0j") + g("task:L0")) / n, 3),
               "sibling_task_F": g("condition:F"), "related_L1": g("condition:L1"), "axis_L2": g("condition:L2"),
               "family_labelled_task": sum(v for x, v in lv.items() if x.startswith("family_labelled")),
               "normalized_unestablished": g("condition:unestablished") + g("task:unestablished"),
               "share_normalized": round(len(assign) / n, 3), "entries": len(out),
               "task_assigned_levels": dict(ta),
               "established_entries": sum(e["established"] for e in out.values())}
    if os.environ.get("SAMPLE"):      # the hub drew on more papers than the sample: the shares that count are the sample's
        sample = set(json.load(open(os.environ["SAMPLE"])))
        slv = collections.Counter(f"{v['kind']}:{v['level'] or 'unestablished'}"
                                  for aid, v in assign.items() if aid in sample)
        slv["unnormalized"] = sum(1 for aid in unnorm if aid in sample)
        sn = len(sample & set(C)); sg = lambda key: slv.get(key, 0)  # noqa: E731
        matched = sg("condition:L0") + sg("condition:L0a") + sg("condition:L0h") + sg("condition:L0j") + sg("task:L0")
        tid = {t["id"] for t in INVENTORY["tasks"]}
        real = sum(1 for aid in sample if aid in C and (C[aid]["tasks"].split(",") or [""])[0] in tid | set(ta_ids))
        summary["sample"] = {"contrasts": sn, "levels": dict(slv), "matched": matched, "share_matched": round(matched / sn, 3),
                             "with_real_task": real, "share_matched_of_real_task": round(matched / max(1, real), 3),
                             "share_normalized": round(sum(1 for aid in sample if aid in assign) / sn, 3)}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "contrasts": out, "parents": pout, "conditions": conditions},
              open(Path(out_dir) / "hub.json", "w"), indent=1)
    json.dump(assign, open(Path(out_dir) / "corpus_assign.json", "w"))
    json.dump(unnorm, open(Path(out_dir) / "corpus_unnormalized.json", "w"))
    chosen = sorted({sig_str(c) for c, _ in canon.values() if flip(c) is not None})
    json.dump(sorted(frozen | set(chosen)), open(Path(out_dir) / "directions.json", "w"))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3], *[x if x != "-" else None for x in sys.argv[4:13]])
    elif sys.argv[1] == "add-claims":
        add_claims(sys.argv[2], sys.argv[3])
