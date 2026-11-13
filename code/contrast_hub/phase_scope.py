"""Which hub signatures mix trial phases (James 2026-11-12: "I don't know if we need a phase, or just change the
associated constructs for those particular contrasts"). Constructs were attributed per signature, so a signature whose
analyses come from different trial phases (MID's outcome:gain>no_outcome holds both "Anticipation of 1 Euro vs.
non-reward" and "Receipt of reward vs. omission of reward") gives all of them one phase's constructs.

For the sample analyses in tasks with trial phases, the phase each names is read from its Neurostore name and
description and its sides (contrasts_all.json pos/neg), by a keyword rule per family (PHASES; the first match in
PRECEDENCE order wins, so "anticipation of reward > no outcome" is anticipation and "subsequent hits > misses"
encoding). A signature (hub entry and direction) is mixed when its analyses name two or more phases. Every analysis
of a mixed signature is in scope for a re-reading of its constructs, those whose names give no phase included.

  names  SAMPLE_IDS.json OUT.json         fetch names and descriptions (public Neurostore API, read-only GETs)
  scope  CORPUS.json HUB_DIR CONTRASTS.json NAMES.json OUT.json

usage: (beast, pilot-results/contrast_hub) python ../code/contrast_hub/phase_scope.py ...
"""
import collections
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

R_ANT = r"anticip|\bcues?\b|cue phase|cue period|expect|prospect|preparat"
R_OUT = r"outcome|receipt|receiv|feedback|consum|omission|delivery|notification|\bwin(s|ning)?\b"
R_ENC = r"encod|subsequent|\bstudy|memoriz|later remember|\bsme\b|\bdm effect"
R_RET = r"retriev|recogni|recall|\btest\b|\bhits?\b|\bmiss(es)?\b|old.?new|familiar|recollect|correct rejection|false alarm"
R_MNT = r"maint|\bdelay|retention|\bhold"
R_PRB = r"\bprobe|\bretriev|\brecogn"
R_CS = r"\bcs\s*[+-]|\bcs\b|conditioned stimul|\bcue"
R_US = r"\bus\b|unconditioned|shock|reinforcer delivery|\boutcome"
PHASES = {   # family -> [(phase, regex)] in precedence order
    "reward_and_value_choice": [("anticipation", R_ANT), ("outcome", R_OUT),
                                ("decision", r"decision|\bchoice|\bchoos|\bselect(ion)? phase|\bbet(ting)?\b|\bwager|\bgamble\b")],
    "recall_and_recognition": [("encoding", R_ENC), ("retrieval", R_RET)],
    "memory_control": [("encoding", R_ENC), ("retrieval", R_RET)],
    "working_memory_span": [("maintenance", R_MNT), ("probe", R_PRB), ("encoding", r"encod|memory set|\bsample\b")],
    "conditioning_and_reinforcement": [("us", R_US), ("cs", R_CS)],
    "attentional_cueing": [("cue", r"\bcue period|\bcue phase|\bcue-related|\bcue onset|\bcue only"), ("target", r"\btarget")],
    "pain": [("anticipation", R_ANT), ("stimulation", r"during (the )?(pain|heat|stimul)|painful stimul|pain period|heat period")],
    "emotion_regulation": [("cue", r"\binstruction|\bcue\b|cue period|cue phase|anticipat"),
                           ("rating", r"\brating|\brate\b|\bratings"),
                           ("regulation", r"regulat|reappraisal|reappraise|suppress|distanc|\blook\b|\bview")],
    "social_games": [("decision", r"decision|\bchoice|\bchoos|\boffer\b|\binvest"), ("outcome", R_OUT + r"|repay|\breturn")],
    "spatial_navigation": [("encoding", r"encod|learning|\bstudy|explor"), ("retrieval", r"retriev|recall|\btest\b|probe")],
}
TASK_FAMILY = {"hedtsk_task_switching": "attentional_cueing", "tct_pain_stimulation": "pain",
               "hedtsk_emotion_regulation": "emotion_regulation", "hedtsk_trust_game": "social_games",
               "hedtsk_ultimatum_game": "social_games", "hedtsk_prisoners_dilemma": "social_games",
               "tct_public_goods_game": "social_games", "hedtsk_dictator_game": "social_games"}


def names():
    idsp, outp = sys.argv[2:4]
    ids = json.load(open(idsp)); out = {}

    def get(a):
        for _ in range(4):
            try:
                d = json.load(urllib.request.urlopen(f"https://neurostore.org/api/analyses/{a}?nested=false", timeout=30))
                return a, {"name": d.get("name") or "", "description": d.get("description") or ""}
            except Exception:  # noqa: BLE001
                time.sleep(3)
        return a, None
    with ThreadPoolExecutor(4) as p:
        for a, r in p.map(get, ids):
            out[a] = r
    json.dump(out, open(outp, "w"))
    print(len(out), "fetched;", sum(v is None for v in out.values()), "failed")


def phase_of(fam, text):
    for ph, rx in PHASES.get(fam, []):
        if re.search(rx, text):
            return ph
    return None


def family_of(t, P, fams):
    if t in TASK_FAMILY:
        return TASK_FAMILY[t]
    return next((p for p in P.get(t, []) if p in fams), t if t in fams else None)


def scope():
    corpusp, hub_dir, contrastsp, namesp, outp = sys.argv[2:7]
    B = json.load(open(corpusp)); P = B["parents"]; A = json.load(open(f"{hub_dir}/corpus_assign.json"))
    C = json.load(open(contrastsp)); N = json.load(open(namesp))
    fams = {p for t, ps in P.items() for p in ps if not p.startswith(("c:", "m:"))}
    sig = collections.defaultdict(list)
    for k, d in B["docs"].items():
        a = A.get(k)
        if not a or k not in N:
            continue
        t = next((i for i in d["ids"] if not i.startswith(("c:", "m:")) and i not in ("background", "deactivation")), None)
        f = family_of(t, P, fams)
        if f not in PHASES:
            continue
        n = N[k] or {}; c = C.get(k) or {}
        text = " ".join([n.get("name", ""), n.get("description", ""), c.get("pos", ""), c.get("neg", "")]).lower()
        sig[(a["hub"], a.get("sign", 1))].append((k, phase_of(f, text), d["group"], f))
    mixed = {s: v for s, v in sig.items() if len({p for _, p, _, _ in v if p}) >= 2}
    inscope = [k for v in mixed.values() for k, _, _, _ in v]
    papers = {g for v in mixed.values() for _, _, g, _ in v}
    byfam = collections.Counter(f for v in mixed.values() for _, _, _, f in v)
    detected = collections.Counter(p for v in sig.values() for _, p, _, _ in v)
    out = {"signatures_checked": len(sig), "analyses_checked": sum(len(v) for v in sig.values()),
           "phase_named": dict(detected), "mixed_signatures": len(mixed), "analyses_in_scope": len(inscope),
           "papers_in_scope": len(papers), "by_family": dict(byfam),
           "largest": sorted(([f"{s[0]}{' (reversed)' if s[1] < 0 else ''}", len(v),
                               dict(collections.Counter(p or 'none' for _, p, _, _ in v))] for s, v in mixed.items()),
                             key=lambda x: -x[1])[:25],
           "analyses": {k: {"signature": s[0], "sign": s[1], "phase": p, "paper": g} for s, v in mixed.items() for k, p, g, _ in v}}
    json.dump(out, open(outp, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "analyses"}, indent=1))


if __name__ == "__main__":
    {"names": names, "scope": scope}[sys.argv[1]]()
