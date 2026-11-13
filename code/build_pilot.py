"""Build the pilot bundles: for sampled papers of chosen canonical tasks, the paper text cut into ev1 units,
a section tag per unit, and the paper's own task/contrast records with the units their evidence cites.

Read-only on the ns-pond catalog, blobs and text. usage: python -I build_pilot.py WORKDIR
"""
import gzip
import hashlib
import json
import random
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

CAT = "file:/data/alejandro/projects/ns-pond/catalog/catalog.sqlite?mode=ro"
BLOBS = Path("/data/alejandro/projects/ns-pond/catalog/blobs")
SEED = 20261110

# (canonical id, papers wanted). Frequent tasks with several contrasts get 3.
PICK = [
    ("hedtsk_n_back", 3), ("hedtsk_stroop_color_word", 3), ("hedtsk_stop_signal", 3),
    ("hedtsk_monetary_incentive_delay", 3), ("tct_emotional_face_matching", 2), ("hedtsk_false_belief", 2),
    ("hedtsk_go_no_go", 2), ("hedtsk_pavlovian_fear_conditioning", 2), ("hedtsk_task_switching", 2),
    ("hedtsk_delay_discounting", 2), ("tct_pain_stimulation", 2),
    ("hedtsk_wason_selection", 2), ("hedtsk_psychological_refractory_period", 2), ("hedtsk_trail_making", 2),
    ("tct_continued_influence", 2), ("tct_wm_updating", 2),
]

# --- ev1's unit split, copied from astra-distill code/vista/evidence.py units() so spans map onto units ---
ABBR = r"(?<!\bet al)(?<!\be\.g)(?<!\bi\.e)(?<!\bFig)(?<!\bFigs)(?<!\bvs)(?<!\bcf)(?<!\bapprox)(?<!\bNo)(?<!\bEq)(?<!\bRef)"
CUT = re.compile(ABBR + r"[.!?](?=\s+[A-Z0-9(\[])")
SHORT = 200


def units(text):
    out, pos = [], 0
    for line in text.split("\n"):
        s, e = pos, pos + len(line)
        pos = e + 1
        if not line.strip():
            continue
        if "\t" in line or len(line) <= SHORT:
            out.append((s, e)); continue
        a = s
        for m in CUT.finditer(line):
            out.append((a, s + m.end())); a = s + m.end()
        if a < e:
            out.append((a, e))
    return [(a + len(text[a:b]) - len(text[a:b].lstrip()), b) for a, b in out if text[a:b].strip()]


HEAD = [
    ("abstract", r"abstract|summary"), ("introduction", r"introduction|background"),
    ("methods", r"(materials? and )?methods?( and materials?)?|experimental procedures?|method and materials"),
    ("results", r"results?( and discussion)?"), ("discussion", r"(general )?discussion|conclusions?|concluding remarks"),
    ("back", r"references?|bibliography|acknowledge?ments?|funding|conflicts? of interest|declaration.*|"
             r"supplementary.*|author contributions?|data availability.*|competing interests?"),
]
HEAD_RE = [(s, re.compile(rf"^(?:#+\s*)?(?:[0-9IVX]+\.?\s*)?(?:{p})\s*[:.]?\s*$", re.I)) for s, p in HEAD]


def sections(text, U):
    cur, out = "front", []
    for a, b in U:
        line = text[a:b].strip()
        if len(line) <= 60 and "\t" not in line:  # table rows ("Supplementary motor area\tL\t6...") are not headings
            for s, rx in HEAD_RE:
                if rx.match(line):
                    cur = s; break
        out.append(cur)
    return out


def blob(d):
    return json.loads(gzip.decompress((BLOBS / d[:2] / f"{d}.json.gz").read_bytes()))


def cited(slot, spans_to_unit):
    idx = set()
    for st in ((slot or {}).get("evidence") or {}).get("sets") or []:
        for sp in st.get("spans") or []:
            u = spans_to_unit.get((sp["start_char"], sp["end_char"]))
            if u is not None:
                idx.add(u)
    return sorted(idx)


def val(slot):
    return (slot or {}).get("value")


def bundle(con, art):
    rows = {(s, src): (b, json.loads(m)) for s, src, b, m in con.execute(
        "select stage, source, blob, summary from artifacts where article_id=? and status='ok'", (art,))}
    nb, _ = rows[("normalize", "ns-27b")]
    _, rsum = rows[("record", "ns-27b")]
    want = rsum["text_sha256"]
    text = None
    for (stage, src), (b, m) in rows.items():
        if stage == "extract" and m.get("text_sha256") == want:
            path = blob(b)["full_text_path"]
            t = Path(path).read_text(encoding="utf-8")
            if hashlib.sha256(t.encode()).hexdigest() == want:
                text = t; break
    if text is None:
        return None
    U = units(text)
    s2u = {(a, b): i for i, (a, b) in enumerate(U)}
    sec = sections(text, U)
    r = blob(nb)
    tasks = []
    for t in r.get("tasks") or []:
        nm = {e["vocabulary"]: e for e in (t.get("name") or {}).get("normalized") or []}
        tasks.append({
            "task": t["local_id"], "name": val(t.get("name")), "description": val(t.get("description")),
            "canonical": (nm.get("astra canonical task") or {}).get("value"),
            "family": (nm.get("astra task family") or {}).get("value"),
            "conditions": [{"condition": c["local_id"], "name": val(c.get("name")), "description": val(c.get("description"))}
                           for c in t.get("conditions") or []],
            "evidence_units": {k: cited(t.get(k), s2u) for k in ("name", "description", "instructions", "references")},
            "condition_units": sorted({u for c in t.get("conditions") or [] for k in ("name", "description")
                                       for u in cited(c.get(k), s2u)}),
        })
    analyses = []
    for a in r.get("analyses") or []:
        nm = {e["vocabulary"].split(" fit-")[0]: e for e in (a.get("name") or {}).get("normalized") or []}
        cs = nm.get("ns-pond contrast set") or {}
        analyses.append({
            "analysis": a["local_id"], "name": val(a.get("name")), "definition": val(a.get("definition")),
            "tasks": a.get("tasks") or [],
            "contrast_set": cs.get("value") if cs.get("status") == "mapped" else None, "contrast_set_label": cs.get("label"),
            "evidence_units": {k: cited(a.get(k), s2u) for k in ("name", "definition", "interpretations")},
            "interpretations": val(a.get("interpretations")),
        })
    meta = rows.get(("metadata", ""))
    md = blob(meta[0]) if meta and meta[0] else {}
    return {
        "article": art, "title": md.get("title"), "journal": md.get("journal"), "year": md.get("publication_year"),
        "text_sha256": want, "units": [{"i": i, "text": text[a:b], "section": sec[i]} for i, (a, b) in enumerate(U)],
        "tasks": tasks, "analyses": analyses,
        "hypothesis": val(r.get("hypothesis")), "hypothesis_units": cited(r.get("hypothesis"), s2u),
    }


def main():
    W = Path(sys.argv[1]); out = W / "out/pilot/papers"; out.mkdir(parents=True, exist_ok=True)
    inv = W / "out/inventory"
    by_canon = defaultdict(list)
    for line in open(inv / "tasks.jsonl"):
        t = json.loads(line)
        if t["canonical"]:
            by_canon[t["canonical"]].append(t)
    n_an = defaultdict(int); n_set = defaultdict(int)
    for line in open(inv / "analyses.jsonl"):
        a = json.loads(line)
        for tid in a["tasks"]:
            n_an[(a["article"], tid)] += 1
            n_set[(a["article"], tid)] += bool(a["set"])
    con = sqlite3.connect(CAT, uri=True)
    rng = random.Random(SEED)
    sel = []
    for canon, k in PICK:
        cands = by_canon[canon]
        strong = [t for t in cands if t["method"] in ("name", "citation") and n_an[(t["article"], t["task"])] >= 2
                  and n_set[(t["article"], t["task"])] >= 1]
        kind = "strong" if len(strong) >= k else "any"
        pool = strong if kind == "strong" else cands
        pool = sorted({t["article"]: t for t in pool}.values(), key=lambda t: t["article"])
        rng.shuffle(pool)
        got = 0
        for t in pool:
            if got == k:
                break
            if any(s["article"] == t["article"] for s in sel):
                continue
            b = bundle(con, t["article"])
            if b is None:
                continue
            b["pilot_task"] = {"canonical": canon, "task": t["task"], "pool": kind, "pool_size": len(pool)}
            (out / f'{t["article"]}.json').write_text(json.dumps(b))
            sel.append({"canonical": canon, "article": t["article"], "task": t["task"], "name": t["name"],
                        "method": t["method"], "units": len(b["units"]), "analyses": len(b["analyses"]),
                        "pool": kind, "pool_size": len(pool)})
            got += 1
    with open(W / "out/pilot/selection.jsonl", "w") as f:
        for s in sel:
            f.write(json.dumps(s) + "\n")
    for s in sel:
        print(s)


if __name__ == "__main__":
    main()
