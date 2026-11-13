"""Silver set for the claim classifier: stage-1 sentences of new PMC papers, rendered for an LLM judge, plus the
same rendering for gold papers so the judge can be scored against the adjudicated gold first.

- New papers: PMC texts in the construct-claims catalog (extract ok, source pmc) found by exactly one task search
  (data/article_tasks.jsonl), not a gold paper, 15k-120k characters, with Results and Discussion sections. One
  paper per task for N tasks drawn at random (seed). Units from build_pilot.units, sections from ns-pond's
  sectionizer (services/coordinate_space.sectionize).
- Gold papers: the pilot bundles of the given folds (groups.json), targets as in offtarget.py.

Writes silver/papers/<id>.json ({article, targets, target_names, units, candidates}) and silver/render/<id>.md.

usage: python build_silver.py N_TASKS SEED GOLD_FOLDS   (beast, from pilot-results/; PYTHONPATH = code/:ns-pond clone)
       e.g. python build_silver.py 60 20261110 0,1
"""
import gzip
import json
import random
import sqlite3
import sys
from pathlib import Path

from build_pilot import units as split_units
from ingestion_workflow.services.coordinate_space import sectionize
from offtarget import Checker, mentions
from stage1 import keep

ROOT = Path.cwd()
CC = Path("/data2/james/construct-claims")
SECTION = {"intro": "introduction", "back": "back", "data_availability": "back", "unknown": "front"}


def blob(d):
    return json.loads(gzip.decompress((CC / "catalog" / "blobs" / d[:2] / f"{d}.json.gz").read_bytes()))


def gold_cc_ids():
    """construct-claims article ids of the 36 gold papers, through shared aliases"""
    prod = sqlite3.connect("file:/data/alejandro/projects/ns-pond/catalog/catalog.sqlite?mode=ro", uri=True)
    cc = sqlite3.connect(f"file:{CC}/catalog/catalog.sqlite?mode=ro", uri=True)
    out = set()
    for f in (ROOT / "pilot" / "papers").glob("*.json"):
        for k, v in prod.execute("select kind, value from aliases where article_id=?", (f.stem,)):
            out |= {b for (b,) in cc.execute("select article_id from aliases where kind=? and value=?", (k, v))}
    return out


def render(article, names, others, units, cand):
    by_i = {u["i"]: k for k, u in enumerate(units)}
    lines = [f"# Paper {article}", f"Searched task: {'; '.join(names)}",
             f"Other tasks this paper names: {', '.join(sorted(others)) or 'none'}", "",
             "Sentences to judge: [number] (section) «end of the previous sentence» SENTENCE", ""]
    shown = set()
    for i in cand:
        k = by_i[i]
        prev = units[k - 1]["text"][-200:] if k and units[k - 1]["i"] not in shown else ""  # not when just above
        lines.append(f"[{i}] ({units[k]['section']}) «{prev}» {units[k]['text']}")
        shown.add(i)
    return "\n".join(lines) + "\n"


def write(article, targets, names, units, ck, text):
    rxs = ck.paper(text)
    cand = keep(units, targets, rxs)
    others = {n for u in units for n in mentions(u["text"], rxs) - targets}
    others = {ck.names.get(t, t) for t in others}
    (ROOT / "silver" / "papers" / f"{article}.json").write_text(json.dumps(
        {"article": article, "targets": sorted(targets), "target_names": names, "units": units, "candidates": cand}))
    (ROOT / "silver" / "render" / f"{article}.md").write_text(render(article, names, others, units, cand))
    return len(units), len(cand)


def main():
    n_tasks, seed, folds = int(sys.argv[1]), int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")]
    for d in ("papers", "render"):
        (ROOT / "silver" / d).mkdir(parents=True, exist_ok=True)
    ck = Checker(ROOT / "code" / "task_inventory_v3.json", ROOT / "code" / "queries.jsonl")
    # gold papers for scoring the judge
    groups = json.load(open(ROOT / "groups.json")); search = json.load(open(ROOT / "pilot" / "search_tasks.json"))
    for a in [a for k in folds for a in groups[k]]:
        p = json.load(open(ROOT / "pilot" / "papers" / f"{a}.json"))
        targets = set(search.get(a) or ({t["canonical"] for t in p["tasks"] if t["canonical"]} |
                                        {p["pilot_task"]["canonical"]}))
        units = [{"i": u["i"], "text": u["text"], "section": u["section"]} for u in p["units"]]
        n, c = write(a, targets, [ck.names[t] for t in sorted(targets) if t in ck.names], units, ck,
                     " ".join(u["text"] for u in units))
        print(f"gold {a}: {c}/{n} units to judge", flush=True)
    # new PMC papers
    con = sqlite3.connect(f"file:{CC}/catalog/catalog.sqlite?mode=ro", uri=True)
    text_of = {}
    for art, b in con.execute("select article_id, blob from artifacts where stage='extract' and status='ok'"):
        d = blob(b)
        if d.get("source") == "pmc":
            text_of[art] = d["full_text_path"]
    gold = gold_cc_ids()
    by_task = {}
    for l in open(CC / "data" / "article_tasks.jsonl"):
        r = json.loads(l); ts = {t for t, _ in r["found"]}
        if len(ts) == 1 and r["article"] in text_of and r["article"] not in gold and next(iter(ts)) in ck.names:
            by_task.setdefault(next(iter(ts)), []).append(r["article"])
    rng = random.Random(seed)
    tasks = sorted(t for t, v in by_task.items() if len(v) >= 5)
    print(f"{len(text_of)} PMC texts; {len(tasks)} tasks with >= 5 single-task PMC papers", flush=True)
    rng.shuffle(tasks)
    picked = 0
    for t in tasks:
        if picked == n_tasks:
            break
        arts = by_task[t][:]; rng.shuffle(arts)
        for a in arts[:20]:
            text = Path(text_of[a]).read_text(encoding="utf-8")
            if not 15000 <= len(text) <= 120000:
                continue
            spans = sectionize(text); labels = {lab for _, _, lab in spans}
            if not {"results", "discussion"} <= labels:
                continue
            units = []
            for k, (x, y) in enumerate(split_units(text)):
                lab = next((lab for s, e, lab in spans if s <= x < e), "unknown")
                units.append({"i": k, "text": text[x:y], "section": SECTION.get(lab, lab)})
            n, c = write(a, {t}, [ck.names[t]], units, ck, text)
            print(f"new {a} {t}: {c}/{n} units to judge", flush=True)
            picked += 1
            break


if __name__ == "__main__":
    main()
