"""A construct for every sample analysis, read from its paper (James 2026-11-12: "ensure all analyses in the sample
have a construct; read the full text, or the sections and sentences that mention the analysis").

For the sample analyses with no contrast-specific construct (constructs.py contrast_constructs.json), grouped by
paper:
  prep (beast)  for each paper, from the ns-pond catalog (read-only): the full text (extract's full_text_path, or the
                hub's imaging_papers copy), its title and abstract, and per analysis its sides and direction, its hub
                signature, the caption of the table its peaks come from (matched by coordinates within 2 mm), and up
                to 8 sentences that name its conditions -> OUT.jsonl, one paper per line
  run (laptop)  Claude through the CLI (claude_extract), the Cognitive Atlas concept list in the system prompt (the
                same on every call), the task's taxonomy constructs in the prompt; every analysis gets 1-3 constructs
                and a confidence -> OUT.json {analysis id: {constructs, isolates, confidence}}; resumable
Answers are checked against the Cognitive Atlas list and the task's constructs; others are dropped and counted.

  merge (beast) writes CONSTRUCTS_OUT: CONSTRUCTS with these analyses' constructs (tier "paper", with confidence)
                where they had no contrast-specific construct

Trial phases (James 2026-11-12, phase_scope.py): constructs were attributed per hub signature, and a signature can
hold analyses from different trial phases (MID anticipation and outcome). With ALL=1, prep takes every listed analysis
(SAMPLE.json is then phase_scope.py's list), not only those without a contrast-specific construct; every paper also
gets its trial-structure sentences, and every answer names the phase the analysis comes from. With OVERRIDE=<tier>,
merge replaces the listed analyses' constructs whatever their tier (the old ones kept as "fallback").

usage: python paper_attribute.py prep SAMPLE.json CONSTRUCTS.json CONTRASTS.json CORPUS.json HUB_DIR OUT.jsonl
       python paper_attribute.py run PAPERS.jsonl COGAT.json TAXONOMY.json OUT.json [MODEL EFFORT]
       python paper_attribute.py merge CONSTRUCTS.json PAPER_ATTRIBUTION.json CONSTRUCTS_OUT.json
"""
import collections
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CAT = Path("/data/alejandro/projects/ns-pond/catalog")
SPECIFIC = ("claims", "atlas", "codex", "task_contrast")
DESIGN = re.compile(r"\b(each trial|trials? (began|started|consisted)|cue|anticipat|outcome|feedback|encoding|retrieval|delay|"
                    r"maintenance|probe|epoch|phase|period|onset|regressors?|modell?ed|convolved)\b", re.I)
N_DESIGN = 8  # trial-structure sentences per paper
STOP = set("the and with for from that this were was are vs versus condition conditions trial trials task blocks block "
           "stimuli stimulus minus compared relative baseline".split())
RULES = """You read a neuroimaging paper's own description of its analyses and name, for each analysis, the cognitive
constructs it isolates, as Cognitive Atlas concepts. An analysis is a contrast between two sides (target > baseline),
or a one-sided simple effect, read in the direction given. Use the paper's text, the table caption and the sentences
given; the hub signature is a normalized reading of the sides and may be coarse.
- Name what the target side has that the baseline lacks; what both sides share is subtracted. A contrast against rest,
  fixation or a low-level baseline (or a simple effect) takes the task's constructs.
- A reversed analysis (baseline > target, rest > task, a negative or deactivation effect) gets what the baseline side
  has, never the target side's constructs: for 0-back > 2-back, not working memory. For rest > task or a deactivation,
  name what the paper says rest or the baseline engages; if it says nothing, the closest concept, confidence "low".
- Every analysis gets at least one construct. When it isolates nothing specific (left > right hand, a sensory
  control), give the closest concept that describes the difference (for example right hand response execution) and
  confidence "low". "high": the paper states what it isolates; "medium": the design makes it clear; "low": a best
  guess.
- Trial phases: when the paper models epochs of a trial separately (cue or anticipation, encoding or study, delay or
  maintenance, probe, retrieval or test, outcome or feedback), name the constructs of the epoch the analysis comes
  from, read from its name, its table caption and the paper's trial-structure sentences; the hub signature does not
  record the phase. Gain > no gain at the cue is reward anticipation, at the outcome reward or feedback processing;
  remembered > forgotten at encoding is memory encoding (a subsequent memory effect), at retrieval recognition or
  recollection. Give the phase as a word ("anticipation", "outcome", "encoding", "retrieval", "maintenance", "cue",
  "target", "whole trial", or "unknown" when the paper does not say).
- Use concept names exactly as listed in the Cognitive Atlas list below, or one of the task's constructs given with
  the paper. Most specific first, at most three.
Answer with the JSON object the schema defines, one item per analysis, with its id.

Cognitive Atlas concepts:
"""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["id", "phase", "isolates", "constructs", "confidence"],
    "properties": {"id": {"type": "integer"}, "phase": {"type": "string"}, "isolates": {"type": "string"},
                   "constructs": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string"}},
                   "confidence": {"type": "string", "enum": ["high", "medium", "low"]}}}}}}


def fold(s):
    return re.sub(r"[^a-z0-9+]+", " ", (s or "").lower()).strip()


def prep():
    import gzip
    import sqlite3
    samplep, consp, contrastsp, corpusp, hub_dir, outp = sys.argv[2:8]
    sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import hub as H
    S = json.load(open(samplep)); R = json.load(open(consp)); C = json.load(open(contrastsp))
    docs = json.load(open(corpusp))["docs"]
    hubj = json.load(open(f"{hub_dir}/hub.json")); entries = {**hubj.get("parents", {}), **hubj["contrasts"]}
    assign = json.load(open(f"{hub_dir}/corpus_assign.json"))
    inv = {t["id"]: t["name"] for t in H.INVENTORY["tasks"]}
    need = list(S) if os.environ.get("ALL") else [a for a in S if not ((R.get(a) or {}).get("constructs") and ((R[a].get("tier") or "") in SPECIFIC or (R[a].get("tier") or "").startswith("hierarchy_")))]
    by_paper = collections.defaultdict(list)
    for a in need:
        by_paper[C[a]["paper"]].append(a)
    db = sqlite3.connect(f"file:{CAT}/catalog.sqlite?mode=ro", uri=True)
    alias = collections.defaultdict(list)
    for art, val in db.execute("select article_id, value from aliases where kind='neurostore'"):
        alias[val].append(art)
    root = Path(__file__).resolve().parents[2]
    n = collections.Counter()
    with open(outp, "w") as fo:
        for p, aids in by_paper.items():
            text, tables = "", []
            for art in alias.get(p, []):
                row = db.execute("select blob from artifacts where article_id=? and stage='extract' and status='ok' order by updated_at desc limit 1", (art,)).fetchone()
                if row and row[0]:
                    ex = json.load(gzip.open(CAT / "blobs" / row[0][:2] / f"{row[0]}.json.gz"))
                    tables = ex.get("tables") or []
                    try:
                        text = open(ex["full_text_path"], errors="replace").read()
                    except Exception:  # noqa: BLE001
                        text = ""
                if not text and (root / "imaging_papers" / f"ns_{art}.json").exists():
                    text = "\n".join(u["text"] for u in json.load(open(root / "imaging_papers" / f"ns_{art}.json"))["units"])
                if text:
                    break
            n["papers"] += 1; n["papers with text"] += bool(text)
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])|\n+", text) if 30 <= len(s.strip()) <= 600]
            fs = [set(fold(s).split()) for s in sents]
            dz = sorted(((len(DESIGN.findall(s)), i) for i, s in enumerate(sents) if len(DESIGN.findall(s)) >= 2), reverse=True)[:N_DESIGN]
            design = [sents[i][:400] for _, i in sorted(dz, key=lambda x: x[1])]
            items = []
            for a in aids:
                c = C[a]; asg = assign.get(a) or {}
                k = asg.get("hub"); sign = asg.get("sign", 1)
                sig = k if k else "(not normalized)"
                if k and sign < 0:
                    sig = f"{k} REVERSED (baseline > target)"
                words = {w for w in fold(c.get("pos", "") + " " + c.get("neg", "")).split() if len(w) >= 3 and w not in STOP}
                scored = sorted(((len(words & f), i) for i, f in enumerate(fs) if words & f), reverse=True)[:8]
                foci = {tuple(round(float(x)) for x in f[:3]) for f in docs.get(a, {}).get("foci", [])}
                cap = ""
                for t in tables:
                    tc = {(round(float(x.get("x", 0))), round(float(x.get("y", 0))), round(float(x.get("z", 0)))) for x in t.get("coordinates") or []}
                    if foci and tc and sum(any(max(abs(u - v) for u, v in zip(f, g)) <= 2 for g in tc) for f in foci) >= 0.5 * len(foci):
                        cap = (t.get("caption") or "")[:400]; break
                n["analyses"] += 1; n["with table caption"] += bool(cap); n["with sentences"] += bool(scored)
                items.append({"aid": a, "task": inv.get(c["tasks"].split(",")[0], c["tasks"]), "task_id": c["tasks"].split(",")[0],
                              "target": c.get("pos", ""), "baseline": c.get("neg", ""), "kind": c.get("kind"), "signature": sig,
                              "caption": cap, "sentences": [sents[i][:400] for _, i in sorted(scored, key=lambda x: x[1])]})
            fo.write(json.dumps({"paper": p, "head": text[:1500], "design": design, "analyses": items}) + "\n")
    print(json.dumps(dict(n), indent=1))


def run():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import claude_extract as cx
    papersp, cogp, taxp, outp = sys.argv[2:6]
    model, effort = (sys.argv[6], sys.argv[7]) if len(sys.argv) > 7 else ("claude-opus-5-5", "medium")
    cog = sorted({c["name"].strip() for c in json.load(open(cogp)) if c.get("name")}, key=str.lower)
    key = lambda s: s.strip().lower().replace("-", " ")  # noqa: E731
    cogk = {key(c): c for c in cog}
    tax = json.load(open(taxp)); task_c = {t: v.get("constructs") or [] for t, v in tax["tasks"].items()}
    system = RULES + "; ".join(cog)
    done = json.load(open(outp)) if os.path.exists(outp) else {}
    papers = [json.loads(x) for x in open(papersp)]
    if os.environ.get("LIMIT"):
        papers = papers[:int(os.environ["LIMIT"])]
    papers = [p for p in papers if any(it["aid"] not in done for it in p["analyses"])]
    print(f"{len(papers)} papers, {sum(len(p['analyses']) for p in papers)} analyses", flush=True)
    bad = collections.Counter()

    def one(p):
        tasks = sorted({it["task_id"] for it in p["analyses"]})
        own = sorted({c for t in tasks for c in task_c.get(t, [])})
        lines = [f"Paper (title and abstract): {p['head']}", ""]
        if p.get("design"):
            lines += ["Trial structure (sentences from the paper):"] + [f"  - {s}" for s in p["design"]] + [""]
        lines += [f"Task constructs: {', '.join(own) or '(none)'}", ""]
        for i, it in enumerate(p["analyses"]):
            lines.append(f"id {i}: task {it['task']}; {it['kind']}: {it['target'] or '(none)'} > {it['baseline'] or '(baseline / none)'}; "
                         f"hub signature {it['signature']}")
            if it["caption"]:
                lines.append(f"  table caption: {it['caption']}")
            for s in it["sentences"]:
                lines.append(f"  - {s}")
        try:
            ans, _, secs = cx.call(model, effort, system, "\n".join(lines), (), schema=SCHEMA)
            return p, ans["items"], own
        except Exception as exc:  # noqa: BLE001
            print("FAILED", p["paper"], str(exc)[:200], flush=True); return p, None, own
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as pool:
        for k, (p, res, own) in enumerate(pool.map(one, papers)):
            if res is None:
                continue
            ok = {key(c): c.lower().replace(" ", "-") for c in own} | {k2: v.lower().replace(" ", "-") for k2, v in cogk.items()}
            for it in res:
                if not 0 <= it["id"] < len(p["analyses"]):
                    continue
                cs = []
                for c in it["constructs"]:
                    if key(c) in ok:
                        cs.append(ok[key(c)])
                    else:
                        bad["not a listed concept"] += 1
                done[p["analyses"][it["id"]]["aid"]] = {"constructs": cs[:3], "phase": it["phase"], "isolates": it["isolates"],
                                                        "confidence": it["confidence"], "paper": p["paper"]}
            if k % 20 == 0:
                json.dump(done, open(outp, "w"))
                print(k, len(papers), len(done), dict(bad), flush=True)
    json.dump(done, open(outp, "w"))
    conf = collections.Counter(v["confidence"] for v in done.values())
    print(json.dumps({"analyses": len(done), "with a construct": sum(bool(v["constructs"]) for v in done.values()),
                      "confidence": dict(conf), "dropped answers": dict(bad)}, indent=1))


def merge():
    consp, attrp, outp = sys.argv[2:5]
    R = json.load(open(consp)); A = json.load(open(attrp)); n = collections.Counter()
    override = os.environ.get("OVERRIDE")   # a tier name: replace whatever the analysis had
    for a, v in A.items():
        r = R.get(a) or {}
        t = r.get("tier") or ""
        if not override and r.get("constructs") and (t in SPECIFIC or t.startswith("hierarchy_")):
            n["kept existing"] += 1; continue
        if not v["constructs"]:
            n["no listed construct"] += 1; continue
        R[a] = {**r, "constructs": v["constructs"], "tier": override or "paper", "confidence": v["confidence"],
                "fallback": r.get("constructs"), **({"phase": v["phase"]} if v.get("phase") else {})}
        n[f"{override or 'paper'} {v['confidence']}"] += 1
    json.dump(R, open(outp, "w")); print(json.dumps(dict(n)))


if __name__ == "__main__":
    {"prep": prep, "run": run, "merge": merge}[sys.argv[1]]()
