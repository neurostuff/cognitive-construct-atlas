"""Constructs for hub contrasts and, through them, for every corpus contrast.

Evidence per hub entry, strongest first:
1. claims: construct-claim sentences whose contrast matches the entry, from the claim catalog's full-text scan
   (scan_contrasts.py output, re-matched to this hub), the Codex-labelled sample (match_claims.py), and claims that
   name the contrast as an effect (named_effects.py, next to CLAIMS_MATCH as named_effects.jsonl). A claim's
   construct strings map to vocabulary entries (vocab/string_to_entries.json) and so to taxonomy constructs (id = the
   entry key with spaces as hyphens). Counted in articles. With OUT_DIR/stance_labels.json (stance.py), a claim's
   construct is credited to the direction its stance gives (A: the claim's own, B: the reverse) or dropped (ruled
   out, the task in general, unrelated).
2. atlas: the Cognitive Atlas concepts asserted for the entry's Atlas contrasts.
3. codex: for established entries with neither, Codex picks the constructs the contrast isolates from the taxonomy
   constructs linked to the entry's tasks (codex_attribute, below), or none.
A corpus contrast takes its exact entry's constructs (contrast-specific); failing those, its emotion -> valence
hierarchy parent's (contrast-specific); a task-vs-rest entry takes its task's (task_contrast); then a two-sided
component's ("related", reported apart) or the axis's ("axis", not a contrast reading); and failing those, its task's constructs (the taxonomy's task -> construct edges, then vocab/task_constructs_pass.json),
and for a contrast labelled only with a family, the family's most common constructs: the weakest tiers. A
task-vs-baseline entry (`task:<id>|@task>@base`) takes its task's constructs as its own (tier task_contrast): the
contrast is the task. Claim evidence for an entry is also evidence for its backoff parents.

usage: python constructs.py HUB_DIR CONTRASTS.json 'SCAN_GLOB[+SCAN_GLOB]' CLAIMS_MATCH.json LEXICON.json OUT_DIR [--codex]
(run from pilot-results/code/contrast_hub on beast; reads ../../vocab and ../../atm/artifact_data_v2)
"""
import collections
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hub as H
import match_claims as M

ROOT = Path(__file__).resolve().parents[2]


def vocab_maps():
    s2e = json.load(open(ROOT / "vocab" / "string_to_entries.json"))
    tax = json.load(open(ROOT / "atm" / "artifact_data_v2" / "taxonomy.json"))
    ids = {c["id"]: c for c in tax["constructs"]}
    by_key = {}
    for c in tax["constructs"]:
        by_key[c["id"].replace("-", " ")] = c["id"]; by_key[c["label"].lower()] = c["id"]
        if c.get("cogat_name"):
            by_key.setdefault(c["cogat_name"].lower(), c["id"])

    def to_ids(strings):
        out = set()
        for s in strings:
            for e in s2e.get(s, []) or s2e.get(s.lower(), []) or [s.lower()]:
                if e in by_key:
                    out.add(by_key[e])
        return out
    return to_ids, ids, tax


#: A contrast and its reverse isolate different functions (go > no-go is not response inhibition), so construct
#: evidence and attribution are kept per direction. A hub entry is stored in one direction; a contrast or claim that
#: runs the other way (sign -1 from match_one / corpus_assign: old > new filed under new > old, rest > task under
#: task > rest) is the directed key `rev:<entry>`.
REV = "rev:"


def directed(k, sign):
    return k if sign >= 0 else REV + k


def undirected(dk):
    return (dk[len(REV):], -1) if dk.startswith(REV) else (dk, 1)


def shown(dk, entries):
    """a directed key as Codex reads it: a reversed entry is written in its own direction when the hub can flip it,
    else marked REVERSED (a baseline-first contrast: rest > task, or a negative simple effect)"""
    k, s = undirected(dk)
    if s > 0 or k not in entries:
        return k
    f = H.flip(H._sig_from_json(entries[k]["sig_json"]))
    return H.sig_str(f) if f is not None else f"{k} REVERSED"


def up(hub, k):
    """an exact entry and its backoff parents, as (key, kind, sign of the parent relative to the entry): a claim
    about fear > neutral is also evidence for negative > neutral (hierarchy), for a two-sided component of a
    multi-component signature (component), and for the axis (axis)"""
    e = hub.entries.get(k)
    if e is None:
        return [(k, "exact", 1)]
    out = [(k, "exact", 1)]
    for p, kind in H.parents(H._sig_from_json(e["sig_json"])):
        for q, qs in ((p, 1), (H.flip(p), -1)):
            if q is not None and H.sig_str(q) in hub.parents:
                out.append((H.sig_str(q), kind, qs)); break
    return out


def credit(stance, sid, construct, sign):
    """the direction a claim's construct is credited to, from its stance label (stance.py: A = the claim's own
    direction, B = its reverse; none / task / unrelated = not credited to the contrast), or its own direction when
    unlabelled"""
    lab = stance.get(sid, {}).get(construct, "A") if stance is not None else "A"
    return sign if lab == "A" else -sign if lab == "B" else 0


def is_task_claim(stance, sid, construct):
    """the claim says the construct of the task as a whole (stance `task`), whatever the contrast's direction"""
    return stance is not None and stance.get(sid, {}).get(construct) == "task"


def task_of(k):
    """the task (or `family:<name>`) an entry key is scoped to"""
    t = k.split("|")[0]
    return t[5:] if t.startswith("task:") else t


def claim_evidence(hub, lex, scan_glob, claims_match, to_ids, stance=None):
    ev = collections.defaultdict(lambda: collections.defaultdict(set))
    task_ev = collections.defaultdict(lambda: collections.defaultdict(set))   # whole-task claims, per task
    n = collections.Counter()

    def add_task(sid, s, hub_key, article):
        if is_task_claim(stance, sid, s):
            for c in to_ids([s]):
                task_ev[task_of(hub_key)][c].add(article)
    for f in sorted(f for pat in scan_glob.split("+") for f in glob.glob(pat)):   # several scans joined by +
        for line in open(f):
            r = json.loads(line)
            if "constructs" not in r:
                continue
            m = M.match_one({"tasks": r["tasks"], "a": r["a"], "b": r["b"]}, hub, lex)
            n["scan_claim_units"] += 1
            if "hub" in m:
                n["scan_claim_matched"] += 1; n["scan_claim_reversed"] += m["sign"] < 0
                sid = f"{r['article']}:{r['i']}|{m['hub']}"
                for s in r["constructs"]:
                    d = credit(stance, sid, s, m["sign"]); n[f"scan_construct_{d:+d}"] += 1
                    add_task(sid, s, m["hub"], r["article"])
                    if d == 0:
                        continue
                    for k, _, ps in up(hub, m["hub"]):
                        for c in to_ids([s]):
                            ev[directed(k, d * ps)][c].add(r["article"])
    named = Path(claims_match).with_name("named_effects.jsonl")
    if named.exists():                      # named_effects.py: claims that name a contrast as an effect
        for line in open(named):
            r = json.loads(line); n["named_effect_claims"] += 1
            sid = f"{r['article']}:{r['i']}|{r['hub']}"
            for s in r["constructs"]:
                d = credit(stance, sid, s, 1 if r.get("sign", 1) > 0 else -1); n[f"named_construct_{d:+d}"] += 1
                add_task(sid, s, r["hub"], r["article"])
                if d == 0:
                    continue
                for k, _, ps in up(hub, r["hub"]):
                    for c in to_ids([s]):
                        ev[directed(k, d * ps)][c].add(r["article"])
    for r in json.load(open(claims_match))["claims"]:
        m = M.match_one({"tasks": r["tasks"], "a": r["a"], "b": r["b"]}, hub, lex)
        if "hub" in m and r.get("construct_link"):
            n["sample_claim_matched"] += 1
            sid = f"{r['article']}:{r.get('i', r['id'])}|{m['hub']}"
            for s in r["constructs"]:
                d = credit(stance, sid, s, m["sign"]); n[f"sample_construct_{d:+d}"] += 1
                add_task(sid, s, m["hub"], r["article"])
                if d == 0:
                    continue
                for k, _, ps in up(hub, m["hub"]):
                    for c in to_ids([s]):
                        ev[directed(k, d * ps)][c].add(r["article"])
    return ev, task_ev, n


#: a construct counts as an entry's claim evidence when this many articles state it (one article is often a passing
#: mention); an axis-level entry (`family|axis:*`) pools every comparison on the axis and needs more
MIN_ARTICLES, MIN_ARTICLES_AXIS = 2, 3

RULES = """You link contrasts from neuroimaging studies to the cognitive constructs they isolate. A contrast is written
`task|axis:target>baseline` (`@rest` = rest/fixation/low-level baseline; `@control` = a matched control condition;
`∅` = the axis has no value on that side; a component without `>` is a simple effect of that condition). Example side
names from the papers follow it.
For each contrast, first state in `isolates` what the target side has that the baseline lacks: what the two sides
share is subtracted and is not attributed. Then pick, from the candidate constructs only, the 1 to 3 constructs of
that difference, most specific first. For a compound signature give at least one construct per axis. Only for a
contrast against `@rest` or a one-sided simple effect may the task's own constructs be used. When no candidate names
the isolated difference, give no constructs: never fall back on a construct that only labels the task as a whole.
Direction matters: a contrast and its reverse isolate different functions (no-go > go is response inhibition, go >
no-go is not). Read each contrast in the direction written; one marked REVERSED runs baseline > target (rest > task: a
deactivation; a negative simple effect), so attribute what the baseline side has that the target lacks, or none.
Answer with the JSON object the schema defines, one item per contrast, with its id."""
#: the Cognitive Atlas concept list (its concepts.json export), the extra candidates of a reversed entry
COGAT = Path(__file__).resolve().parent / "cogat_concepts.json"
#: extra candidates for a reversed contrast against rest (a deactivation), whose constructs are rarely the task's own
DEACTIVATION = ["self-referential", "autobiographical-memory", "theory-of-mind"]
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["id", "isolates", "constructs"],
    "properties": {"id": {"type": "integer"}, "isolates": {"type": "string"}, "constructs": {"type": "array", "items": {"type": "string"}}}}}}}


def codex_attribute(entries, hubj, tax, out_path, C=None, assign=None, model="gpt-6.1-sol", effort="low", batch=50):
    """Codex picks, for each directed entry (an entry key, or `rev:<key>` for its reverse), the constructs its
    comparison isolates, from the taxonomy constructs linked to its task and family (one candidate list per task,
    shared by the batch). Each is shown in its own direction with up to three example side names from its contrasts
    that run that way. A reversed entry's candidates also hold every Cognitive Atlas concept (COGAT, as hyphenated
    ids like the taxonomy's): what its baseline side has (houses > faces: visual scene perception; forgotten >
    remembered: forgetting) is rarely among the task's constructs, and often not in the taxonomy."""
    sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import importlib
    cx = importlib.import_module("claude_extract" if os.environ.get("BACKEND") == "claude" else "codex_extract")   # BACKEND=claude: the Claude CLI
    by_task = collections.defaultdict(set)
    for c in tax["constructs"]:
        for t in c.get("tasks", []):
            by_task[t].add(c["id"])
    fam_tasks = collections.defaultdict(collections.Counter)
    for e in hubj["contrasts"].values():
        fam_tasks[e["family"]].update({t: n for t, n in e["tasks"].items() if t in by_task})
    E = hubj["contrasts"]
    scope_of = lambda dk: E[undirected(dk)[0]].get("scope", E[undirected(dk)[0]]["family"])  # noqa: E731
    ids = {c["id"] for c in tax["constructs"]}
    cogat = sorted({c["name"].strip().lower().replace(" ", "-") for c in json.load(open(COGAT)) if c.get("name")})

    def group(dk):
        """the batch and candidate list: the task, or (task, "reversed")"""
        sc = scope_of(dk)
        if undirected(dk)[1] < 0:
            return (sc, "reversed")
        return (sc, "cogat") if os.environ.get("COGAT_ALL") else sc   # COGAT_ALL=1: forward entries get the list too
    cand = {}
    for dk in entries:                                 # candidates: the entry's own task's constructs, then its family's
        g = group(dk); sc = scope_of(dk); fam = E[undirected(dk)[0]]["family"]
        if g not in cand:
            own = sorted(by_task.get(sc, ()))
            fam_c = sorted({c for t, _ in fam_tasks[fam].most_common(12) for c in by_task[t]} - set(own))
            cand[g] = (own + fam_c)[:150]
            cand[g] += [c for c in DEACTIVATION if c in ids and c not in cand[g]]
            if isinstance(g, tuple):                   # reversed (or COGAT_ALL): the whole Cognitive Atlas besides
                cand[g] += [c for c in cogat if c not in cand[g]]
    examples = collections.defaultdict(list)
    if C and assign:
        for aid, a in assign.items():
            dk = directed(a["hub"], a.get("sign", 1))
            if len(examples[dk]) < 3:
                examples[dk].append(f"{C[aid]['pos'].split(' | ')[0][:40]} vs {C[aid]['neg'].split(' | ')[0][:40] or '-'}")
    done = json.load(open(out_path)) if Path(out_path).exists() else {}
    iso_path = str(out_path).replace(".json", "_isolates.json")
    isolates = json.load(open(iso_path)) if Path(iso_path).exists() else {}
    todo = [k for k in entries if k not in done and cand.get(group(k))]
    by_fam = collections.defaultdict(list)
    for k in todo:
        by_fam[group(k)].append(k)
    batches = [(f, ks[i:i + batch]) for f, ks in by_fam.items() for i in range(0, len(ks), batch)]
    disabled = [t for t in cx.TOOLS if t in cx.known_features()]
    print(f"codex attribution: {len(todo)} entries in {len(batches)} batches", flush=True)

    def run(b):
        f, ks = b
        prompt = (f"task: {f[0] if isinstance(f, tuple) else f}\ncandidate constructs: {', '.join(cand[f])}\n\n" +
                  "\n".join(f"id {i}: {shown(k, E)} (e.g. {'; '.join(examples.get(k, []))})" for i, k in enumerate(ks)))
        try:
            ans, _, secs = cx.call(model, effort, RULES, prompt, disabled, schema=SCHEMA)
            return b, ans["items"]
        except Exception as exc:  # noqa: BLE001
            print("FAILED", str(exc)[:200], flush=True); return b, None
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as pool:
        for b, res in pool.map(run, batches):
            if res is None:
                continue
            f, ks = b
            for it in res:
                if 0 <= it["id"] < len(ks):
                    done[ks[it["id"]]] = [c for c in it["constructs"] if c in cand[f]][:3]
                    isolates[ks[it["id"]]] = it.get("isolates", "")
            json.dump(done, open(out_path, "w")); json.dump(isolates, open(iso_path, "w"))
    return done


def main():
    hub_dir, contrasts, scan_glob, claims_match, lexp, out_dir = sys.argv[1:7]
    use_codex = "--codex" in sys.argv
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    hubj = json.load(open(f"{hub_dir}/hub.json")); hub = H.Hub(hubj["contrasts"], hubj.get("parents"))
    allent = {**hubj.get("parents", {}), **hubj["contrasts"]}
    assign = json.load(open(f"{hub_dir}/corpus_assign.json")); C = json.load(open(contrasts))
    lex = H.Lexicon(lexp)
    to_ids, ids, tax = vocab_maps()
    stance_p = out / "stance_labels.json"            # stance.py labels (Codex, else NLI), next to the output
    stance = json.load(open(stance_p)) if stance_p.exists() else None
    ev, task_ev, n = claim_evidence(hub, lex, scan_glob, claims_match, to_ids, stance)
    task_claims = {t: {c: len(a) for c, a in sorted(d.items(), key=lambda x: -len(x[1])) if len(a) >= MIN_ARTICLES}
                   for t, d in task_ev.items()}
    task_claims = {t: d for t, d in task_claims.items() if d}
    print(json.dumps(dict(n)), flush=True)
    entry_c = {}                                       # per directed key
    for dk in set(allent) | set(ev):
        k, s = undirected(dk); e = allent.get(k)
        if e is None:
            continue
        floor = MIN_ARTICLES_AXIS if e.get("kind") == "axis" else MIN_ARTICLES
        cl = {c: len(a) for c, a in ev.get(dk, {}).items() if len(a) >= floor}
        at = to_ids([x for cg in e.get("cogat", []) if cg.get("sign", 1) == s for x in cg["concepts"]])
        if cl or at:
            entry_c[dk] = {"claims": cl, "atlas": sorted(at)}
    used = collections.Counter(directed(a["hub"], a.get("sign", 1)) for a in assign.values())
    need = [dk for dk, _ in used.most_common() if undirected(dk)[0] in hubj["contrasts"] and dk not in entry_c
            and not dk.startswith("task:")]           # a task-vs-rest entry is its task; its reverse is attributed
    codex = codex_attribute(need, hubj, tax, str(out / "codex_attribution.json"), C, assign) if use_codex else (
        json.load(open(out / "codex_attribution.json")) if (out / "codex_attribution.json").exists() else {})
    task_c = json.load(open(ROOT / "vocab" / "task_constructs_pass.json"))
    tax_task = {t: [c for c in v.get("constructs") or [] if c in ids] for t, v in tax["tasks"].items()}
    fam_c = collections.defaultdict(collections.Counter)
    for t, v in tax["tasks"].items():
        for c in tax_task[t]:
            fam_c[v.get("family")][c] += v.get("contrasts", 1)
    task_ids = {}
    for t, v in task_c.items():   # most distinctive first (log-odds z)
        seen = []
        for x in sorted(v, key=lambda x: -x.get("z", 0)):
            for c in to_ids([x["entry"]]):
                if c not in seen:
                    seen.append(c)
        task_ids[t] = seen
    for t, cs in tax_task.items():          # the taxonomy's own task -> construct edges first
        if cs:
            task_ids[t] = cs + [c for c in task_ids.get(t, []) if c not in cs]

    def task_fallback(t):
        if task_claims.get(t):                # what papers say the task as a whole involves
            return list(task_claims[t])[:3], "task_claims"
        if task_ids.get(t):
            return task_ids[t][:3], "task"
        fam = t[7:] if t.startswith("family:") else t if t in fam_c else H.TASK_FAMILY.get(t)
        if task_claims.get(f"family:{fam}"):
            return list(task_claims[f"family:{fam}"])[:3], "family_claims"
        if fam in fam_c:
            return [c for c, _ in fam_c[fam].most_common(3)], "family"
        return [], None

    def of_entry(k):
        if k in entry_c:
            d = entry_c[k]
            if d["claims"]:
                return sorted(d["claims"], key=lambda c: -d["claims"][c])[:3], "claims"
            return d["atlas"][:3], "atlas"
        if codex.get(k):
            return codex[k], "codex"
        return None, None

    res = {}; tiers = collections.Counter()
    for aid, a in assign.items():
        k = a["hub"]; s = a.get("sign", 1); cs, tier = of_entry(directed(k, s))
        if not cs and k.startswith("task:") and s > 0:
            cs, tier = task_fallback(k.split("|")[0][5:])
            tier = "task_contrast" if cs else None
        if not cs:
            for pk, kind, ps in up(hub, k)[1:]:
                cs, tier = of_entry(directed(pk, s * ps))
                if cs:
                    tier = {"hierarchy": "hierarchy", "family": "sibling", "component": "related", "axis": "axis"}[kind] + "_" + tier; break
        if not cs and s > 0:          # a reversed contrast never takes its task's constructs: they describe task > base
            cs, tier = task_fallback(C[aid]["tasks"].split(",")[0])
        res[aid] = {"hub": k, "sign": s, "constructs": cs or [], "tier": tier or ("reversed_none" if s < 0 else None)}
        tiers[tier or ("reversed_none" if s < 0 else "none")] += 1
    for aid in C:
        if aid not in res:
            cs, tier = task_fallback(C[aid]["tasks"].split(",")[0])
            tier = (tier + "_unmatched") if cs else None
            res[aid] = {"hub": None, "constructs": cs, "tier": tier}
            tiers[tier or "none"] += 1
    N = len(C)
    contrast_level = sum(v for t, v in tiers.items() if t in ("claims", "atlas", "codex", "task_contrast") or t.startswith("hierarchy_"))
    related = sum(v for t, v in tiers.items() if t.startswith("related_"))
    summary = {"contrasts": N, "tiers": dict(tiers), "contrast_specific": contrast_level,
               "share_contrast_specific": round(contrast_level / N, 3),
               "related_entry": related, "share_specific_or_related": round((contrast_level + related) / N, 3),
               "share_any_construct": round(sum(v for t, v in tiers.items() if t not in ("none", "reversed_none")) / N, 3),
               "reversed_contrasts": sum(r.get("sign", 1) < 0 for r in res.values()),
               "entries_with_claim_constructs": sum(bool(d["claims"]) for d in entry_c.values()),
               "entries_with_atlas": sum(bool(d["atlas"]) for d in entry_c.values()), "entries_codex": len(codex),
               "tasks_with_task_claims": len(task_claims), **dict(n)}
    json.dump({"summary": summary, "entries": entry_c, "codex": codex, "task_claims": task_claims}, open(out / "entry_constructs.json", "w"), indent=1)
    json.dump(res, open(out / "contrast_constructs.json", "w"))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
