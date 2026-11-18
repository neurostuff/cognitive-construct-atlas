"""Task -> construct table from the claims and vocabulary v0.

Counts are papers: a task's paper claims construct entry e when one of its kept claim sentences (single task, not
ambiguous) has e as its top construct span (vocab/string_to_entries.json; a list span counts for each part). A
variant entry ("working memory load") counts toward its base ("working memory"); stated parents stay apart. A
paper that uses a task variant (vocab/task_variants.json: "emotional" Stroop) counts for "<task>::<variant>". Distinctiveness is the weighted log-odds
with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008, Political Analysis 16(4):372-403, eq. 22):
the prior is the entry's share over all tasks times A0; z = delta / sqrt(variance). A construct passes for a task
when z >= Z and it is claimed in >= MINP of the task's papers; at most TOP pass per task. A task with none passing
falls back to its FALLBACK_TOP most-claimed constructs in >= MINP papers ("how": "fallback"), so a small task whose
construct is common elsewhere (Sternberg: working memory) still gets one.
Writes vocab/task_constructs.tsv (every task x entry with >= 2 papers) and vocab/task_constructs_pass.json.

usage: python task_constructs.py [Z] [MINP] [TOP]   (from pilot-results/; defaults 2.0 5 10)
"""
import collections
import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path.cwd()
V = ROOT / "vocab"
A0 = 1000.0
FALLBACK_TOP = 5


def main():
    Z = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    MINP = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    s2e = json.load(open(V / "string_to_entries.json"))
    entries = {r["key"]: r for r in json.load(open(V / "vocab_v0.json"))}
    variants = json.load(open(V / "task_variants.json")) if (V / "task_variants.json").exists() else {}

    def base(e):
        """roll a variant ("working memory load") up to its base ("working memory"); stated parents are kept apart"""
        seen = set()
        while e in entries and (entries[e].get("parent") or {}).get("relation") == "variant" and e not in seen:
            seen.add(e); e = entries[e]["parent"]["key"]
        return e
    leaf = collections.defaultdict(lambda: collections.defaultdict(set))
    import re as _re
    sys.path.insert(0, str(ROOT / "code"))
    from offtarget import Checker
    ck = Checker(ROOT / "code" / "task_inventory_v3.json", ROOT / "code" / "queries.jsonl")
    as_task = {}
    for v in variants.values():
        for t, mod in v.items():
            if (t, mod) in as_task:
                continue
            phrase = f"{mod} " + _re.sub(r"\s*\(.*\)", "", ck.names.get(t, t)).replace(" Task", "").replace(" Color-Word", "")
            hits = [o for o, rx in ck.regexes.items() if o != t and rx.fullmatch(phrase + " task") or (o != t and rx.fullmatch(phrase))]
            if hits:
                as_task[(t, mod)] = hits[0]
    print("variants mapped to inventory tasks:", {f"{mod} {t}": o for (t, mod), o in as_task.items()})
    inv0 = {t["id"]: t["name"] for t in json.load(open(ROOT / "code" / "task_inventory_v3.json"))["tasks"]}

    class Names(dict):
        def get(self, t, d=None):
            b, _, mod = t.partition("::")
            return f"{inv0.get(b, b)} [{mod}]" if mod else inv0.get(b, d if d is not None else b)
    inv = Names()
    tp = collections.defaultdict(lambda: collections.defaultdict(set))  # task -> entry -> papers
    task_papers = collections.defaultdict(set)
    for f in sorted(glob.glob(str(ROOT / "preds" / "corpus_claims*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            if not r["constructs"] or len(r["task"]) != 1 or r["task"] == ["ambiguous"]:
                continue
            t = r["task"][0]
            mod = variants.get(r["article"], {}).get(t)
            if mod:  # the paper uses a variant: an inventory task when one has that name ("Emotional Stroop Task")
                t = as_task.get((t, mod), f"{t}::{mod}")
            for e in s2e.get(r["constructs"][0][0].strip(), []):
                if e not in entries:
                    continue
                leaf[t][e].add(r["article"])
                tp[t][base(e)].add(r["article"]); task_papers[t].add(r["article"])
    y = {t: {e: len(p) for e, p in d.items()} for t, d in tp.items()}
    tot = collections.Counter()
    for d in y.values():
        tot.update(d)
    N = sum(tot.values())
    rows, passing = [], {}
    for t, d in y.items():
        n_t = sum(d.values())
        scored = []
        for e, k in d.items():
            a = A0 * tot[e] / N                       # prior count for e
            k_rest, a_rest = n_t - k, A0 - a
            delta = math.log((k + a) / (k_rest + a_rest)) - math.log((tot[e] + a) / (N - tot[e] + a_rest))
            var = 1 / (k + a) + 1 / (tot[e] + a)
            z = delta / math.sqrt(var)
            scored.append((z, e, k))
            if k >= 2:
                rows.append((t, e, k, z))
        ok = [(z, e, k) for z, e, k in sorted(scored, reverse=True) if z >= Z and k >= MINP][:TOP]
        how = "distinctive"
        if not ok:  # fallback: nothing distinctive, so the task's most-claimed constructs in >= MINP papers
            ok = sorted([(z, e, k) for z, e, k in scored if k >= MINP], key=lambda x: -x[2])[:FALLBACK_TOP]
            how = "fallback"
        passing[t] = [{"entry": e, "how": how, "label": entries[e].get("cogat_name") or entries[e]["label"], "papers": k, "z": round(z, 2),
                       "status": entries[e]["status"], "cogat": entries[e].get("cogat_name"),
                       "parent": (entries[e].get("parent") or {}).get("name")} for z, e, k in ok]
    with open(V / "task_constructs.tsv", "w") as f:
        f.write("task\ttask_name\tentry\tpapers\tz\tstatus\tcogat\n")
        for t, e, k, z in sorted(rows, key=lambda r: (r[0], -r[3])):
            f.write(f"{t}\t{inv.get(t, t)}\t{e}\t{k}\t{z:.2f}\t{entries[e]['status']}\t{entries[e].get('cogat_name') or ''}\n")
    json.dump(passing, open(V / "task_constructs_pass.json", "w"), indent=1)
    json.dump({t: {e: len(p) for e, p in d.items()} for t, d in leaf.items()}, open(V / "task_constructs_leaf.json", "w"))
    fb = sorted((t for t in y if passing[t] and passing[t][0]["how"] == "fallback"), key=lambda t: -len(task_papers[t]))
    print(f"\n{len(fb)} tasks by fallback (no distinctive construct; most-claimed in >= {MINP} papers):")
    for t in fb:
        print(f"  {inv.get(t, t)} ({len(task_papers[t])} papers): " + "; ".join(f"{c['label']} {c['papers']}" for c in passing[t]))
    none = sorted((t for t in y if not passing[t]), key=lambda t: -len(task_papers[t]))
    print(f"\n{len(none)} tasks with claims but no construct passing:")
    for t in none:
        d = y[t]; best = max(d, key=d.get) if d else None
        print(f"  {inv.get(t, t)}: {len(task_papers[t])} papers with claims; most claimed: "
              + (f"{entries[best].get('cogat_name') or entries[best]['label']} ({d[best]} papers)" if best else "-"))
    searched = {json.loads(l)["canonical"] for l in open(ROOT / "code" / "queries.jsonl") if json.loads(l).get("terms")}
    never = sorted(t for t in searched if t not in y)
    print(f"\n{len(never)} searched tasks with no claims at all: " + "; ".join(inv.get(t, t) for t in never))
    have = [t for t in passing if passing[t]]
    n = [len(passing[t]) for t in have]
    print(f"{len(y)} tasks with claims; {len(have)} have >= 1 construct passing (z >= {Z}, >= {MINP} papers, top {TOP}); "
          f"constructs per task: median {sorted(n)[len(n) // 2]}; distinct constructs used {len({c['entry'] for t in have for c in passing[t]})}; "
          f"linked to Cognitive Atlas {sum(c['status'] == 'cogat' for t in have for c in passing[t]) / max(1, sum(n)):.0%}")
    for t in ("hedtsk_stroop_color_word", "hedtsk_n_back", "hedtsk_stop_signal", "hedtsk_pavlovian_fear_conditioning",
              "hedtsk_iowa_gambling", "hedtsk_false_belief", "hedtsk_attention_network", "hedtsk_mental_rotation"):
        if t in passing:
            print(f"- {inv.get(t, t)} ({len(task_papers[t])} papers): " + "; ".join(
                f"{c['label']}{'*' if c['status'] == 'cogat' else ''} {c['papers']}" for c in passing[t]))


if __name__ == "__main__":
    main()
