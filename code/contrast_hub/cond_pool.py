"""Constructs for directed entries from stance-scored condition-search items (condition_search.py -> ft_stance.py apply).

A (sentence, entry, construct) is credited when the stance model says A (the entry's own direction) or B (its
reverse, `rev:`) at p >= P; a construct counts for a directed entry when credited in MIN_ART or more articles. The
hand-labelled sample (v7/condsearch/hand_labels.json) puts credit precision at 83% for p >= 0.9 with one article;
agreement across articles is the precision lever, since one article's error rarely repeats in another. Constructs
rank by articles; a reversed entry needs one article more; left/right entries are skipped. (Ranking by specificity,
against the task's other entries or the whole corpus, was tried and judged worse on 40 analyses: DESIGN.md.)

Reports, for the sample's analyses, how many get at least one construct this way, overall and among those with no
contrast-specific construct yet (constructs.py's contrast_constructs.json), at 1 and 2 articles.

usage: (code/contrast_hub on beast) python cond_pool.py HUB_DIR ITEMS.jsonl PRED.json SAMPLE.json CONSTRUCTS_DIR OUT.json [P]
"""
import collections
import json
import sys

from constructs import directed

SPECIFIC = ("claims", "atlas", "codex", "task_contrast")
#: left/right entries collect hemisphere, ear and hand talk, not conditions
SKIP_AXES = ("side",)


def main():
    hub_dir, itemsp, predp, samplep, cdir, outp = sys.argv[1:7]
    P = float(sys.argv[7]) if len(sys.argv) > 7 else 0.9
    pred = json.load(open(predp)); assign = json.load(open(f"{hub_dir}/corpus_assign.json"))
    cur = json.load(open(f"{cdir}/contrast_constructs.json")); sample = json.load(open(samplep))
    ev = collections.defaultdict(lambda: collections.defaultdict(set))       # directed key -> construct id -> articles
    n = collections.Counter()
    for line in open(itemsp):
        r = json.loads(line)
        for c, ids in r["construct_ids"].items():
            p = pred.get(r["id"], {}).get(c)
            if p is None:
                continue
            n["scored"] += 1
            if p["label"] not in ("A", "B") or p["p"] < P:
                continue
            n[f"credited_{p['label']}"] += 1
            if any(c.split(":")[0] in SKIP_AXES for c in r["hub"].split("|", 1)[1].split(";")):
                n["skipped_side"] += 1; continue
            dk = directed(r["hub"], 1 if p["label"] == "A" else -1)
            for cid in ids:
                ev[dk][cid].add(r["article"])
    out = {}
    for k in (1, 2):
        ent = {dk: sorted((c for c, a in d.items() if len(a) >= k + dk.startswith("rev:")), key=lambda c, d=d: -len(d[c]))
               for dk, d in ev.items()}
        ent = {dk: cs for dk, cs in ent.items() if cs}
        sk = [a for a in sample if a in assign]
        dk_of = {a: directed(assign[a]["hub"], assign[a].get("sign", 1)) for a in sk}
        spec = lambda a: (cur.get(a, {}).get("tier") or "") in SPECIFIC or (cur.get(a, {}).get("tier") or "").startswith("hierarchy_")  # noqa: E731
        got = [a for a in sk if dk_of[a] in ent]
        out[f"min_articles_{k}"] = {
            "directed_entries": len(ent),
            "sample_with_construct": len(got),
            "sample_with_construct_share": round(len(got) / len(sample), 3),
            "of_those_without_contrast_specific_now": sum(not spec(a) for a in got),
            "sample_specific_now_or_here": sum(spec(a) or a in set(got) for a in sk),
        }
        out[f"entries_{k}"] = {dk: cs[:5] for dk, cs in ent.items()}
    R = {"p": P, "counts": dict(n), **{k: v for k, v in out.items() if k.startswith("min_")}}
    print(json.dumps(R, indent=1))
    json.dump({**R, **{k: v for k, v in out.items() if k.startswith("entries_")}}, open(outp, "w"))


if __name__ == "__main__":
    main()
