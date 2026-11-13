"""Construct authors per contrast, from the contrast hub, instead of per task (add_construct_authors.py).

add_construct_authors.py gives every contrast, as authors, all the constructs its tasks have as parents, so every
contrast of a task has the same authors and the model cannot tell them apart (an author-topic document's expected
mixture is the mean of its authors'). Here a contrast's construct authors follow what the contrast compares
(DESIGN.md, "Reversed analyses"; James 2026-11-12):
- task vs a rest/control baseline, or a one-sided simple effect, in its own direction: the task's constructs (as in
  the input corpus; the contrast is the task), or its own constructs (its paper reading) when the task has none;
- a condition comparison, either direction: its own constructs from the hub (constructs.py contrast_constructs.json,
  tiers claims / codex / atlas / hierarchy_*); a construct the corpus has no node for (a Cognitive Atlas concept
  outside the taxonomy, from a reversed entry's attribution) becomes a new node with no parent, shrunk toward the
  global mixture like the shared author; none when the contrast has no constructs;
- baseline > task or a negative simple effect (a deactivation): the one DEACTIVATION author, no constructs;
- a contrast the hub could not normalize: the constructs read from its paper (tier paper), else none.
Task, modality and shared-pattern authors, the input's parents, labels and documents are unchanged, so CV folds
and held-out-task groups match the input corpus's.

usage: python contrast_construct_authors.py CORPUS_IN HUB_DIR CONSTRUCTS.json CORPUS_OUT
"""
import collections
import json
import sys

DEACTIVATION = "deactivation"
BASES = ("@rest", "@control")
SPECIFIC = ("claims", "codex", "atlas", "paper")   # paper: read from the analysis's paper (paper_attribute.py)


def kind(k, entries):
    """task_vs_baseline | simple | condition, from the hub entry's signature"""
    e = entries.get(k)
    comps = json.loads(e["sig_json"])[1] if e else []
    base = [b for _, _, b in comps]
    if k.startswith("task:") or (base and all(b in BASES for b in base)):
        return "task_vs_baseline"
    if base and all(b is None for b in base):
        return "simple"
    return "condition"


def main():
    cin, hub_dir, consp, cout = sys.argv[1:5]
    H = json.load(open(cin)); hub = json.load(open(f"{hub_dir}/hub.json"))
    entries = {**hub.get("parents", {}), **hub["contrasts"]}
    assign = json.load(open(f"{hub_dir}/corpus_assign.json")); R = json.load(open(consp))
    nodes = set(H["parents"])
    n = collections.Counter(); per = []
    for did, d in H["docs"].items():
        keep = [x for x in d["ids"] if not x.startswith("c:")]
        task_c = [x for x in d["ids"] if x.startswith("c:")]
        a = assign.get(did)
        if a is None:                                     # not normalized: its paper reading, if any
            r = R.get(did) or {}
            new = ["c:" + c.replace("-", " ") for c in r.get("constructs") or []] if r.get("tier") == "paper" else []
            why = "not normalized: " + ("paper constructs" if new else "none")
            for i in new:
                if i not in nodes:
                    H["parents"][i] = []; H["labels"][i] = i[2:]; nodes.add(i); n["new construct nodes"] += 1
        else:
            kd, fwd = kind(a["hub"], entries), a.get("sign", 1) > 0
            if kd in ("task_vs_baseline", "simple"):
                r = R.get(did) or {}
                tier = r.get("tier") or ""
                own = ["c:" + c.replace("-", " ") for c in r.get("constructs") or []] if tier in SPECIFIC or tier.startswith("hierarchy_") else []
                if not fwd:
                    new, why = [DEACTIVATION], f"{kd} reversed: deactivation"
                elif task_c:
                    new, why = task_c, f"{kd}: task constructs"
                else:                                     # the task has none in the corpus: the contrast's own (paper)
                    new, why = own, f"{kd}: own constructs (task has none)" if own else f"{kd}: none"
                for i in new:
                    if i != DEACTIVATION and i not in nodes:
                        H["parents"][i] = []; H["labels"][i] = i[2:]; nodes.add(i); n["new construct nodes"] += 1
            else:
                r = R.get(did) or {}
                tier = r.get("tier") or ""
                cs = r.get("constructs") or [] if tier in SPECIFIC or tier.startswith("hierarchy_") else []
                new = ["c:" + c.replace("-", " ") for c in cs]
                for i in new:
                    if i not in nodes:                    # a new construct node, no parent
                        H["parents"][i] = []; H["labels"][i] = i[2:]; nodes.add(i); n["new construct nodes"] += 1
                why = f"condition{'' if fwd else ' reversed'}: " + ("own constructs" if new else "none")
        n[why] += 1; per.append(sum(x.startswith("c:") for x in new))
        d["ids"] = keep + [x for x in new if x not in keep]
    H["parents"][DEACTIVATION] = []                       # like the shared author: shrunk toward the global mixture
    H["labels"][DEACTIVATION] = "deactivation"
    cnt = collections.Counter(i for d in H["docs"].values() for i in d["ids"])
    pap = collections.defaultdict(set)
    for d in H["docs"].values():
        for i in d["ids"]:
            pap[i].add(d.get("group"))
    H["n_contrasts"] = {k: cnt.get(k, 0) for k in H["parents"]}
    H["n_papers"] = {k: len(pap.get(k, ())) for k in H["parents"]}
    H["construct_authors"] = {"rule": "contrast_construct_authors.py", "counts": dict(n)}
    json.dump(H, open(cout, "w"))
    print(json.dumps(dict(n), indent=1))
    print(f"{cout}: construct authors per contrast mean {sum(per) / len(per):.2f}; contrasts with one {sum(x > 0 for x in per)} "
          f"of {len(per)}; construct authors used {sum(1 for k, v in cnt.items() if k.startswith('c:') and v)}; "
          f"deactivation author on {cnt[DEACTIVATION]}")


if __name__ == "__main__":
    main()
