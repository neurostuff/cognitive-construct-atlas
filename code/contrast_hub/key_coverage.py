"""How far ns-pond's contrast keys already cover the coordinate corpus (contrast hub, step 2).

Input: the analysis_features export (e20_contrast_points.tsv: one row per peak; columns analysis_id, base_study_id,
task_ids, task_family_ids, effect_kind, contrast_positive, contrast_negative, contrast_key_ids, x, y, z, space) and
the corpus (docs keyed by analysis id). A key is `family|target|baseline` (ns-pond records.py): each side is a set of
condition tokens joined by ';', a token being sorted atoms (`congruency:incongruent,phase:cue`) or, when no atom
matched, `~` + the folded residue or name (unnormalized). Baselines without atoms are condition kinds (`rest`,
`fixation`, `control_state`).

usage: python key_coverage.py TSV CORPUS OUT.json
"""
import collections
import json
import sys


def side_tokens(s):
    return [t for t in s.split(";") if t and t != "-"]


def classify(key):
    """'atom' when every token on both sides is atoms or a baseline kind, 'residue' when any is `~`, 'one_sided'."""
    fam, tgt, base = key.split("|", 2)
    base = base[3:] if base.startswith("vs:") else base
    toks = side_tokens(tgt) + side_tokens(base.replace("+", ";"))
    if tgt == "-" or base == "-":
        return "one_sided"
    return "residue" if any(t.startswith("~") for t in toks) else "atom"


def main():
    tsv, corpus, out = sys.argv[1:4]
    docs = json.load(open(corpus))["docs"]
    A = {}
    for line in open(tsv):
        f = line.rstrip("\n").split("\t")
        if f[0] in docs and f[0] not in A:
            A[f[0]] = {"paper": f[1], "tasks": f[2], "families": f[3], "kind": f[4], "pos": f[5], "neg": f[6],
                       "keys": [k for k in f[7].split(",") if k] if f[7] else []}
    for a in A.values():   # contrast_key_ids is comma-joined, and atom lists inside a key use commas too
        a["keys"] = _split_keys(",".join(a["keys"]))
    n = len(A); st = collections.Counter(); papers_by_key = collections.defaultdict(set)
    fam_n = collections.Counter(); fam_atom = collections.Counter(); residue_tokens = collections.Counter()
    for aid, a in A.items():
        st["kind:" + a["kind"]] += 1
        fam = a["families"].split(",")[0] or "?"
        fam_n[fam] += 1
        if not a["keys"]:
            st["no_key"] += 1; continue
        st["keyed"] += 1; st[f"n_keys:{min(len(a['keys']), 3)}"] += 1
        cls = {classify(k) for k in a["keys"]}
        best = "atom" if "atom" in cls else "residue" if "residue" in cls else "one_sided"
        st["class:" + best] += 1
        fam_atom[fam] += best == "atom"
        for k in a["keys"]:
            papers_by_key[k].add(a["paper"])
            _, tgt, base = k.split("|", 2)
            for t in side_tokens(tgt) + side_tokens(base[3:] if base.startswith("vs:") else ""):
                if t.startswith("~"):
                    residue_tokens[(k.split("|")[0], t)] += 1
    recurring = {k for k, p in papers_by_key.items() if len(p) >= 2}
    st["in_recurring_key"] = sum(any(k in recurring for k in a["keys"]) for a in A.values())
    st["in_recurring_atom_key"] = sum(any(k in recurring and classify(k) == "atom" for k in a["keys"]) for a in A.values())
    res = {"contrasts": n, "papers": len({a["paper"] for a in A.values()}), "stats": dict(st),
           "distinct_keys": len(papers_by_key), "recurring_keys": len(recurring),
           "recurring_atom_keys": sum(classify(k) == "atom" for k in recurring),
           "families": {f: {"contrasts": c, "atom_keyed": fam_atom[f]} for f, c in fam_n.most_common()},
           "top_residue_tokens": [[f, t, c] for (f, t), c in residue_tokens.most_common(300)],
           "top_keys": [[k, len(p)] for k, p in sorted(papers_by_key.items(), key=lambda kv: -len(kv[1]))[:400]]}
    json.dump(res, open(out, "w"), indent=1)
    json.dump({aid: {**a} for aid, a in A.items()}, open(out.replace(".json", "_contrasts.json"), "w"))
    print(json.dumps({k: res[k] for k in ("contrasts", "papers", "stats", "distinct_keys", "recurring_keys", "recurring_atom_keys")}, indent=1))


def _split_keys(raw):
    """split a comma-joined list of keys whose atom lists also use commas: a new key starts at a part that holds a '|'
    and whose first field is a family name (no ':' before the first '|')"""
    if not raw:
        return []
    keys = []
    for part in raw.split(","):
        head = part.split("|", 1)[0]
        if "|" in part and ":" not in head and not head.startswith("~") or not keys:
            keys.append(part)
        else:
            keys[-1] += "," + part
    return keys


if __name__ == "__main__":
    main()
