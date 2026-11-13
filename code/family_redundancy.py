"""How redundant are the curated task families and the mined constructs? For every pair of corpus tasks that both
have constructs: Jaccard overlap of their construct sets (direct parents in the construct corpus), and whether the
curated hierarchy puts them in one family. Reports the AUC of construct overlap for predicting a shared family
(Mann-Whitney), mean overlap within vs between families, and, per family, how many of its tasks share a construct.

With ANCESTORS=1 a task's set is its constructs and all their ancestors in the construct hierarchy.

usage: python family_redundancy.py CONSTRUCT_CORPUS ORIGINAL_CORPUS   (from pilot-results/atm)
"""
import collections
import itertools
import json
import os
import sys

import numpy as np


def main():
    Hc = json.load(open(sys.argv[1])); Ho = json.load(open(sys.argv[2]))
    fams = {p for ps in Ho["parents"].values() for p in ps}
    tasks = [t for t in Ho["parents"] if t not in fams and t != "background"]
    fam = {t: set(Ho["parents"][t]) for t in tasks}
    cons = {t: {p for p in Hc["parents"].get(t, []) if p.startswith("c:")} for t in tasks}
    if os.environ.get("ANCESTORS") == "1":
        def up(c, seen):
            for p in Hc["parents"].get(c, []):
                if p.startswith("c:") and p not in seen:
                    seen.add(p); up(p, seen)
            return seen
        cons = {t: set().union(*[up(c, {c}) for c in cs]) if cs else set() for t, cs in cons.items()}
    tasks = [t for t in tasks if cons[t]]
    rows = []
    for a, b in itertools.combinations(tasks, 2):
        j = len(cons[a] & cons[b]) / len(cons[a] | cons[b])
        rows.append((j, bool(fam[a] & fam[b])))
    j = np.array([r[0] for r in rows]); same = np.array([r[1] for r in rows])
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(j[same], j[~same]).statistic
    auc = u / (same.sum() * (~same).sum())
    print(f"{len(tasks)} tasks with constructs, {len(rows)} pairs, {same.sum()} in one family")
    print(f"construct Jaccard: same family {j[same].mean():.3f}, different families {j[~same].mean():.3f}; "
          f"share >= 1 construct: {np.mean(j[same] > 0):.0%} vs {np.mean(j[~same] > 0):.0%}; "
          f"AUC for a shared family {auc:.3f}")
    lab = Ho["labels"]
    by = collections.defaultdict(list)
    for t in tasks:
        for f in fam[t]:
            by[f].append(t)
    print("families (tasks with constructs): share of task pairs with >= 1 common construct")
    for f, ts in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(ts) < 2:
            continue
        pr = list(itertools.combinations(ts, 2))
        s = np.mean([bool(cons[a] & cons[b]) for a, b in pr])
        common = collections.Counter(c for t in ts for c in cons[t]).most_common(3)
        print(f"  {lab.get(f, f)} ({len(ts)}): {s:.0%}; most shared: " + ", ".join(f"{Hc['labels'].get(c, c)} {n}" for c, n in common))


if __name__ == "__main__":
    main()
