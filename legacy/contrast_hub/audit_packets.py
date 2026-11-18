"""Evidence packets for the audit's readers (DESIGN.md, Audit workflow). From audit_candidates.py's placements, the
top of three rankings (task label suspect by task margin; every construct flagged with peaks unlike them; one
construct flagged with peaks unlike it), with every other flagged analysis of the same papers. Per paper:
paper_attribute.py's prep context (title and abstract, N_DESIGN trial-structure sentences, per analysis its sides,
signature, table caption and the sentences naming its conditions), plus per analysis its Neurostore name, its task,
its constructs and which are flagged, and, for a normalized contrast, the constructs the other papers with its
signature carry. Written in batches of BATCH papers for one reader each.

EXCLUDE=papers.json leaves out papers read in earlier rounds.

usage: (beast, pilot-results/atm; venv-cc) [EXCLUDE=...] python ../code/contrast_hub/audit_packets.py audit/candidates.json OUT_DIR N_TASK N_ALL N_ONE
"""
import collections
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BATCH = 13
HERE = Path(__file__).resolve().parent


def main():
    candp, out, n_task, n_all, n_one = sys.argv[1], Path(sys.argv[2]), *map(int, sys.argv[3:6])
    out.mkdir(parents=True, exist_ok=True)
    P = json.load(open(candp))["placements"]; mi = lambda p: p["min_imaging"] if p["min_imaging"] is not None else 0  # noqa: E731
    a = sorted([p for p in P if p["all_flagged"] and (p.get("task_margin") or 0) >= 1], key=lambda p: -p["task_margin"])[:n_task]
    b = sorted([p for p in P if p["all_flagged"] and mi(p) < -2], key=mi)[:n_all]
    c = sorted([p for p in P if not p["all_flagged"] and mi(p) < -3], key=mi)[:n_one]
    why = {}
    for lst, w in ((a, "task label suspect"), (b, "all constructs unlike its peaks"), (c, "one construct unlike its peaks")):
        for p in lst:
            why.setdefault(p["analysis"], w)
    done = set(json.load(open(os.environ["EXCLUDE"]))) if os.environ.get("EXCLUDE") else set()   # papers read in earlier rounds
    papers = {p["paper"] for p in a + b + c} - done
    sel = {p["analysis"]: p for p in P if p["paper"] in papers}
    json.dump(sorted(sel), open(out / "sample.json", "w"))
    env = dict(os.environ, ALL="1")
    code = ("import sys; sys.argv=['paper_attribute.py','prep',%r,%r,'../contrast_hub/contrasts_all.json','corpus_w_B.json','../contrast_hub/v7',%r];"
            "sys.path.insert(0,%r); import paper_attribute as pa; pa.N_DESIGN=20; pa.prep()") % (
        str(out / "sample.json"), "../contrast_hub/v7/constructs_dir/contrast_constructs_phase.json", str(out / "papers.jsonl"), str(HERE))
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
    H = json.load(open("corpus_w_B.json")); A = json.load(open("../contrast_hub/v7/corpus_assign.json")); lab = H["labels"]
    sys.path.insert(0, str(HERE)); import hub
    inv = {t["id"]: t["name"] for t in hub.INVENTORY["tasks"]}; tname = lambda t: inv.get(t) or lab.get(t, t)  # noqa: E731
    sig_c = collections.defaultdict(collections.Counter); sig_p = collections.defaultdict(set)
    for k, d in H["docs"].items():
        x = A.get(k)
        if x:
            key = (x["hub"], x.get("sign", 1)); sig_p[key].add(d["group"])
            for i in d["ids"]:
                if i.startswith("c:"):
                    sig_c[key, d["group"]][i[2:]] += 1

    def name(aid):
        for _ in range(3):
            try:
                d = json.load(urllib.request.urlopen(f"https://neurostore.org/api/analyses/{aid}?nested=false", timeout=30))
                return aid, (d.get("name") or "") + ((" | " + d["description"]) if d.get("description") else "")
            except Exception:  # noqa: BLE001
                time.sleep(2)
        return aid, ""
    with ThreadPoolExecutor(4) as ex:
        names = dict(ex.map(name, sorted(sel)))
    packets = []
    for line in open(out / "papers.jsonl"):
        p = json.loads(line)
        for it in p["analyses"]:
            c_ = sel[it["aid"]]; x = A.get(it["aid"]); others = collections.Counter()
            if x:
                key = (x["hub"], x.get("sign", 1))
                for g in sig_p[key] - {p["paper"]}:
                    others.update(sig_c[key, g].keys())
            it.update({"name": names.get(it["aid"], ""), "task": tname(c_["task"]), "task_id": c_["task"], "constructs": c_["constructs"],
                       "flagged_constructs": [f["construct"] for f in c_["flagged"]], "tag_source": c_["tier"],
                       "why_selected": why.get(it["aid"], "same paper as a selected analysis"),
                       "peaks_resemble_task": tname(c_["best_task"]) if (c_.get("task_margin") or 0) >= 1 else None,
                       "other_papers_with_signature": c_["other_papers_with_signature"],
                       "their_constructs": [[k, n] for k, n in others.most_common(6)]})
        packets.append(p)
    packets.sort(key=lambda p: p["paper"])
    for i in range(0, len(packets), BATCH):
        json.dump(packets[i:i + BATCH], open(out / f"batch_{i // BATCH + 1:02d}.json", "w"), indent=1)
    print(json.dumps({"papers": len(packets), "analyses": sum(len(p["analyses"]) for p in packets), "batches": -(-len(packets) // BATCH),
                      "selected for": dict(collections.Counter(why.values()))}))


if __name__ == "__main__":
    main()
