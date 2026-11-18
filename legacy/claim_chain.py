"""Construct claims for a paper's searched task(s), no LLM: claim classifier ensemble -> off-target check -> tagger.

For each paper (a text in the construct-claims catalog, with the tasks whose searches found it):
1. units (build_pilot.units) and their sections (ns-pond's sectionizer);
2. the units to score: every non-back unit of 25+ characters, or with --stage1 only those stage1.keep passes (55% of
   units, 89% of gold claims); scored by the three classifier members, q = sigmoid(a * mean z + b)
   (construct-claim-clf/v1/calibration.json);
3. kept: q >= Q (default 0.9) and the off-target check does not say the unit names another task;
4. the task the claim is about: the target tasks the unit names; else the last target named within the 5 units
   before it; else the paper's only target; else "ambiguous" (several targets, none named nearby);
5. the tagger's spans on each kept unit, most confident first, are its constructs.

Two steps, because the ns-pond sectionizer and the GPU models need different packages (both are in venv-cc):
  prep   writes PAPERS_DIR/<id>.json {article, targets, target_names, units}
  score  writes one JSON line per kept unit: article, targets, task, i, section, text, q, names, constructs

usage: python claim_chain.py prep PAPERS_DIR sample N SEED [WORKERS]   N papers, as in build_silver.py, excluding the
                                                                     gold and silver papers (PMC texts)
       python claim_chain.py prep PAPERS_DIR ids ID1,ID2,... [WORKERS]
       python claim_chain.py prep PAPERS_DIR corpus - [WORKERS]       every extracted text found by a named search,
                                                                     skipping papers already in PAPERS_DIR
       python claim_chain.py score PAPERS_DIR OUT.jsonl [--q 0.9] [--stage1] [--shard K/N] [--cascade -2]
                                   [--newer-than FILE]
(from pilot-results/; PYTHONPATH = code/; run/env.sh for prep)
"""
import argparse
import json
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

from offtarget import Checker, mentions

ROOT = Path.cwd()
CLF = Path("/data2/models/construct-claim-clf/v1")
TAG = Path("/data2/models/construct-claim-tagger/v1/weights")
LOOKBACK = 5


def checker():
    return Checker(ROOT / "code" / "task_inventory_v3.json", ROOT / "code" / "queries.jsonl")


def papers(mode, arg, seed, ck, out_dir):
    import build_silver as bs
    con = bs.sqlite3.connect(f"file:{bs.CC}/catalog/catalog.sqlite?mode=ro", uri=True)
    text_of, src = {}, {}
    for art, b in con.execute("select article_id, blob from artifacts where stage='extract' and status='ok'"):
        d = bs.blob(b)
        if d.get("full_text_path"):
            text_of[art] = d["full_text_path"]; src[art] = d.get("source")
    found = {}
    for l in open(bs.CC / "data" / "article_tasks.jsonl"):
        r = json.loads(l); found[r["article"]] = sorted({t for t, _ in r["found"]} & set(ck.names))
    if mode == "ids":
        return [(a, set(found[a]), text_of[a]) for a in arg.split(",")]
    if mode == "corpus":
        done = {f.stem for f in Path(out_dir).glob("*.json")}
        return [(a, set(ts), text_of[a]) for a, ts in found.items() if ts and a in text_of and a not in done]
    skip = bs.gold_cc_ids() | {f.stem for f in (ROOT / "silver" / "papers").glob("*.json")}
    pool = [a for a, ts in found.items() if len(ts) == 1 and src.get(a) == "pmc" and a not in skip]
    rng = random.Random(seed); rng.shuffle(pool)
    out, used = [], set()
    for a in pool:
        t = found[a][0]
        if t in used:
            continue
        n = len(Path(text_of[a]).read_text(encoding="utf-8"))
        if 15000 <= n <= 120000:
            out.append((a, {t}, text_of[a])); used.add(t)
        if len(out) == int(arg):
            break
    return out


def _prep_one(job):
    import build_silver as bs
    from build_pilot import units as split_units
    from ingestion_workflow.services.coordinate_space import sectionize
    a, targets, path, out, names = job
    try:
        text = Path(path).read_text(encoding="utf-8")
        spans = sectionize(text)
        U = []
        for k, (x, y) in enumerate(split_units(text)):
            lab = next((lab for s, e, lab in spans if s <= x < e), "unknown")
            U.append({"i": k, "text": text[x:y], "section": bs.SECTION.get(lab, lab)})
    except Exception as e:  # one bad text must not stop a corpus run
        return f"{a}: {type(e).__name__}: {e}"
    (Path(out) / f"{a}.json").write_text(json.dumps({"article": a, "targets": sorted(targets),
                                                     "target_names": [names.get(t, t) for t in sorted(targets)],
                                                     "units": U}))
    return None


def prep(out_dir, mode, arg, seed, workers):
    ck = checker()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    todo = papers(mode, arg, seed, ck, out_dir)
    print(f"{len(todo)} papers to prepare", flush=True)
    jobs = [(a, t, p, out_dir, ck.names) for a, t, p in todo]
    errors = 0
    with Pool(workers) as pool:
        for k, err in enumerate(pool.imap_unordered(_prep_one, jobs, chunksize=20), 1):
            if err:
                errors += 1; print(err, flush=True)
            if k % 2000 == 0:
                print(f"prepared {k}/{len(jobs)} ({errors} errors)", flush=True)
    print(f"prep done: {len(jobs) - errors} written, {errors} errors", flush=True)


def claim_task(U, k, targets, ments):
    named = ments[k] & targets
    for j in range(k - 1, max(-1, k - 1 - LOOKBACK), -1):
        if named:
            break
        named = ments[j] & targets
    if named:
        return sorted(named)
    return sorted(targets) if len(targets) == 1 else ["ambiguous"]


def margins(members, tok, rows, bs=64, skip_below=None):
    """per row, each member's logit margin; each batch is tokenized once for all members.
    skip_below: rows whose first member's margin is below it get None for the other members (they cannot reach
    the cutoff: see --cascade)."""
    import torch
    out = []
    with torch.no_grad():
        for k in range(0, len(rows), bs):
            sel = rows[k:k + bs]
            enc = tok([r["a"] for r in sel], [r["b"] for r in sel], truncation="longest_first", max_length=320,
                      padding=True, return_tensors="pt")
            enc = {x: v.cuda(non_blocking=True) for x, v in enc.items()}
            def run(m, e):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = m(**e).logits.float()
                return lg[:, 1] - lg[:, 0]
            z0 = run(members[0], enc)
            go = torch.ones_like(z0, dtype=torch.bool) if skip_below is None else z0 >= skip_below
            rest = torch.full((len(sel), len(members) - 1), float("nan"), device=z0.device)
            if go.any():
                sub = {x: v[go] for x, v in enc.items()}
                rest[go] = torch.stack([run(m, sub) for m in members[1:]], 1)
            for z, r in zip(z0.cpu().tolist(), rest.cpu().tolist()):
                out.append([z] + [None if x != x else x for x in r])
    return out


def score(papers_dir, out_path, qmin, use_stage1, shard, cascade=None, newer_than=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer
    import clf_cv2
    import span_cv2
    from stage1 import keep
    ck = checker()
    cal = json.load(open(CLF / "calibration.json"))
    tok = AutoTokenizer.from_pretrained(CLF / "weights" / "m0")
    members = [AutoModelForSequenceClassification.from_pretrained(CLF / "weights" / m).cuda().eval()
               for m in ("m0", "m1", "m2")]
    ttok = AutoTokenizer.from_pretrained(TAG)
    tagger = AutoModelForTokenClassification.from_pretrained(TAG).cuda().eval()
    k_shard, n_shard = shard
    files = sorted(Path(papers_dir).glob("*.json"))
    if newer_than:  # only papers prepared after an earlier run's output (incremental runs)
        t0 = Path(newer_than).stat().st_mtime
        files = [f for f in files if f.stat().st_mtime > t0]
    files = [f for j, f in enumerate(files) if j % n_shard == k_shard]
    n_units = n_kept = 0
    with open(out_path, "w") as f:
        for n_paper, pf in enumerate(files, 1):
            p = json.loads(pf.read_text()); a, U, targets = p["article"], p["units"], set(p["targets"])
            rxs, ments = ck.unit_mentions([u["text"] for u in U])  # once per paper: off-target, stage 1, task use them
            only = set(keep(U, targets, rxs, ments)) if use_stage1 else None
            rows = clf_cv2.rows_for(a, U, targets, ck, lambda i: 0, only=only, rxs=rxs, ments=ments)
            srt = sorted(rows, key=lambda r: len(r["b"]) + len(r["a"]))  # similar lengths per batch: less padding
            zs = margins(members, tok, srt, skip_below=cascade) if srt else []  # per row: the members' margins
            kept = []
            for r, z in zip(srt, zs):
                if None in z:  # skipped by the cascade: cannot reach the cutoff
                    continue
                q = 1 / (1 + math.exp(-(cal["a"] * sum(z) / len(z) + cal["b"])))
                if q >= qmin and r["names"] != "other":
                    kept.append((r, q))
            pos = {u["i"]: k for k, u in enumerate(U)}
            for k in range(0, len(kept), 32):
                batch = kept[k:k + 32]
                enc = span_cv2.encode(ttok, [{"text": r["b"]} for r, _ in batch], False); offs = enc.pop("offset_mapping")
                with torch.no_grad():
                    pr = tagger(**{x: v.cuda() for x, v in enc.items()}).logits.softmax(-1).cpu()
                for j, (r, q) in enumerate(batch):
                    cons = sorted(span_cv2.spans_of(pr[j], offs[j].tolist(), r["b"]), key=lambda x: -x[1])
                    f.write(json.dumps({"article": a, "targets": p["targets"],
                                        "task": claim_task(U, pos[r["i"]], targets, ments), "i": r["i"],
                                        "section": U[pos[r["i"]]]["section"], "text": r["b"], "q": round(q, 4),
                                        "names": r["names"], "constructs": cons}) + "\n")
            n_units += len(rows); n_kept += len(kept)
            if n_paper % 200 == 0:
                f.flush()
                print(f"shard {k_shard}: {n_paper}/{len(files)} papers, {n_units} units scored, {n_kept} kept",
                      flush=True)
    print(f"shard {k_shard} done: {len(files)} papers, {n_units} units, {n_kept} kept", flush=True)


def main():
    if sys.argv[1] == "prep":
        _, _, out_dir, mode, arg, *rest = sys.argv
        seed = int(rest[0]) if mode == "sample" else 0
        workers = int(rest[1 if mode == "sample" else 0]) if len(rest) > (1 if mode == "sample" else 0) else 4
        prep(out_dir, mode, arg, seed, workers)
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd"); ap.add_argument("papers_dir"); ap.add_argument("out")
        ap.add_argument("--q", type=float, default=0.9); ap.add_argument("--stage1", action="store_true")
        ap.add_argument("--shard", default="0/1")
        ap.add_argument("--cascade", type=float, default=None,
                        help="run members 2-3 only where member 1's margin >= this. Exact for q >= 0.9 when it is "
                             "below 3 * 5.21 - (the other members' largest margins); on the gold those are 7.62 and "
                             "7.36, so 0.64; -2 leaves room and still skips 94%% of stage-1 units")
        ap.add_argument("--newer-than", default=None, help="score only papers whose prepared file is newer than this file")
        o = ap.parse_args()
        score(o.papers_dir, o.out, o.q, o.stage1, tuple(int(x) for x in o.shard.split("/")), o.cascade, o.newer_than)


if __name__ == "__main__":
    main()
