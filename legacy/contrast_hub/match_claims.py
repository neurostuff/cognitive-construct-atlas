"""Claim-side contrasts -> hub contrasts (contrast hub, claim route).

A claim contrast is two condition spans (A, B) from a construct-claim sentence plus the claim's canonical task(s):
Codex labels now (contrast_labels/sol_low_*.jsonl), the contrast tagger's spans at scale. Each side is normalized with
the same atoms as the imaging side (hub.side_from_text, in the task's family), the pair becomes a signature, and the
signature is looked up in the hub:
- exact: the same signature (or its flip, sign -1);
- relaxed: an established entry of the family whose components contain the claim's or are contained in them (a
  claim that says "incongruent versus congruent" of a Stroop with words; one that names an extra qualifier).
Signatures are scoped to the claim's task (an entry is a task's comparison); when the task has no such entry, the
family-level parent (the same comparison in sibling tasks) is tried, reported as how='family'. A claim with several
tasks is tried in each; the first that matches wins.

usage: python match_claims.py HUB_DIR LABELS_GLOB OUT.json [LEXICON.json]
"""
import collections
import glob
import json
import sys

import hub as H


def claim_contrasts(pattern):
    rows = {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            r = json.loads(line); rows[r["id"]] = r
    for r in rows.values():
        L = r["label"]
        if L["has_contrast"] and L["condition_a"]:
            yield {"id": r["id"], "article": r["article"], "tasks": r["task"], "a": L["condition_a"], "b": L["condition_b"],
                   "kind": L["kind"], "measure": L["measure"], "construct_link": L["construct_link"],
                   "constructs": r["constructs"], "text": r["text"]}


def match_one(c, hub, lex):
    tried = []
    for tid in c["tasks"] or [None]:
        fam = H.TASK_FAMILY.get(tid)
        if fam is None:
            continue
        A = H.side_from_text(c["a"], tid, fam, lex)
        B = H.side_from_text(c["b"], tid, fam, lex) if c["b"] else None
        if B is not None and B["kind"] == "not_a_condition":
            B = None
        T, B2, osign = H.orient(A, B)
        sig = H.scoped(H.signature(fam, T, B2), tid)      # a claim's contrast is read within its task
        tried.append((fam, A, B, sig))
        if sig is None:
            continue
        for how, f in (("exact_established", lambda s: hub.lookup(s, True)), ("exact", lambda s: hub.lookup(s)),
                       ("relaxed", lambda s: hub.relaxed(s))):
            hit = f(sig)
            if hit:
                return {"how": how, "hub": hit[0], "sign": hit[1] * osign, "family": fam, "sig": H.sig_str(sig)}
        fsig = (H.FAMILY_PREFIX + fam, sig[1])            # the same comparison in a sibling task of the family
        for s2, sg in ((fsig, 1), (H.flip(fsig), -1)):
            if s2 is not None and H.sig_str(s2) in hub.parents:
                return {"how": "family", "hub": H.sig_str(s2), "sign": sg * osign, "family": fam, "sig": H.sig_str(sig)}
    if not tried:
        return {"how": "no_family"}
    fam, A, B, sig = tried[0]
    if sig is None:
        miss = [s for s in (A, B) if s is not None and not s["atoms"] and s["kind"] not in H.BASE_KINDS]
        return {"how": "unnormalized", "family": fam, "missing": [m["residue"] for m in miss]}
    return {"how": "no_entry", "family": fam, "sig": H.sig_str(sig), "sig_json": H.sig_json(sig)}


def main():
    hub_dir, pattern, out = sys.argv[1:4]
    lex = H.Lexicon(sys.argv[4] if len(sys.argv) > 4 else None)
    hub = H.Hub.load(f"{hub_dir}/hub.json")
    res = []; st = collections.Counter(); by_measure = collections.defaultdict(collections.Counter)
    unmatched = collections.Counter(); no_entry = collections.Counter()
    for c in claim_contrasts(pattern):
        m = match_one(c, hub, lex); res.append({**c, "match": m})
        st[m["how"]] += 1; by_measure[c["measure"]][m["how"]] += 1
        for s in m.get("missing", []):
            unmatched[(m["family"], s)] += 1
        if m["how"] == "no_entry":
            no_entry[m["sig"]] += 1
    n = len(res); hit = sum(st[k] for k in ("exact_established", "exact", "relaxed"))
    imaging = sum("hub" in r["match"] and (hub.entries.get(r["match"]["hub"]) or hub.parents.get(r["match"]["hub"], {})).get("papers", 0) > 0 for r in res)
    summary = {"claim_contrasts": n, "matched": hit, "share_matched": round(hit / max(1, n), 3),
               "matched_imaging": imaging, "share_matched_imaging": round(imaging / max(1, n), 3), "how": dict(st),
               "by_measure": {k: dict(v) for k, v in by_measure.items()},
               "construct_linked_matched": sum(r["match"]["how"] in ("exact_established", "exact", "relaxed") and r["construct_link"] for r in res),
               "construct_linked": sum(bool(r["construct_link"]) for r in res)}
    json.dump({"summary": summary, "claims": res,
               "unmatched_sides": [[f, s, k] for (f, s), k in unmatched.most_common()],
               "no_entry": no_entry.most_common()}, open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
