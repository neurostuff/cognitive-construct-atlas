"""Re-match scan_contrasts.py output to a hub (CPU; the tagger spans are kept, only normalization and lookup rerun).

Counts, over tagged units: matched to any established entry, matched to an entry with imaging contrasts, and the
same for units that are construct claims (contrast-level construct claims); plus the corpus papers reached (claim
articles that are coordinate-corpus papers are not linked here; see DESIGN.md).

usage: python rematch_scan.py HUB_DIR 'SCAN_GLOB' OUT.json [LEXICON.json]
"""
import collections
import glob
import json
import sys

import hub as H
import match_claims as M


def main():
    hub_dir, pattern, out = sys.argv[1:4]
    lex = H.Lexicon(sys.argv[4] if len(sys.argv) > 4 else None)
    hub = H.Hub.load(f"{hub_dir}/hub.json")
    st = collections.Counter(); entries = collections.Counter(); arts = collections.defaultdict(set)
    unmatched = collections.Counter()
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            r = json.loads(line)
            m = M.match_one({"tasks": r["tasks"], "a": r["a"], "b": r["b"]}, hub, lex)
            claim = "constructs" in r
            st["units"] += 1; st["claim_units"] += claim; st["how:" + m["how"]] += 1
            if "hub" in m:
                e = hub.entries[m["hub"]]
                est = e["established"]; img = e["papers"] > 0
                st["established"] += est; st["imaging"] += img
                st["claim_established"] += claim and est; st["claim_imaging"] += claim and img
                if img:
                    entries[m["hub"]] += 1; arts[m["hub"]].add(r["article"])
            for s in m.get("missing", []):
                unmatched[(m["family"], s)] += 1
    n, c = st["units"], st["claim_units"]
    summary = {**st, "share_established": round(st["established"] / n, 3), "share_imaging": round(st["imaging"] / n, 3),
               "claim_share_imaging": round(st["claim_imaging"] / max(1, c), 3),
               "imaging_entries_reached": len(entries)}
    json.dump({"summary": summary,
               "top_entries": [[k, v, len(arts[k])] for k, v in entries.most_common(200)],
               "unmatched_sides": [[f, s, k] for (f, s), k in unmatched.most_common(5000)]}, open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
