"""Before/after check of a vocabulary rebuild: entries of the old vocabulary (vocab_v0.pre-synonyms.json) in
>= MIN papers whose top wording is now dropped, and the largest synonym merges (vocab/synonym_merges.json), so a
regression ("response inhibition" dropped as a task name) or a wrong merge shows up before the task tables move.

usage: python vocab_diff.py [MIN]   (from pilot-results/)
"""
import collections
import json
import sys
from pathlib import Path

V = Path.cwd() / "vocab"


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    old = json.load(open(V / "vocab_v0.pre-synonyms.json"))
    new = {r["key"]: r for r in json.load(open(V / "vocab_v0.json"))}
    s2e = json.load(open(V / "string_to_entries.json"))
    merges = json.load(open(V / "synonym_merges.json"))
    lost = [r for r in old if r["papers"] >= k and not s2e.get(r["strings"][0][0])]
    print(f"old entries in >= {k} papers whose top wording is now dropped: {len(lost)}")
    for r in lost[:40]:
        print(f"  {r['label']} ({r['papers']}p, top wording {r['strings'][0][0]!r})")
    moved = collections.Counter()
    for r in old:
        if r["papers"] >= k:
            now = s2e.get(r["strings"][0][0]) or []
            if now and r["key"] not in now:
                moved[(r["key"], ", ".join(now))] = r["papers"]
    print(f"\nold entries in >= {k} papers now in another entry: {len(moved)}")
    for (a, b), n in moved.most_common(60):
        print(f"  {a} ({n}p) -> {b}" + (f" [{merges[a] == b and 'merge' or ''}]" if a in merges else ""))
    status = collections.Counter((r["status"]) for r in new.values() if r["papers"] >= k)
    print(f"\nnew vocabulary, entries in >= {k} papers: {dict(status)}")


if __name__ == "__main__":
    main()
