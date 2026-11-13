"""Fill `analyses` for own_contrast claims a model left unlinked, from its own `contrast_text`.

Candidates are the paper's analyses of the claim's task (all analyses when the task has none). Each is scored by
the content words its name and contrast-set label share with contrast_text (methods.words: stop words such as
task/condition/contrast/vs removed); the best-scoring analyses (score >= 1) are attached, ties all kept.

usage: python link_contrasts.py TAG NEWTAG   (reads llm/<a>.TAG.jsonl, writes llm/<a>.NEWTAG.jsonl)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from methods import words  # noqa: E402


def main():
    tag, new = sys.argv[1:3]
    filled = total = 0
    for f in sorted(x for x in (ROOT / "llm").glob(f"*.{tag}.jsonl") if not x.name.startswith("usage.")):
        art = f.name.split(".")[0]
        p = json.loads((ROOT / "pilot/papers" / f"{art}.json").read_text())
        out = []
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("category") == "own_contrast" and not r.get("analyses") and r.get("contrast_text"):
                total += 1
                cands = [a for a in p["analyses"] if r.get("task") in a["tasks"]] or p["analyses"]
                cw = words(r["contrast_text"])
                scored = [(len(cw & (words(a["name"]) | words(a.get("contrast_set_label")))), a["analysis"]) for a in cands]
                best = max((s for s, _ in scored), default=0)
                if best >= 1:
                    r["analyses"] = [a for s, a in scored if s == best]; r["linked_by"] = "contrast_text"; filled += 1
            out.append(json.dumps(r))
        (ROOT / "llm" / f"{art}.{new}.jsonl").write_text("\n".join(out) + "\n")
    print(f"{tag} -> {new}: filled {filled} of {total} unlinked contrast claims")


if __name__ == "__main__":
    main()
