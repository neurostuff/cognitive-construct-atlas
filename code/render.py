"""Render a pilot bundle as plain text for labelling: header with the paper's tasks and analyses, then numbered
units with section tags. Back matter (references, acknowledgements) is left out.

usage: python render.py pilot/papers/<article>.json > rendered/<article>.txt
"""
import json
import sys

p = json.load(open(sys.argv[1]))
out = [f"# Paper {p['article']}: {p.get('title')} ({p.get('journal')}, {p.get('year')})", "", "## Tasks"]
for t in p["tasks"]:
    out.append(f"- {t['task']}: {t['name']!r} (canonical {t['canonical']}, family {t['family']})")
    if t.get("description"):
        out.append(f"    description: {t['description']}")
    for c in t["conditions"]:
        out.append(f"    condition {c['condition']}: {c['name']!r}")
out += ["", "## Analyses (contrasts)"]
for a in p["analyses"]:
    lab = f" | contrast set: {a['contrast_set_label']!r}" if a.get("contrast_set") else ""
    out.append(f"- {a['analysis']} [task {','.join(a['tasks']) or '-'}]: {a['name']!r}{lab}")
out += ["", "## Units"]
for u in p["units"]:
    if u["section"] == "back":
        continue
    out.append(f"[{u['i']}] ({u['section']}) {u['text']}")
print("\n".join(out))
