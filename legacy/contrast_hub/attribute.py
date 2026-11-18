"""Codex construct attribution for hub entries, run where the codex CLI is logged in (the laptop): constructs.py's
codex_attribute with local copies of the taxonomy, the hub and the contrasts. The answers file is then copied next to
constructs.py's output on beast and read by `constructs.py` without --codex.

usage: [ONLY=KEYS.json] [SAMPLE=CONTRASTS.json] python attribute.py HUB_DIR CONTRASTS.json TAXONOMY.json OUT.json [MODEL EFFORT]
"""
import collections
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # codex_extract.py lives in cc/
import constructs as K  # noqa: E402

K.ROOT = Path(__file__).resolve().parent.parent.parent / "_unused"   # constructs imports codex_extract from ROOT/code otherwise


def main():
    hub_dir, contrasts, taxp, out = sys.argv[1:5]
    model, effort = (sys.argv[5], sys.argv[6]) if len(sys.argv) > 6 else ("gpt-6.1-sol", "low")
    hubj = json.load(open(f"{hub_dir}/hub.json")); assign = json.load(open(f"{hub_dir}/corpus_assign.json"))
    C = json.load(open(contrasts)); tax = json.load(open(taxp))
    sample = set(json.load(open(os.environ["SAMPLE"]))) if os.environ.get("SAMPLE") else None   # entries the sample uses
    used = collections.Counter(K.directed(a["hub"], a.get("sign", 1)) for aid, a in assign.items()
                               if sample is None or aid in sample)            # per direction: `rev:<entry>` = its reverse
    need = [k for k, _ in used.most_common() if K.undirected(k)[0] in hubj["contrasts"] and not k.startswith("task:")]
    if os.environ.get("ONLY"):                          # a JSON list of entry keys (e.g. an audit sample)
        only = set(json.load(open(os.environ["ONLY"]))); need = [k for k in need if k in only]
    K.codex_attribute(need, hubj, tax, out, C, assign, model, effort)


if __name__ == "__main__":
    main()
