"""Task variants a paper uses, from its own text: "emotional Stroop", "numerical Stroop", "affective go/no-go".

For each paper and each of its searched tasks, every mention of a task search term (offtarget.Checker patterns) is
checked for a modifier word right before it (MODIFIERS). A paper uses variant "<modifier>" of the task when >= 3
mentions carry that modifier and they are >= 50% of the task's mentions in the paper; otherwise the plain task.
Writes vocab/task_variants.json: {article: {task: modifier}} for papers with a variant, and prints the variants
used by >= 10 papers.

usage: python task_variants.py WORKERS   (from pilot-results/)
"""
import collections
import glob
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "code"))
from offtarget import Checker  # noqa: E402

MODIFIERS = ("emotional", "affective", "emotion", "numerical", "numeric", "number", "spatial", "counting", "auditory",
             "food", "alcohol", "drug", "smoking", "cocaine", "cannabis", "nicotine", "semantic", "phonological",
             "reverse", "reversed", "threat")
# Words that do not change the paradigm ("modified", "visual", "verbal", "letter", "computerized") are not variants.
CK = None


def init():
    global CK
    CK = Checker(ROOT / "code" / "task_inventory_v3.json", ROOT / "code" / "queries.jsonl")


def one(path):
    p = json.loads(Path(path).read_text())
    text = " ".join(u["text"] for u in p["units"] if u["section"] != "back")
    out = {}
    for t in p["targets"]:
        rx = CK.regexes.get(t)
        if rx is None:
            continue
        mods = collections.Counter(); n = 0
        for m in rx.finditer(text):
            n += 1
            before = re.findall(r"[A-Za-z-]+", text[max(0, m.start() - 30):m.start()])
            if before and before[-1].lower() in MODIFIERS:
                mods[before[-1].lower()] += 1
        if mods:
            mod, k = mods.most_common(1)[0]
            if k >= 3 and k >= 0.5 * n:
                out[t] = {"emotion": "emotional", "affective": "emotional", "numeric": "numerical",
                          "number": "numerical", "reversed": "reverse", "nicotine": "smoking"}.get(mod, mod)
    return p["article"], out


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    files = sorted(glob.glob(str(ROOT / "corpus_papers" / "*.json")))
    res = {}
    with Pool(workers, initializer=init) as pool:
        for a, v in pool.imap_unordered(one, files, chunksize=50):
            if v:
                res[a] = v
    json.dump(res, open(ROOT / "vocab" / "task_variants.json", "w"))
    c = collections.Counter((t, m) for v in res.values() for t, m in v.items())
    inv = {t["id"]: t["name"] for t in json.load(open(ROOT / "code" / "task_inventory_v3.json"))["tasks"]}
    print(f"{len(files)} papers; {len(res)} use a variant of a searched task")
    for (t, m), n in c.most_common():
        if n >= 10:
            print(f"  {n:5d}  {m} {inv.get(t, t)}")


if __name__ == "__main__":
    main()
