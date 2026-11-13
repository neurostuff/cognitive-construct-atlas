"""Compare a title/abstract animal-term NOT filter with PubMed's MeSH human filter, per task query.
MeSH reference: human = humans[mh] OR NOT animals[mh] (no animal indexing); animal-only = animals[mh] NOT humans[mh].
usage: PUBMED_API_KEY=... python -I human_filter_check.py queries.jsonl canon1,canon2,...
"""
import json, os, sys, time, urllib.parse, urllib.request
KEY = os.environ.get("PUBMED_API_KEY")
ANIMALS = ["rat", "rats", "mice", "mouse", "rodent", "rodents", "murine", "monkey", "monkeys", "macaque", "macaques",
           "primate", "primates", "zebrafish", "drosophila", "pigeons", "birds", "songbird", "dogs", "cats", "sheep",
           "pigs", "animals"]
def ids(term):
    out, start = set(), 0
    while True:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(
            {"db": "pubmed", "term": term, "retmax": 10000, "retstart": start, "retmode": "json", "api_key": KEY})
        with urllib.request.urlopen(url, timeout=120) as r:
            d = json.loads(r.read().decode("utf-8", "replace"), strict=False)["esearchresult"]
        out |= set(d["idlist"]); start += 10000; time.sleep(0.12)
        if start >= int(d["count"]) or start >= 30000:
            return out
MODE = sys.argv[3] if len(sys.argv) > 3 else "not"
rows = {r["canonical"]: r for r in map(json.loads, open(sys.argv[1]))}
tail = ' AND "journal article"[pt] AND 1960:2026[dp]'
notf = " NOT (" + " OR ".join(f'"{a}"[tiab]' for a in ANIMALS) + ")"
HUMANS = ["participants", "subjects", "humans", "human", "patients", "volunteers", "adults", "children", "adolescents",
          "healthy", "individuals", "people", "men", "women", "students", "infants", "older adults"]
hum = " AND (" + " OR ".join(f'"{h}"[tiab]' for h in HUMANS) + ")"
for c in sys.argv[2].split(","):
    base = "(" + " OR ".join(f'"{t}"[tiab]' for t in rows[c]["terms"]) + ")"
    A = ids(base + tail)
    F = ids("((" + base + hum + ") OR (" + base + notf + "))" + tail) if MODE == "soft" else ids(base + notf + tail)
    animal_only = ids(base + " AND (animals[mh] NOT humans[mh])" + tail)
    human = A - animal_only
    kept_animal = len(F & animal_only); lost_human = len(human - F)
    print(f"{c:36s} all {len(A):6d} | MeSH animal-only {len(animal_only):5d} | filter keeps {len(F):6d}: "
          f"drops {len(animal_only - F)}/{len(animal_only)} animal-only ({(len(animal_only-F)/max(1,len(animal_only))):.0%}), "
          f"loses {lost_human}/{len(human)} non-animal ({lost_human/max(1,len(human)):.1%}), keeps {kept_animal} animal-only")
