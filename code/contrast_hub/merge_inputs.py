"""Merge the sample's hub inputs with the whole ns-pond corpus's (ns_features.py, abbrev.py over its papers, ft_infer
readings of its contrasts) into *_all.json, the files the whole-corpus build reads. A sample entry always wins.

usage: python merge_inputs.py
"""
import json
from pathlib import Path


def merged(out, *paths):
    d = {}
    for p in reversed(paths):                 # the first file wins
        if Path(p).exists():
            d.update(json.load(open(p)))
    json.dump(d, open(out, "w"))
    print(out, len(d))


merged("contrasts_all.json", "key_coverage_contrasts.json", "ns_all_contrasts.json")
merged("glossary_all.json", "glossary.json", "ns_all_glossary.json")
merged("abbrev_all.json", "abbrev.json", "ns_all_abbrev.json")
merged("overrides_all.json", "overrides.json", "ns_all_ft_readings.json")
