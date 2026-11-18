"""Add `pmc_efetch` to an ns-pond checkout: false keeps the pmc source to PMC's Cloud Service (the public S3 bucket),
without NCBI efetch for what the bucket lacks (construct-claims first pass; James 2026-11-10). Default true: no change.

usage: python -I patch_pmc_efetch.py NS_POND_CHECKOUT
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])

cfg = root / "ingestion_workflow/config/__init__.py"
s = cfg.read_text()
old = "    pdf_engines: List[Literal["
assert old in s and "pmc_efetch" not in s, "config anchor"
s = s.replace(old, '''    pmc_efetch: bool = Field(
        default=True,
        description=(
            "Let the pmc source ask NCBI efetch for the articles PMC's Cloud Service (the public S3 bucket) does "
            "not hold. False keeps the pmc source to the bucket"
        ),
    )
''' + old, 1)
cfg.write_text(s)

p = root / "ingestion_workflow/extractors/pmc_extractor.py"
s = p.read_text()
old = '''        articles, reasons = self._fetch_cloud(pmcids)
        rest = [p for p in pmcids if p not in articles]
        if rest:'''
assert old in s, "pmc anchor"
s = s.replace(old, '''        articles, reasons = self._fetch_cloud(pmcids)
        rest = [p for p in pmcids if p not in articles]
        if rest and not getattr(self.settings, "pmc_efetch", True):
            reasons.update({p: "not in PMC's Cloud Service (pmc_efetch off)" for p in rest if p not in reasons})
            rest = []
        if rest:''')
p.write_text(s)
print("patched")
