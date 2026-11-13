"""Parity with the legacy scripts on the full corpus: the package must reproduce their output byte for byte.

Needs CCA_PARITY_DIR, the working directory the scripts ran in (pilot-results/atm on beast: corpus_w_B.json,
corpus_w_B_audit1.json, audit/, cogat_*.json, ../contrast_hub/v7). Skipped elsewhere.
"""
import filecmp
import os
from pathlib import Path

import pytest

ATM = os.environ.get("CCA_PARITY_DIR")
REPO = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.parity, pytest.mark.skipif(not ATM, reason="set CCA_PARITY_DIR to the scripts' working directory")]


def test_export_release_matches_the_published_tables(tmp_path):
    """export_release.py wrote data/ from corpus_w_B.json; the package's export of the same corpus is identical"""
    from construct_atlas.cli import main
    atm = Path(ATM); hub = atm.parent / "contrast_hub" / "v7"
    main(["export-release", "--corpus-dir", str(atm), "--corpus", "corpus_w_B.json", "--hub-dir", str(hub),
          "--sources", str(hub / "constructs_dir" / "contrast_constructs_phase.json"),
          "--cogat", str(REPO / "legacy" / "contrast_hub" / "cogat_concepts.json"),
          "--inventory", str(REPO / "legacy" / "contrast_hub" / "task_inventory_v3_9e96082b.json"),
          "--connect", str(atm / "cogat_connect.json"),
          "--proposals", str(atm / "audit" / "proposals.json"), str(atm / "audit" / "proposals2.json"),
          "--legacy-root", str(REPO / "legacy"), "--out", str(tmp_path)])
    names = ["analyses.tsv", "concepts.tsv", "concept_relations.tsv", "removed_relations.tsv", "tasks.tsv", "task_concepts.tsv",
             "contrasts.tsv", "contrast_concepts.tsv", "summary.json", "cognitive_atlas/concepts.json",
             "cognitive_atlas/tasks.json", "cognitive_atlas/assertions.json"]
    different = [n for n in names if not filecmp.cmp(tmp_path / n, REPO / "data" / n, shallow=False)]
    assert different == []


def test_apply_audit_round_two_matches_the_corpus_in_use(tmp_path):
    """audit_apply.py made corpus_w_B.json from corpus_w_B_audit1.json and round two; the package makes the same file"""
    from construct_atlas.cli import main
    atm = Path(ATM)
    out = tmp_path / "out.json"
    cwd = os.getcwd()
    os.chdir(atm)   # the provenance recorded in the corpus is the paths as the script was given them
    try:
        main(["apply-audit", "--corpus-in", "corpus_w_B_audit1.json", "--corpus-out", str(out),
              "--proposals", "audit/proposals2.json", "--moves", "audit/task_moves2.json",
              "--normalize", "cogat_normalize.json", "--overrides", "cogat_overrides.json", "--concepts"])
    finally:
        os.chdir(cwd)
    assert filecmp.cmp(out, atm / "corpus_w_B.json", shallow=False)
