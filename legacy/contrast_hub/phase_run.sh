#!/bin/bash
# Per-analysis constructs for the hub signatures that mix trial phases (James 2026-11-12: "change the associated
# constructs for those particular contrasts"; DESIGN.md, Trial phase): phase_scope.py found 100 signatures, 1,091
# analyses in 491 papers; paper_attribute.py prep (ALL=1) on beast wrote phase/papers.jsonl. This runs the Claude CLI
# reading, with one retry pass for failed papers.
cd "$(dirname "$0")"
for pass in 1 2; do
  WORKERS=6 python3 paper_attribute.py run phase/papers.jsonl cogat_concepts.json taxonomy_v2.json phase/phase_attribution.json claude-opus-5-5 medium > phase/run$pass.log 2>&1
  echo "$(date) pass $pass exit $? ($(grep -c FAILED phase/run$pass.log) failed)"
done
tail -12 phase/run2.log
