#!/usr/bin/env bash
# After the next reset: re-read with Codex the contrasts the fine-tuned model read (its readings audited at 50%
# strict against Codex's 77%), rebuild, finish the judge (resumes from judge.json), rebuild with it, then attribute
# constructs (resumes). Each Codex step resumes from its saved answers if rerun.
set -u
cd "$(dirname "$0")"
start=${START:-"03:05"}
now=$(date +%s); at=$(date -d "$start" +%s); [ "$at" -le "$now" ] && at=$(date -d "tomorrow $start" +%s); sleep $((at - now))
H="lexicon.json abbrev.json glossary.json paper_lexicon.json overrides.json v4/directions.json attest.json"
REREAD=ft WORKERS=6 BATCH=60 python3 codex_normalize.py v4 key_coverage_contrasts.json glossary.json abbrev.json overrides.json paper_lexicon.json gpt-6.1-sol low > codex_reread.log 2>&1
echo "$(date) reread exit $? ($(grep -c FAILED codex_reread.log) failed batches)"
python3 hub.py build key_coverage_contrasts.json v5j $H judge.json > build_v5j.log 2>&1; echo "$(date) build v5j exit $?"
WORKERS=6 python3 judge.py v5j key_coverage_contrasts.json glossary.json abbrev.json judge.json gpt-6.1-sol low > judge_r2.log 2>&1
echo "$(date) judge exit $? ($(grep -c FAILED judge_r2.log) failed batches)"
python3 hub.py build key_coverage_contrasts.json v5j $H judge.json > build_v5j.log 2>&1; echo "$(date) build v5j exit $?"
WORKERS=6 python3 attribute.py v5j key_coverage_contrasts.json taxonomy_v2.json codex_attribution.json gpt-6.1-sol low > attribute_r2.log 2>&1
echo "$(date) attribute exit $? ($(grep -c FAILED attribute_r2.log) failed batches)"
