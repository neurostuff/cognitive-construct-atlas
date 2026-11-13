#!/usr/bin/env bash
# After the Codex usage window resets: normalize the contrasts nothing has read, rebuild, judge unique contrasts against
# their task's entries, rebuild with the judge's answers, then attribute constructs to entries (largest first). Each
# Codex step resumes from its saved answers if rerun.
set -u
cd "$(dirname "$0")"
start=${START:-"22:00"}
now=$(date +%s); at=$(date -d "today $start" +%s); [ "$at" -gt "$now" ] && sleep $((at - now))
H="lexicon.json abbrev.json glossary.json paper_lexicon.json overrides.json v4/directions.json attest.json"
echo "$(date) normalize"
WORKERS=6 BATCH=60 python3 codex_normalize.py v4 key_coverage_contrasts.json glossary.json abbrev.json overrides.json paper_lexicon.json gpt-6.1-sol low > codex_norm_r2.log 2>&1
echo "$(date) normalize exit $? ($(grep -c FAILED codex_norm_r2.log) failed batches)"
python3 hub.py build key_coverage_contrasts.json v5 $H > build_v5.log 2>&1; echo "$(date) build v5 exit $?"
WORKERS=6 python3 judge.py v5 key_coverage_contrasts.json glossary.json abbrev.json judge.json gpt-6.1-sol low > judge_r1.log 2>&1
echo "$(date) judge exit $? ($(grep -c FAILED judge_r1.log) failed batches)"
python3 hub.py build key_coverage_contrasts.json v5j $H judge.json > build_v5j.log 2>&1; echo "$(date) build v5j exit $?"
WORKERS=6 python3 attribute.py v5j key_coverage_contrasts.json taxonomy_v2.json codex_attribution.json gpt-6.1-sol low > attribute_r1.log 2>&1
echo "$(date) attribute exit $? ($(grep -c FAILED attribute_r1.log) failed batches)"
