#!/bin/bash
# After night_run3 (attribution resume): assign tasks to the family-labelled contrasts, rebuild as v6, judge the new
# single-paper contrasts, rebuild, attribute the new entries. Every Codex step resumes from its saved answers.
cd "$(dirname "$0")"
while pgrep -f night_run3.sh > /dev/null; do sleep 60; done
H="lexicon.json abbrev.json glossary.json paper_lexicon.json overrides.json v4/directions.json attest.json"
WORKERS=6 python3 task_assign.py key_coverage_contrasts.json glossary.json task_assign.json gpt-6.1-sol low > task_assign.log 2>&1
echo "$(date) task_assign exit $? ($(grep -c FAILED task_assign.log) failed batches)"
python3 hub.py build key_coverage_contrasts.json v6 $H judge.json task_assign.json > build_v6.log 2>&1; echo "$(date) build v6 exit $?"
WORKERS=6 python3 judge.py v6 key_coverage_contrasts.json glossary.json abbrev.json judge.json gpt-6.1-sol low > judge_r3.log 2>&1
echo "$(date) judge exit $? ($(grep -c FAILED judge_r3.log) failed batches)"
python3 hub.py build key_coverage_contrasts.json v6 $H judge.json task_assign.json > build_v6.log 2>&1; echo "$(date) build v6 exit $?"
WORKERS=6 python3 attribute.py v6 key_coverage_contrasts.json taxonomy_v2.json codex_attribution.json gpt-6.1-sol low > attribute_r4.log 2>&1
echo "$(date) attribute exit $? ($(grep -c FAILED attribute_r4.log) failed batches)"
