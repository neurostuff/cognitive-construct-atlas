#!/bin/bash
# After the 1:06 PM Codex reset: the Codex steps the 8:25 usage limit cut short, smallest and most needed first.
#  1. stance labels for the claim sentences still unlabelled (stance.py)
#  2. attribution, direction-aware, on every directed entry the sample uses (12.7k; resumes)
#  3. the wide judge (resumes), then task assignment for the rest of the family-labelled contrasts
#  4. build v7
cd "$(dirname "$0")"
sleep $(( $(date -d '13:08' +%s) - $(date +%s) ))
export SAMPLE=key_coverage_contrasts.json
H="lexicon.json abbrev_all.json glossary_all.json paper_lexicon.json overrides_all.json v4/directions.json attest.json"
J="contrasts_all.json glossary_all.json abbrev_all.json judge.json gpt-6.1-sol low"
f() { echo "$(date) $1 exit $2 ($(grep -c FAILED $3 2>/dev/null) failed)"; }
WORKERS=4 python3 stance.py codex stance/stance_items_v2.jsonl stance/stance_codex.json gpt-6.1-sol low > stance/stance_codex_r3.log 2>&1; f stance $? stance/stance_codex_r3.log
WORKERS=6 python3 attribute.py v7 contrasts_all.json taxonomy_v2.json codex_attribution_v2.json gpt-6.1-sol low > attribute_v7.log 2>&1; f attribute_v7 $? attribute_v7.log
WIDE=1 WORKERS=6 python3 judge.py v7 $J > judge_r3_wide.log 2>&1; f judge_wide $? judge_r3_wide.log
WORKERS=6 python3 task_assign.py key_coverage_contrasts.json glossary.json task_assign.json gpt-6.1-sol low > task_assign.log 2>&1; f task_assign $? task_assign.log
python3 hub.py build contrasts_all.json v7 $H judge.json task_assign.json > build_v7.log 2>&1; f build_v7 $? /dev/null
