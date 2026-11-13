#!/bin/bash
# After the 8:05 AM Codex reset, with the audit's prompt fixes (reports/hub_v5j_r2_audit.md):
#  1. attribution with the new prompt on the 40 audited entries (to re-audit before the full run)
#  2. task assignment for the family-labelled contrasts, build v6
#  3. re-judge the earlier "same" rulings, judge new items, judge the rest with wide candidates, build v6
#  4. attribution with the new prompt on every v6 entry
# Each Codex step resumes from its saved answers (judge_r2.json and codex_attribution_r2.json are the old answers).
cd "$(dirname "$0")"
sleep $(( $(date -d '08:10' +%s) - $(date +%s) ))
H="lexicon.json abbrev.json glossary.json paper_lexicon.json overrides.json v4/directions.json attest.json"
J="key_coverage_contrasts.json glossary.json abbrev.json judge.json gpt-6.1-sol low"
f() { echo "$(date) $1 exit $2 ($(grep -c FAILED $3 2>/dev/null) failed)"; }
ONLY=../../reports/audit_v5j_r2/attribution_keys.json WORKERS=6 python3 attribute.py v5j key_coverage_contrasts.json taxonomy_v2.json codex_attribution_v2.json gpt-6.1-sol low > attribute_sample.log 2>&1; f attribute_sample $? attribute_sample.log
WORKERS=6 python3 task_assign.py key_coverage_contrasts.json glossary.json task_assign.json gpt-6.1-sol low > task_assign.log 2>&1; f task_assign $? task_assign.log
python3 hub.py build key_coverage_contrasts.json v6 $H judge.json task_assign.json > build_v6.log 2>&1; f build_v6 $? /dev/null
REJUDGE=same WORKERS=6 python3 judge.py v6 $J > judge_r3_rejudge.log 2>&1; f rejudge $? judge_r3_rejudge.log
WORKERS=6 python3 judge.py v6 $J > judge_r3.log 2>&1; f judge $? judge_r3.log
WIDE=1 WORKERS=6 python3 judge.py v6 $J > judge_r3_wide.log 2>&1; f judge_wide $? judge_r3_wide.log
python3 hub.py build key_coverage_contrasts.json v6 $H judge.json task_assign.json > build_v6.log 2>&1; f build_v6 $? /dev/null
WORKERS=6 python3 attribute.py v6 key_coverage_contrasts.json taxonomy_v2.json codex_attribution_v2.json gpt-6.1-sol low > attribute_v6.log 2>&1; f attribute_v6 $? attribute_v6.log
