#!/bin/bash
# After the 8:05 AM Codex reset; the hub now draws on the whole ns-pond corpus (ns_features.py), and Codex works only
# on the sample's contrasts (SAMPLE). Prompt fixes from reports/hub_v5j_r2_audit.md. Each Codex step resumes.
#  1. attribution, new prompt, on the 40 audited entries (re-audit before the full run)
#  2. task assignment for the sample's family-labelled contrasts
#  3. build v7; re-judge earlier "same" rulings; judge new items; judge the rest with wide candidates; build v7
#  4. attribution, new prompt, on every entry the sample uses
cd "$(dirname "$0")"
sleep $(( $(date -d '08:10' +%s) - $(date +%s) ))
B=/data2/james/construct-claims/work/pilot-results/contrast_hub
scp -q beast-proxy:$B/ns_all_abbrev.json . 2>/dev/null; scp -q beast-proxy:$B/ns_all_ft_readings.json . 2>/dev/null
python3 merge_inputs.py
export SAMPLE=key_coverage_contrasts.json
H="lexicon.json abbrev_all.json glossary_all.json paper_lexicon.json overrides_all.json v4/directions.json attest.json"
J="contrasts_all.json glossary_all.json abbrev_all.json judge.json gpt-6.1-sol low"
f() { echo "$(date) $1 exit $2 ($(grep -c FAILED $3 2>/dev/null) failed)"; }
ONLY=../../reports/audit_v5j_r2/attribution_keys.json SAMPLE= WORKERS=6 python3 attribute.py v5j key_coverage_contrasts.json taxonomy_v2.json codex_attribution_v2.json gpt-6.1-sol low > attribute_sample.log 2>&1; f attribute_sample $? attribute_sample.log
WORKERS=6 python3 task_assign.py key_coverage_contrasts.json glossary.json task_assign.json gpt-6.1-sol low > task_assign.log 2>&1; f task_assign $? task_assign.log
python3 hub.py build contrasts_all.json v7 $H judge.json task_assign.json > build_v7.log 2>&1; f build_v7 $? /dev/null
REJUDGE=same WORKERS=6 python3 judge.py v7 $J > judge_r3_rejudge.log 2>&1; f rejudge $? judge_r3_rejudge.log
WORKERS=6 python3 judge.py v7 $J > judge_r3.log 2>&1; f judge $? judge_r3.log
WIDE=1 WORKERS=6 python3 judge.py v7 $J > judge_r3_wide.log 2>&1; f judge_wide $? judge_r3_wide.log
python3 hub.py build contrasts_all.json v7 $H judge.json task_assign.json > build_v7.log 2>&1; f build_v7 $? /dev/null
WORKERS=6 python3 attribute.py v7 contrasts_all.json taxonomy_v2.json codex_attribution_v2.json gpt-6.1-sol low > attribute_v7.log 2>&1; f attribute_v7 $? attribute_v7.log
