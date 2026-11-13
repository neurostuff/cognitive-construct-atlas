#!/bin/bash
# Re-attribute the reversed entries that came back empty or were never attributed
# (rev_reattribute.json), now with every Cognitive Atlas concept among the candidates (constructs.codex_attribute);
# merge into codex_attribution_v2.json (an answer replaces an empty or missing one); constructs.py on beast; rebuild
# corpus B (contrast_construct_authors.py) and rerun its K=21 CV and held-out-task test (atm_k21_authors.sh ONLY=B).
cd "$(dirname "$0")"
# runs now, on Claude (BACKEND=claude; claude_extract.py), not after the Codex reset
export SAMPLE=key_coverage_contrasts.json
BACKEND=claude ONLY=rev_reattribute.json WORKERS=6 python3 attribute.py v7 contrasts_all.json taxonomy_v2.json codex_attribution_rev.json claude-opus-5-5 medium > attribute_rev.log 2>&1
echo "$(date) attribute_rev exit $? ($(grep -c FAILED attribute_rev.log) failed)"
cp -p codex_attribution_v2.json codex_attribution_v2_pre_rev.json
python3 - <<'PY'
import json
a = json.load(open("codex_attribution_v2.json")); r = json.load(open("codex_attribution_rev.json"))
n = sum(1 for k, v in r.items() if v and not a.get(k))
a.update({k: v for k, v in r.items() if v or k not in a})
json.dump(a, open("codex_attribution_v2.json", "w")); print("reversed entries now with constructs:", n, "of", len(r))
PY
B=/data2/james/construct-claims/work/pilot-results
scp -q codex_attribution_v2.json beast-proxy:$B/contrast_hub/v7/constructs_dir/codex_attribution.json
ssh beast-proxy "cd $B/code/contrast_hub; V=../../contrast_hub/v7; cp -rp \$V/constructs_dir \$V/constructs_dir_pre_rev; /data2/james/construct-claims/venv-cc/bin/python constructs.py \$V ../../contrast_hub/contrasts_all.json '../../contrast_hub/scan.*.jsonl+../../contrast_hub/imaging_scan.*.jsonl' \$V/claims_match.json \$V/lexicon.json \$V/constructs_dir > \$V/constructs_dir/run_rev.log 2>&1; tail -c 400 \$V/constructs_dir/run_rev.log; cd $B/atm; mv k21_authors/cv_abs_K21_B.npy k21_authors/cv_abs_K21_B_pre_rev.npy 2>/dev/null; mv k21_authors/zeroshot_K21_B.json k21_authors/zeroshot_K21_B_pre_rev.json 2>/dev/null; /data2/james/construct-claims/venv-cc/bin/python $B/code/contrast_construct_authors.py corpus_v2_task_authors.json ../contrast_hub/v7 ../contrast_hub/v7/constructs_dir/contrast_constructs.json corpus_v2_contrast_authors.json | tail -2; ONLY=B nohup ./atm_k21_authors.sh > atm_k21_authors_rev.log 2>&1 < /dev/null & disown"
echo "$(date) night_run8 done"
