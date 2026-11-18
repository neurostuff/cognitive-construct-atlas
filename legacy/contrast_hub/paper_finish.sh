#!/bin/bash
# Finish the paper readings after the account swap: wait for the first pass, rerun the papers that failed ("Not
# logged in" during the swap; the run resumes and only redoes papers with no answer), then the rest of paper_run.sh.
cd "$(dirname "$0")"
while kill -0 1524093 2>/dev/null; do sleep 60; done
for pass in 1 2; do
  WORKERS=6 python3 paper_attribute.py run paper_attr/papers.jsonl cogat_concepts.json taxonomy_v2.json paper_attr/paper_attribution.json claude-opus-5-5 medium > paper_attr/run_retry$pass.log 2>&1
  echo "$(date) retry $pass: $(head -1 paper_attr/run_retry$pass.log); $(grep -c FAILED paper_attr/run_retry$pass.log) failed"
done
B=/data2/james/construct-claims/work/pilot-results
scp -q paper_attr/paper_attribution.json beast-proxy:$B/contrast_hub/v7/paper_attr/
ssh beast-proxy "set -e; cd $B/code/contrast_hub; V=../../contrast_hub/v7; PY=/data2/james/construct-claims/venv-cc/bin/python
\$PY paper_attribute.py merge \$V/constructs_dir/contrast_constructs.json \$V/paper_attr/paper_attribution.json \$V/constructs_dir/contrast_constructs_full.json
cd $B/atm
\$PY $B/code/contrast_construct_authors.py corpus_v2_task_authors_prenorm.json ../contrast_hub/v7 ../contrast_hub/v7/constructs_dir/contrast_constructs_full.json corpus_v2_contrast_authors_full.json | tail -2
\$PY $B/code/cogat_apply.py corpus_v2_contrast_authors_full.json cogat_normalize.json cogat_overrides.json corpus_v2_contrast_authors_full_cogat.json
while pgrep -f atm_k21_authors.sh > /dev/null; do sleep 120; done
mkdir -p k21_authors_prenorm; cp -p k21_authors/* k21_authors_prenorm/ 2>/dev/null || true
cp -p corpus_v2_contrast_authors_full_cogat.json corpus_v2_contrast_authors.json
OUT=k21_authors_norm nohup ./atm_k21_authors_v2.sh > atm_k21_authors_norm.log 2>&1 < /dev/null &
echo queued"
echo "$(date) paper_run done"
