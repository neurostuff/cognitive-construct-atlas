#!/usr/bin/env bash
# Add three jobs to atm_queue.sh's phase B once its job list exists:
#   - the K=16 material held-out-task test, rerun to save absolute log-likelihoods (for the baselines);
#   - constructs alone + modality + material (the clean constructs-vs-families comparison), held-out tasks and CV.
# Phase B's workers read the job list a line at a time, so lines added before they finish are run.
A=/data2/james/construct-claims/work/pilot-results/atm; cd $A
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; Z=/data2/james/construct-claims/work/pilot-results/code/zeroshot_tasks.py
until [ "$(wc -l < queue/B.jobs 2>/dev/null || echo 0)" -ge 12 ]; do sleep 10; done
cat >> queue/B.jobs <<JOBS
$PY $Z corpus_constructs_both_mat.json corpus_family_mat.json zeroshot_K16_both_mat_ll.json 16 cuda:0 1 0 > zeroshot_K16_both_mat_ll.log 2>&1
$PY $Z corpus_constructs_all_mat.json corpus_family_mat.json zeroshot_K16_all_mat.json 16 cuda:0 1 0 > zeroshot_K16_all_mat.log 2>&1
$PY -m task_networks cv corpus_constructs_all_mat.json 16 cv/K16ax.npy --restarts 2 --s 100 --seed 0 --devices cuda:0 > cv_K16ax.log 2>&1
JOBS
echo "$(date -Is) appended 3 jobs to phase B ($(wc -l < queue/B.jobs) lines)"
