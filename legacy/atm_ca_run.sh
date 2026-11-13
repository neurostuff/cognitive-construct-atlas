#!/usr/bin/env bash
# After the material runs: constructs as authors (on constructs + family + modality): fit K=16 (5 restarts),
# 5-fold paper CV (2 restarts), the held-out-task test, and the memory / response inhibition maps.
P=/data2/james/construct-claims/work/pilot-results; R=/data/james/agents/dev/task-networks-repro
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; VC=/data2/james/construct-claims/venv-cc/bin/python
SEED=${SEED:-0}; export PYTHONHASHSEED=0  # fits: restarts use SEED .. SEED + restarts - 1; CV folds are fixed by paper
cd $P/atm
until grep -q "MAT RUNS DONE" atm_mat_run.log; do sleep 60; done
$VC $P/code/add_construct_authors.py corpus_constructs_both_mod.json corpus_constructs_authors_mod.json
export PYTHONPATH=$R/repo/src
( $PY -m task_networks fit corpus_constructs_authors_mod.json 16 fits/K16ca --s 100 --restarts 5 --seed $SEED --devices cuda:0 > fit_K16ca.log 2>&1
  for c in memory "response inhibition" "working memory" "episodic memory"; do
    $PY $P/code/construct_map.py fits/K16ca corpus_constructs_authors_mod.json "$c" maps_K16ca > /dev/null 2>&1; done ) &
$PY -m task_networks cv corpus_constructs_authors_mod.json 16 cv/K16ca.npy --s 100 --restarts 2 --seed $SEED --devices cuda:1 > cv_K16ca.log 2>&1 &
$PY $P/code/zeroshot_tasks.py corpus_constructs_authors_mod.json - zeroshot_K16_ca.json 16 cuda:2 1 $SEED > zeroshot_K16_ca.log 2>&1 &
wait
echo "CA RUNS DONE $(date -Is)"
