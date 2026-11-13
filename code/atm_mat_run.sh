#!/usr/bin/env bash
# After the modality runs: modality + stimulus-material authors on the family and constructs + family hierarchies
# (fit K=16, 5-fold paper CV, held-out-task test), then the construct maps for memory and response inhibition.
P=/data2/james/construct-claims/work/pilot-results; R=/data/james/agents/dev/task-networks-repro
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python
SEED=${SEED:-0}; export PYTHONHASHSEED=0  # fits: restarts use SEED .. SEED + restarts - 1; CV folds are fixed by paper
VC=/data2/james/construct-claims/venv-cc/bin/python
cd $P/atm
until grep -q "MOD RUNS DONE" atm_mod_run.log; do sleep 60; done
MATERIAL=1 $VC $P/code/add_modality.py $R/corpus.json corpus_family_mat.json
MATERIAL=1 $VC $P/code/add_modality.py corpus_constructs_both.json corpus_constructs_both_mat.json
export PYTHONPATH=$R/repo/src
( $PY -m task_networks fit corpus_family_mat.json 16 fits/K16fx --s 100 --restarts 5 --seed $SEED --devices cuda:0 > fit_K16fx.log 2>&1
  $PY -m task_networks cv corpus_family_mat.json 16 cv/K16fx.npy --s 100 --restarts 2 --seed $SEED --devices cuda:0 > cv_K16fx.log 2>&1 ) &
( $PY -m task_networks fit corpus_constructs_both_mat.json 16 fits/K16bx --s 100 --restarts 5 --seed $SEED --devices cuda:1 > fit_K16bx.log 2>&1
  for c in memory "response inhibition" "working memory" "episodic memory"; do
    $PY $P/code/construct_map.py fits/K16bx corpus_constructs_both_mat.json "$c" maps_K16bx > /dev/null 2>&1; done
  $PY -m task_networks cv corpus_constructs_both_mat.json 16 cv/K16bx.npy --s 100 --restarts 2 --seed $SEED --devices cuda:1 > cv_K16bx.log 2>&1 ) &
$PY $P/code/zeroshot_tasks.py corpus_constructs_both_mat.json corpus_family_mat.json zeroshot_K16_both_mat.json 16 cuda:2 1 $SEED > zeroshot_K16_both_mat.log 2>&1 &
wait
echo "MAT RUNS DONE $(date -Is)"
