#!/usr/bin/env bash
# Modality authors on the three hierarchies (families, constructs, constructs + family): fit K=16 (5 restarts),
# 5-fold paper CV (2 restarts) and the held-out-task test, as atm_run.sh.
P=/data2/james/construct-claims/work/pilot-results; R=/data/james/agents/dev/task-networks-repro
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python
SEED=${SEED:-0}; export PYTHONHASHSEED=0  # fits: restarts use SEED .. SEED + restarts - 1; CV folds are fixed by paper
VC=/data2/james/construct-claims/venv-cc/bin/python
cd $P/atm
$VC $P/code/add_modality.py $R/corpus.json corpus_family_mod.json
$VC $P/code/add_modality.py corpus_constructs_all.json corpus_constructs_all_mod.json
$VC $P/code/add_modality.py corpus_constructs_both.json corpus_constructs_both_mod.json
export PYTHONPATH=$R/repo/src
( $PY -m task_networks fit corpus_family_mod.json 16 fits/K16fm --s 100 --restarts 5 --seed $SEED --devices cuda:0 > fit_K16fm.log 2>&1
  $PY -m task_networks cv corpus_family_mod.json 16 cv/K16fm.npy --s 100 --restarts 2 --seed $SEED --devices cuda:0 > cv_K16fm.log 2>&1 ) &
( $PY -m task_networks fit corpus_constructs_all_mod.json 16 fits/K16am --s 100 --restarts 5 --seed $SEED --devices cuda:1 > fit_K16am.log 2>&1
  $PY -m task_networks cv corpus_constructs_all_mod.json 16 cv/K16am.npy --s 100 --restarts 2 --seed $SEED --devices cuda:1 > cv_K16am.log 2>&1 ) &
( $PY -m task_networks fit corpus_constructs_both_mod.json 16 fits/K16bm --s 100 --restarts 5 --seed $SEED --devices cuda:2 > fit_K16bm.log 2>&1
  $PY -m task_networks cv corpus_constructs_both_mod.json 16 cv/K16bm.npy --s 100 --restarts 2 --seed $SEED --devices cuda:2 > cv_K16bm.log 2>&1 ) &
( $PY $P/code/zeroshot_tasks.py corpus_constructs_both_mod.json corpus_family_mod.json zeroshot_K16_both_mod.json 16 cuda:3 1 $SEED > zeroshot_K16_both_mod.log 2>&1
  $PY $P/code/zeroshot_tasks.py corpus_constructs_all_mod.json corpus_family_mod.json zeroshot_K16_all_mod.json 16 cuda:3 1 $SEED > zeroshot_K16_all_mod.log 2>&1 ) &
wait
echo "MOD RUNS DONE $(date -Is)"
