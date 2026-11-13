#!/usr/bin/env bash
# Rebuild the two construct hierarchies (unfiltered constructs; constructs + family), both with the hand-mapped
# category tasks, then fit K=16 (5 restarts), 5-fold paper CV (2 restarts) and the held-out-task test for each.
P=/data2/james/construct-claims/work/pilot-results; R=/data/james/agents/dev/task-networks-repro
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python
SEED=${SEED:-0}; export PYTHONHASHSEED=0  # fits: restarts use SEED .. SEED + restarts - 1; CV folds are fixed by paper
VC=/data2/james/construct-claims/venv-cc/bin/python
cd $P
for m in all:constructs both:both; do
  NLTK_DATA=/data2/james/construct-claims/nltk_data $VC code/build_construct_corpus.py $R/corpus.json atm/corpus_constructs_${m%%:*}.json ${m##*:} 1 | head -1
done
$VC code/no_constructs.py | head -1
cd atm && rm -f fits/K16a* fits/K16b* cv/K16a.npy cv/K16b.npy
export PYTHONPATH=$R/repo/src
( $PY -m task_networks fit corpus_constructs_all.json 16 fits/K16a --s 100 --restarts 5 --seed $SEED --devices cuda:0 > fit_K16a.log 2>&1
  $PY $P/code/zeroshot_tasks.py corpus_constructs_all.json $R/corpus.json zeroshot_K16_all.json 16 cuda:0 1 $SEED > zeroshot_K16_all.log 2>&1 ) &
$PY -m task_networks cv corpus_constructs_all.json 16 cv/K16a.npy --s 100 --restarts 2 --seed $SEED --devices cuda:2 > cv_K16a.log 2>&1 &
( $PY -m task_networks fit corpus_constructs_both.json 16 fits/K16b --s 100 --restarts 5 --seed $SEED --devices cuda:1 > fit_K16b.log 2>&1
  $PY -m task_networks cv corpus_constructs_both.json 16 cv/K16b.npy --s 100 --restarts 2 --seed $SEED --devices cuda:1 > cv_K16b.log 2>&1 ) &
$PY $P/code/zeroshot_tasks.py corpus_constructs_both.json $R/corpus.json zeroshot_K16_both.json 16 cuda:3 1 $SEED > zeroshot_K16_both.log 2>&1 &
wait
echo "ATM RUNS DONE $(date -Is)"
