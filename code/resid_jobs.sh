#!/bin/bash
# James 2026-11-11: does the residual-map model help prediction? K = 21; 5 CV folds and the 5 held-out-task groups of
# zeroshot_tasks.py; each job fits the network model on its training data (plain EM, seed 0) and continues it with
# residual maps at each shrinkage strength c, scoring both on the same held-out contrasts (paired). Second run:
# smoothed residuals, own counts left out of the centre, c 1e5-1e7 (first run: resid_K21_v1).
cd /data2/james/construct-claims/work/pilot-results/atm || exit 1
export PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; C=corpus_constructs_both_mat.json; D=resid_K21
CS=100000,1000000,10000000
mkdir -p $D; [ -f $D/queue.txt ] || { for i in 0 1 2 3 4; do echo "cv $i"; echo "zeroshot $i"; done > $D/queue.txt; }
worker() {
  local g=$1
  while true; do
    J=$(flock $D/queue.lock sh -c "head -1 $D/queue.txt; sed -i 1d $D/queue.txt")
    [ -z "$J" ] && break
    read -r kind i <<< "$J"
    CUDA_VISIBLE_DEVICES=$g $PY ../code/resid_fit.py job $C 21 $D/${kind}_$i.json $kind $i $CS 0 > $D/${kind}_$i.log 2>&1
    echo "$(date -Is) gpu $g $kind $i exit $? $(grep -v Warn $D/${kind}_$i.log | tail -1)"
  done
}
for g in 0 1 2 3; do worker $g & done; wait
echo "$(date -Is) RESID DONE"
