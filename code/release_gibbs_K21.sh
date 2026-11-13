#!/bin/bash
# James 2026-11-11: release = 100 seeds of Gibbs burn-in + EM at K = 21 (gibbs_init.py). Seeds 100-119 are the pilot
# runs (linked into release_gibbs_K21/), seeds 120-199 run here; each fit's E-step counts are cached right after it
# (construct_stability.py counts). Run on beast from pilot-results/atm.
cd /data2/james/construct-claims/work/pilot-results/atm || exit 1
export PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; C=corpus_constructs_both_mat.json; D=release_gibbs_K21
worker() {
  local g=$1
  while true; do
    s=$(flock $D/queue.lock sh -c "head -1 $D/queue.txt; sed -i 1d $D/queue.txt")
    [ -z "$s" ] && break
    CUDA_VISIBLE_DEVICES=$g $PY ../code/gibbs_init.py fit $C 21 $D/gibbs_s$s $s 1000 > $D/gibbs_s$s.log 2>&1; e=$?
    CUDA_VISIBLE_DEVICES=$g $PY ../code/construct_stability.py counts $C $D/gibbs_s$s > /dev/null 2>&1; e2=$?
    echo "$(date -Is) gpu $g seed $s fit exit $e counts exit $e2"
  done
}
for g in 0 1 2 3; do worker $g & done; wait
echo "$(date -Is) RELEASE FITS DONE"
