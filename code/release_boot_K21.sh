#!/bin/bash
# James 2026-11-11 "I want both sources of variance measured": 50 bootstrap resamples of papers at K = 21, each a
# Gibbs burn-in + EM fit with its own seed (seeds 500-549, resamples 0-49), so the band over them holds sampling and
# optimizer variability. Starts when the 100 release fits are done; then rebuilds the release maps with both bands.
cd /data2/james/construct-claims/work/pilot-results/atm || exit 1
while ! grep -q "RELEASE FITS DONE" release_gibbs_K21.log; do sleep 30; done
cp ../code/gibbs_init_next.py ../code/gibbs_init.py   # adds BOOT and the fitted-data counts; unchanged otherwise
export PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; C=corpus_constructs_both_mat.json; D=release_boot_K21
mkdir -p $D; [ -f $D/queue.txt ] || seq 0 49 > $D/queue.txt
worker() {
  local g=$1
  while true; do
    b=$(flock $D/queue.lock sh -c "head -1 $D/queue.txt; sed -i 1d $D/queue.txt")
    [ -z "$b" ] && break
    s=$((500 + b))
    CUDA_VISIBLE_DEVICES=$g $PY ../code/gibbs_init.py fit $C 21 $D/gibbs_s$s $s 1000 $b > $D/gibbs_s$s.log 2>&1
    echo "$(date -Is) gpu $g resample $b seed $s exit $?"
  done
}
for g in 0 1 2 3; do worker $g & done; wait
echo "$(date -Is) BOOT FITS DONE"
while ! grep -q "RELEASE MAPS DONE" release_post_K21.log; do sleep 30; done
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=24
$PY ../code/construct_maps_ensemble.py $C release_gibbs_K21 maps_release/K21_gibbs100 $D 2>&1 | grep -v Warn
echo "$(date -Is) RELEASE WITH BANDS DONE"
