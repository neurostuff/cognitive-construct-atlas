#!/bin/bash
# James 2026-11-11 "Yes, both": refit everything on the corrected hierarchy (corpus_constructs_both_mat_v2.json: kind-of
# pooling, namesake tasks). One queue on 4 GPUs: converged CV with absolute log-likelihoods (cv_abs.py), the held-out-
# task test (zeroshot_tasks.py), 100 Gibbs + EM release seeds (100-199), 50 paper-bootstrap fits (seeds 500-549,
# resamples 0-49); then the seed-averaged maps with both bands and the atlas export. Run on beast from pilot-results/atm.
cd /data2/james/construct-claims/work/pilot-results/atm || exit 1
export PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; C=corpus_constructs_both_mat_v2.json; D=release_v2_K21
mkdir -p $D/seeds $D/boot
CUDA_VISIBLE_DEVICES="" $PY -c "from task_networks.corpus import load_corpus; load_corpus('$C')"   # build the cache once
if [ ! -f $D/queue.txt ]; then
  { echo "cv 0"; echo "zs 0"; for s in $(seq 100 199); do echo "gibbs $s"; done; for b in $(seq 0 49); do echo "boot $b"; done; } > $D/queue.txt
fi
worker() {
  local g=$1
  while true; do
    J=$(flock $D/queue.lock sh -c "head -1 $D/queue.txt; sed -i 1d $D/queue.txt")
    [ -z "$J" ] && break
    read -r kind i <<< "$J"
    export CUDA_VISIBLE_DEVICES=$g
    case $kind in
      cv) $PY ../code/cv_abs.py $C 21 $D/cv_abs_K21.npy --restarts 2 --iters 1000 --s 100 --seed 0 --devices cuda:0 > $D/cv.log 2>&1 ;;
      zs) $PY ../code/zeroshot_tasks.py $C - $D/zeroshot_K21.json 21 cuda:0 > $D/zeroshot.log 2>&1 ;;
      gibbs) $PY ../code/gibbs_init.py fit $C 21 $D/seeds/gibbs_s$i $i 1000 > $D/seeds/gibbs_s$i.log 2>&1 ;;
      boot) s=$((500 + i)); $PY ../code/gibbs_init.py fit $C 21 $D/boot/gibbs_s$s $s 1000 $i > $D/boot/gibbs_s$s.log 2>&1 ;;
    esac
    echo "$(date -Is) gpu $g $kind $i exit $?"
  done
}
for g in 0 1 2 3; do worker $g & done; wait
echo "$(date -Is) V2 FITS DONE"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=24
$PY ../code/construct_maps_ensemble.py $C $D/seeds maps_release/K21_v2_refit $D/boot 2>&1 | grep -v Warn
$PY ../code/export_taxonomy.py $C maps_release/K21_v2_refit ../vocab/vocab_v0.json artifact_data_v2_refit
echo "$(date -Is) V2 RELEASE DONE"
