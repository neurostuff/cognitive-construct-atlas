#!/bin/bash
# James 2026-11-12: update the Cognitive Construct Atlas with contrast-level construct authors (corpus B; "the tasks are
# just mixtures of the contrasts") on the within-task sample
# and the updated concepts (Cognitive Atlas normalization, attachment, add-only connections, task_overrides.json).
# Same release recipe as release_v2.sh: 100 Gibbs + EM seeds (100-199), 50 paper-bootstrap fits (seeds 500-549,
# resamples 0-49), seed-averaged maps with both bands, then the export with MKDAChi2. CV and the held-out-task test
# were run on this corpus in k21_within/. Run on beast from pilot-results/atm.
cd /data2/james/construct-claims/work/pilot-results/atm || exit 1
export PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; C=corpus_w_B.json; D=release_wB_K21
mkdir -p $D/seeds $D/boot
CUDA_VISIBLE_DEVICES="" $PY -c "from task_networks.corpus import load_corpus; load_corpus('$C')"
if [ ! -f $D/queue.txt ]; then
  { for s in $(seq 100 199); do echo "gibbs $s"; done; for b in $(seq 0 49); do echo "boot $b"; done; } > $D/queue.txt
fi
worker() {
  local g=$1
  while true; do
    J=$(flock $D/queue.lock sh -c "head -1 $D/queue.txt; sed -i 1d $D/queue.txt")
    [ -z "$J" ] && break
    read -r kind i <<< "$J"
    export CUDA_VISIBLE_DEVICES=$g
    case $kind in
      gibbs) $PY ../code/gibbs_init.py fit $C 21 $D/seeds/gibbs_s$i $i 1000 > $D/seeds/gibbs_s$i.log 2>&1 ;;
      boot) s=$((500 + i)); $PY ../code/gibbs_init.py fit $C 21 $D/boot/gibbs_s$s $s 1000 $i > $D/boot/gibbs_s$s.log 2>&1 ;;
    esac
    echo "$(date -Is) gpu $g $kind $i exit $?"
  done
}
for g in 0 1 2 3; do worker $g & done; wait
echo "$(date -Is) W FITS DONE"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=24
$PY ../code/construct_maps_ensemble.py $C $D/seeds maps_release/K21_w_B $D/boot 2>&1 | grep -v Warn
$PY ../code/export_taxonomy.py $C maps_release/K21_w_B ../vocab/vocab_v0.json artifact_data_w_B
echo "$(date -Is) W RELEASE DONE"
