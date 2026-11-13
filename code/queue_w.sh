#!/bin/bash
# One GPU queue for the within-task work (James 2026-11-13: "update the data corpus again and restart B's fit with the
# updated data"). Corpus B now carries the trial-phase constructs (phase_apply.py; corpus_w_B_prephase.json kept), so
# B's release ensemble restarts from scratch: 100 Gibbs + EM seeds (100-199) and 50 paper-bootstrap fits (seeds
# 500-549, resamples 0-49), as release_v2.sh. Then the nested tests not yet started (nested_tests.sh: construct share
# 9/10 at s = 100, 3/4 at s = 10; those runs use the pre-phase corpus_w_B_pretask.json, like R's and B's tests), and
# the comparisons for all four variants once every result exists. Each worker owns one GPU and starts once the job
# already on it (a nested run from nested_tests.sh) has ended. Run on beast from pilot-results/atm.
cd /data2/james/construct-claims/work/pilot-results/atm || exit 1
export PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; VC=/data2/james/construct-claims/venv-cc/bin/python
C=corpus_w_B.json; D=release_wB_K21; K=k21_within; Q=queue_w.txt
mkdir -p $D/seeds $D/boot
CUDA_VISIBLE_DEVICES="" $PY -c "from task_networks.corpus import load_corpus; load_corpus('$C')" > /dev/null 2>&1
if [ ! -f $Q ]; then
  { for s in $(seq 100 199); do echo "gibbs $s"; done; for b in $(seq 0 49); do echo "boot $b"; done
    for v in c90_s100 c75_s10; do echo "ncv $v"; echo "nzs $v"; done; } > $Q
fi
idle() {   # GPU $1 under 1 GB for three checks a minute apart
  for t in 1 2 3; do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)" -lt 1000 ] || return 1
    [ $t -lt 3 ] && sleep 60
  done
  return 0
}
worker() {
  local g=$1
  until idle $g; do sleep 120; done
  echo "$(date -Is) gpu $g free"
  while true; do
    J=$(flock $Q.lock sh -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$J" ] && break
    read -r kind i <<< "$J"
    export CUDA_VISIBLE_DEVICES=$g
    case $kind in
      gibbs) $PY ../code/gibbs_init.py fit $C 21 $D/seeds/gibbs_s$i $i 1000 > $D/seeds/gibbs_s$i.log 2>&1 ;;
      boot) s=$((500 + i)); $PY ../code/gibbs_init.py fit $C 21 $D/boot/gibbs_s$s $s 1000 $i > $D/boot/gibbs_s$s.log 2>&1 ;;
      ncv) $PY ../code/nested_eval.py cv corpus_w_N_${i%_s*}.json 21 $K/cv_abs_K21_N_$i.npy ${i#*_s} cuda:0 > $K/cv_N_$i.log 2>&1 ;;
      nzs) $PY ../code/nested_eval.py zs corpus_w_N_${i%_s*}.json corpus_w_B_pretask.json 21 $K/zeroshot_K21_N_$i.json ${i#*_s} cuda:0 > $K/zeroshot_N_$i.log 2>&1 ;;
    esac
    echo "$(date -Is) gpu $g $kind $i exit $?"
  done
}
for g in 0 1 2 3; do worker $g & done; wait
echo "$(date -Is) QUEUE DONE"
for v in c75_s100 c50_s100 c90_s100 c75_s10; do
  for x in B R; do
    $VC ../code/compare_authors.py corpus_w_B_pretask.json $K/cv_abs_K21_$x.npy $K/cv_abs_K21_N_$v.npy $K/zeroshot_K21_$x.json $K/zeroshot_K21_N_$v.json ../contrast_hub/v7 $K/compare_${x}_vs_N_$v.json > $K/compare_${x}_vs_N_$v.log 2>&1
  done
done
echo "$(date -Is) COMPARISONS DONE"
