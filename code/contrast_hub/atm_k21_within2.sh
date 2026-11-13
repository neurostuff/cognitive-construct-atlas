#!/usr/bin/env bash
# The rest of atm_k21_within.sh after corpus B's new concepts were attached to the hierarchy (cogat_attach.py,
# James 2026-11-12): A's held-out-task test and B's CV and held-out-task test on the attached corpus_w_B.json; then,
# once all six results exist (R's and A's CV and R's held-out-task test were still running), the three comparisons.
P=/data2/james/construct-claims/work/pilot-results; R=/data/james/agents/dev/task-networks-repro
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; VC=/data2/james/construct-claims/venv-cc/bin/python
export PYTHONHASHSEED=0 PYTHONPATH=$R/repo/src
cd $P/atm; D=k21_within; mkdir -p $D
free_gpu() {
  while true; do
    for i in 0 1 2 3; do
      ok=1
      for t in 1 2 3; do
        m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i)
        [ "$m" -lt 1000 ] || { ok=0; break; }
        [ $t -lt 3 ] && sleep 60
      done
      [ $ok = 1 ] && { G=$i; return; }
    done
    sleep 120
  done
}
CUDA_VISIBLE_DEVICES="" $PY -c "from task_networks.corpus import load_corpus; load_corpus('corpus_w_B.json')"
rm -f $D/cv_abs_K21_B.npy $D/zeroshot_K21_B.json
for job in "zs A" "cv B" "zs B"; do
  set -- $job; free_gpu; echo "$(date -Is) $1 $2 on gpu $G"
  if [ $1 = cv ]; then
    CUDA_VISIBLE_DEVICES=$G $PY ../code/cv_abs.py corpus_w_$2.json 21 $D/cv_abs_K21_$2.npy --restarts 2 --iters 1000 --s 100 --seed 0 --devices cuda:0 > $D/cv_$2.log 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$G $PY ../code/zeroshot_tasks.py corpus_w_$2.json - $D/zeroshot_K21_$2.json 21 cuda:0 > $D/zeroshot_$2.log 2>&1 &
  fi
  sleep 90
done
for x in R A B; do
  until [ -f $D/cv_abs_K21_$x.npy ] && [ -f $D/zeroshot_K21_$x.json ]; do sleep 300; done
done
for pair in "R A" "R B" "A B"; do
  set -- $pair
  $VC ../code/compare_authors.py corpus_w_$2.json $D/cv_abs_K21_$1.npy $D/cv_abs_K21_$2.npy $D/zeroshot_K21_$1.json $D/zeroshot_K21_$2.json ../contrast_hub/v7 $D/compare_$1_vs_$2.json > $D/compare_$1_vs_$2.log 2>&1
done
echo "K21 WITHIN RUNS DONE $(date -Is)"
