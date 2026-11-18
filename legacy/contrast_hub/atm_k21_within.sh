#!/usr/bin/env bash
# K=21 on the within-task corpora (James 2026-11-12): the 27,905 sample analyses, one Cognitive Atlas-normalized
# hierarchy; R = constructs as parents only (the release), A = task-level construct authors, B = contrast-level.
# Release settings (cv_abs.py --restarts 2 --iters 1000 --s 100 --seed 0; zeroshot_tasks.py); each job starts on a GPU
# idle for 3 minutes; then compare_authors.py: R vs A, R vs B, A vs B. Outputs in atm/k21_within/.
# Run on beast from pilot-results/atm.
P=/data2/james/construct-claims/work/pilot-results; R=/data/james/agents/dev/task-networks-repro
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; VC=/data2/james/construct-claims/venv-cc/bin/python
export PYTHONHASHSEED=0 PYTHONPATH=$R/repo/src
cd $P/atm; D=k21_within; mkdir -p $D
free_gpu() {   # a GPU with < 1 GB used for 3 consecutive minutes
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
for x in R A B; do CUDA_VISIBLE_DEVICES="" $PY -c "from task_networks.corpus import load_corpus; load_corpus('corpus_w_$x.json')"; done
pids=()
for job in "cv R" "cv A" "cv B" "zs R" "zs A" "zs B"; do
  set -- $job; free_gpu; echo "$(date -Is) $1 $2 on gpu $G"
  if [ $1 = cv ]; then
    CUDA_VISIBLE_DEVICES=$G $PY ../code/cv_abs.py corpus_w_$2.json 21 $D/cv_abs_K21_$2.npy --restarts 2 --iters 1000 --s 100 --seed 0 --devices cuda:0 > $D/cv_$2.log 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$G $PY ../code/zeroshot_tasks.py corpus_w_$2.json - $D/zeroshot_K21_$2.json 21 cuda:0 > $D/zeroshot_$2.log 2>&1 &
  fi
  pids+=($!); sleep 90
done
wait "${pids[@]}"
for pair in "R A" "R B" "A B"; do
  set -- $pair
  $VC ../code/compare_authors.py corpus_w_$2.json $D/cv_abs_K21_$1.npy $D/cv_abs_K21_$2.npy $D/zeroshot_K21_$1.json $D/zeroshot_K21_$2.json ../contrast_hub/v7 $D/compare_$1_vs_$2.json > $D/compare_$1_vs_$2.log 2>&1
done
echo "K21 WITHIN RUNS DONE $(date -Is)"
