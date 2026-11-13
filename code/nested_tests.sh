#!/usr/bin/env bash
# James 2026-11-12 "Make and test the structure that matches my description": contrasts nested within tasks
# (nested_corpus.py) against corpus B and R, on the same documents and authors as k21_within/ (corpus_w_B_pretask.json):
# 5-fold paper CV and held-out tasks for constructs' share of a contrast's prior of 1/2, 3/4 and 9/10 at s = 100 (the
# release value), and 3/4 at s = 10 (contrasts keep more of their own pattern); then compare_authors.py against B and R.
# Run on beast from pilot-results/atm.
P=/data2/james/construct-claims/work/pilot-results; R=/data/james/agents/dev/task-networks-repro
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; VC=/data2/james/construct-claims/venv-cc/bin/python
export PYTHONHASHSEED=0 PYTHONPATH=$R/repo/src
cd $P/atm; D=k21_within
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
    sleep 60
  done
}
for sh in 1/2 3/4 9/10; do
  tg=c$(python3 -c "from fractions import Fraction as F; print(int(F('$sh') * 100))")
  [ -f corpus_w_N_$tg.json ] || $PY ../code/nested_corpus.py corpus_w_B_pretask.json $sh corpus_w_N_$tg.json
  CUDA_VISIBLE_DEVICES="" $PY -c "from task_networks.corpus import load_corpus; load_corpus('corpus_w_N_$tg.json')" > /dev/null 2>&1
done
V="c75_s100 c50_s100 c90_s100 c75_s10"
for v in $V; do
  for kind in cv zs; do
    free_gpu; tg=${v%_s*}; sv=${v#*_s}; echo "$(date -Is) $kind $v on gpu $G"   # after free_gpu, whose loops use t
    if [ $kind = cv ]; then
      CUDA_VISIBLE_DEVICES=$G $PY ../code/nested_eval.py cv corpus_w_N_$tg.json 21 $D/cv_abs_K21_N_$v.npy $sv cuda:0 > $D/cv_N_$v.log 2>&1 &
    else
      CUDA_VISIBLE_DEVICES=$G $PY ../code/nested_eval.py zs corpus_w_N_$tg.json corpus_w_B_pretask.json 21 $D/zeroshot_K21_N_$v.json $sv cuda:0 > $D/zeroshot_N_$v.log 2>&1 &
    fi
    sleep 90
  done
done
wait
for v in $V; do
  for x in B R; do
    $VC ../code/compare_authors.py corpus_w_B_pretask.json $D/cv_abs_K21_$x.npy $D/cv_abs_K21_N_$v.npy $D/zeroshot_K21_$x.json $D/zeroshot_K21_N_$v.json ../contrast_hub/v7 $D/compare_${x}_vs_N_$v.json > $D/compare_${x}_vs_N_$v.log 2>&1
  done
done
echo "NESTED TESTS DONE $(date -Is)"
