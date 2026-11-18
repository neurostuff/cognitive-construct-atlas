#!/usr/bin/env bash
# Find the best K to a step of 1 (James 2026-11-11), for the construct design (constructs + family + modality +
# material) and the families design (+ modality + material), after atm_queue.sh. Each round, k_decide.py names the K
# to evaluate next: it brackets the peak of each out-of-sample score (held-out contrasts, held-out tasks), then
# refines by midpoints. Each K gets CV and the held-out-task test, and the jobs share the 4 GPUs as a queue. When
# both designs stop, the construct design's release K (the smallest K tied with the best on both scores) gets a full
# fit if it has none, and its maps (construct_maps_all.py).
set -uo pipefail
P=/data2/james/construct-claims/work/pilot-results; PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python
export PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src
cd $P/atm
until grep -q "ATM ALL DONE" ../atm_all.log; do sleep 60; done
Z=$P/code/zeroshot_tasks.py; F="--s 100 --seed 0 --devices cuda:0"

run_jobs() {  # one shell command per line on stdin, as a queue over GPUs 0-3
  local list; list=$(mktemp -p $P/atm/queue jobs.XXXX); cat > $list; echo 0 > $list.next
  worker() {
    while true; do
      local i; i=$(flock $list.lock bash -c "n=\$(cat $list.next); echo \$((n + 1)) > $list.next; echo \$n")
      local job; job=$(sed -n "$((i + 1))p" $list); [ -z "$job" ] && break
      echo "$(date -Is) gpu $1 start: ${job:0:120}"; CUDA_VISIBLE_DEVICES=$1 bash -c "$job"; echo "$(date -Is) gpu $1 end ($?)"
    done
  }
  for g in 0 1 2 3; do worker $g & done; wait
}

declare -A done_=([constructs]=0 [families]=0)
while [ "${done_[constructs]}" = 0 ] || [ "${done_[families]}" = 0 ]; do
  jobs=""
  for d in constructs families; do
    [ "${done_[$d]}" = 1 ] && continue
    out=$($PY $P/code/k_decide.py $d); echo "$(date -Is) $out"
    set -- $out
    if [ "$1" = WAIT ]; then sleep 300; continue; fi
    if [ "$1" = STOP ]; then
      done_[$d]=1
      if [ $d = constructs ]; then echo "$2" > k_best_constructs.txt; fi
      continue
    fi
    shift
    for K in "$@"; do
      [[ "$K" =~ ^[0-9]+$ ]] || break
      if [ $d = constructs ]; then
        jobs+="$PY $Z corpus_constructs_both_mat.json - zeroshot_sweep_K$K.json $K cuda:0 1 0 > zeroshot_sweep_K$K.log 2>&1"$'\n'
        jobs+="$PY -m task_networks cv corpus_constructs_both_mat.json $K cv/sweep_K$K.npy --restarts 2 $F > cv_sweep_K$K.log 2>&1"$'\n'
      else
        jobs+="SIDES=families $PY $Z corpus_constructs_both_mat.json corpus_family_mat.json zeroshot_famsweep_K$K.json $K cuda:0 1 0 > zeroshot_famsweep_K$K.log 2>&1"$'\n'
        jobs+="$PY -m task_networks cv corpus_family_mat.json $K cv/famsweep_K$K.npy --restarts 2 $F > cv_famsweep_K$K.log 2>&1"$'\n'
      fi
    done
  done
  [ -n "$jobs" ] && printf "%s" "$jobs" | run_jobs
done
B=$(cat k_best_constructs.txt)
FIT=fits/sweep_K$B; [ "$B" = 16 ] && FIT=fits/K16bx
if [ ! -s ${FIT}_beta.npy ]; then
  CUDA_VISIBLE_DEVICES=0 $PY -m task_networks fit corpus_constructs_both_mat.json $B $FIT --restarts 5 $F > fit_sweep_K$B.log 2>&1
fi
if [ ! -s maps_all/K$B/manifest.json ]; then
  CUDA_VISIBLE_DEVICES=0 $PY $P/code/construct_maps_all.py $FIT corpus_constructs_both_mat.json maps_all/K$B > maps_all_K$B.log 2>&1
fi
echo "K EXTEND DONE $(date -Is): construct design best K $B (maps_all/K$B)"
