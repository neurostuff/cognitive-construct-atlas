#!/usr/bin/env bash
# Run list E3-E5 and G1 as a GPU job queue: each of the 4 GPUs takes the next job as soon as it is free (longest
# first), with a barrier only where a decision needs the results before it.
#   prep     material corpora and the constructs-as-authors corpus (CPU), after E2 (atm_mod_run.sh) ends;
#   phase A  material (family; constructs + family) and constructs as authors: fit K=16, CV, held-out-task test;
#   phase B  the design with the best held-out-task gain (held-out contrasts break a tie; atm_ksweep.sh's rule),
#            at K = 12, 14, 18, 20: CV, held-out-task test and a full fit, so any K can be mapped;
#   phase C  maps of every construct for that design at every K (construct_maps_all.py).
# Same seeds as the run scripts: SEED (default 0), PYTHONHASHSEED=0, fits 5 restarts, CV 2, held-out tasks 1.
# usage: bash atm_queue.sh   (GPUs 0-3)
set -uo pipefail
P=/data2/james/construct-claims/work/pilot-results; R=/data/james/agents/dev/task-networks-repro
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; VC=/data2/james/construct-claims/venv-cc/bin/python
export SEED=${SEED:-0} PYTHONHASHSEED=0 NLTK_DATA=/data2/james/construct-claims/nltk_data PYTHONPATH=$R/repo/src
GPUS=(0 1 2 3)
cd $P/atm
Q=queue; mkdir -p $Q

run_phase() {  # run_phase NAME: the jobs in $Q/NAME.jobs (one shell command per line, cuda:0 = the worker's GPU)
  local name=$1 jobs=$Q/$1.jobs; echo 0 > $Q/$name.next
  echo "$(date -Is) phase $name: $(wc -l < $jobs) jobs"
  worker() {
    local g=$1
    while true; do
      local i
      i=$(flock $Q/$name.lock bash -c "n=\$(cat $Q/$name.next); echo \$((n + 1)) > $Q/$name.next; echo \$n")
      local job; job=$(sed -n "$((i + 1))p" $jobs)
      [ -z "$job" ] && break
      echo "$(date -Is) gpu $g start: $job"
      CUDA_VISIBLE_DEVICES=$g bash -c "$job"; echo "$(date -Is) gpu $g end ($?): ${job:0:90}"
    done
  }
  for g in "${GPUS[@]}"; do worker $g & done
  wait
  echo "$(date -Is) phase $name done"
}

until grep -q "MOD RUNS DONE" atm_mod_run.log 2>/dev/null; do sleep 30; done
echo "$(date -Is) E2 done; prep"
MATERIAL=1 $VC $P/code/add_modality.py $R/corpus.json corpus_family_mat.json | head -c 300; echo
MATERIAL=1 $VC $P/code/add_modality.py corpus_constructs_both.json corpus_constructs_both_mat.json | head -c 300; echo
$VC $P/code/add_construct_authors.py corpus_constructs_both_mod.json corpus_constructs_authors_mod.json

F="--s 100 --seed $SEED --devices cuda:0"
cat > $Q/A.jobs <<EOF
$PY $P/code/zeroshot_tasks.py corpus_constructs_both_mat.json corpus_family_mat.json zeroshot_K16_both_mat.json 16 cuda:0 1 $SEED > zeroshot_K16_both_mat.log 2>&1
$PY $P/code/zeroshot_tasks.py corpus_constructs_authors_mod.json - zeroshot_K16_ca.json 16 cuda:0 1 $SEED > zeroshot_K16_ca.log 2>&1
$PY -m task_networks cv corpus_constructs_authors_mod.json 16 cv/K16ca.npy --restarts 2 $F > cv_K16ca.log 2>&1
$PY -m task_networks cv corpus_constructs_both_mat.json 16 cv/K16bx.npy --restarts 2 $F > cv_K16bx.log 2>&1
$PY -m task_networks cv corpus_family_mat.json 16 cv/K16fx.npy --restarts 2 $F > cv_K16fx.log 2>&1
$PY -m task_networks fit corpus_constructs_authors_mod.json 16 fits/K16ca --restarts 5 $F > fit_K16ca.log 2>&1
$PY -m task_networks fit corpus_constructs_both_mat.json 16 fits/K16bx --restarts 5 $F > fit_K16bx.log 2>&1
$PY -m task_networks fit corpus_family_mat.json 16 fits/K16fx --restarts 5 $F > fit_K16fx.log 2>&1
EOF
run_phase A

pick=$($PY - <<'PYX'
import json, sys
import numpy as np
def zs(p):
    T = json.load(open(p))["tasks"]
    v = [x["constructs"]["gain_nats_per_peak"] for x in T.values() if x.get("constructs", {}).get("gain_nats_per_peak") is not None]
    return float(np.mean(v))
def cv(p):
    a = np.load(p, allow_pickle=True); return sum(float(r[2]) for r in a) / sum(int(r[1]) for r in a)
c = {"corpus_constructs_both_mod.json": ("K16bm", zs("zeroshot_K16_both_mod.json"), cv("cv/K16bm.npy")),
     "corpus_constructs_both_mat.json": ("K16bx", zs("zeroshot_K16_both_mat.json"), cv("cv/K16bx.npy")),
     "corpus_constructs_authors_mod.json": ("K16ca", zs("zeroshot_K16_ca.json"), cv("cv/K16ca.npy"))}
best = max(c, key=lambda k: (round(c[k][1], 3), c[k][2]))
print(json.dumps(c), file=sys.stderr); print(best, c[best][0])
PYX
)
set -- $pick; DESIGN=$1; TAG16=$2
echo "$(date -Is) design for the K sweep and maps: $DESIGN (K=16 fit fits/$TAG16)"
: > $Q/B.jobs
for K in 20 18 14 12; do echo "$PY $P/code/zeroshot_tasks.py $DESIGN - zeroshot_sweep_K$K.json $K cuda:0 1 $SEED > zeroshot_sweep_K$K.log 2>&1" >> $Q/B.jobs; done
for K in 20 18 14 12; do echo "$PY -m task_networks cv $DESIGN $K cv/sweep_K$K.npy --restarts 2 $F > cv_sweep_K$K.log 2>&1" >> $Q/B.jobs; done
for K in 20 18 14 12; do echo "$PY -m task_networks fit $DESIGN $K fits/sweep_K$K --restarts 5 $F > fit_sweep_K$K.log 2>&1" >> $Q/B.jobs; done
run_phase B

: > $Q/C.jobs
echo "$PY $P/code/construct_maps_all.py fits/$TAG16 $DESIGN maps_all/K16 > maps_all_K16.log 2>&1" >> $Q/C.jobs
for K in 12 14 18 20; do echo "$PY $P/code/construct_maps_all.py fits/sweep_K$K $DESIGN maps_all/K$K > maps_all_K$K.log 2>&1" >> $Q/C.jobs; done
run_phase C

echo "$(date -Is) held-out contrasts, K=16, paired over papers (reference: families + modality)"
$PY $P/code/cv_compare.py corpus_family_mod.json cv/K16fm.npy cv/K16am.npy cv/K16bm.npy cv/K16fx.npy cv/K16bx.npy cv/K16ca.npy
echo "$(date -Is) K sweep, held-out contrasts (reference: K=16 of the design)"
$PY $P/code/cv_compare.py $DESIGN cv/$TAG16.npy cv/sweep_K12.npy cv/sweep_K14.npy cv/sweep_K18.npy cv/sweep_K20.npy
for f in zeroshot_K16_*.json zeroshot_sweep_K*.json; do echo "$f: $(tail -n 1 ${f%.json}.log)"; done
cat maps_all_K*.log
echo "ATM ALL DONE $(date -Is)"
