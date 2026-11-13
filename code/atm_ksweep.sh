#!/usr/bin/env bash
# After the constructs-as-authors runs: pick the construct design (constructs + family + modality as parents, the same
# with material, or constructs as authors) by
# the held-out-task gain (held-out contrasts break a tie), then at K = 12, 14, 18, 20 (K = 16 exists): 5-fold paper
# CV and the held-out-task test (constructs side only), one K per GPU.
P=/data2/james/construct-claims/work/pilot-results; R=/data/james/agents/dev/task-networks-repro
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python
SEED=${SEED:-0}; export PYTHONHASHSEED=0  # fits: restarts use SEED .. SEED + restarts - 1; CV folds are fixed by paper
cd $P/atm
until grep -q "CA RUNS DONE" atm_ca_run.log 2>/dev/null; do sleep 60; done
pick=$($PY - <<'PYX'
import json, numpy as np
def zs(p):
    T = json.load(open(p))["tasks"]
    v = [x["constructs"]["gain_nats_per_peak"] for x in T.values() if x.get("constructs", {}).get("gain_nats_per_peak") is not None]
    return float(np.mean(v))
def cv(p):
    a = np.load(p, allow_pickle=True); n = sum(int(r[1]) for r in a); return sum(float(r[2]) for r in a) / n
c = {"corpus_constructs_both_mod.json": (zs("zeroshot_K16_both_mod.json"), cv("cv/K16bm.npy")),
     "corpus_constructs_both_mat.json": (zs("zeroshot_K16_both_mat.json"), cv("cv/K16bx.npy")),
     "corpus_constructs_authors_mod.json": (zs("zeroshot_K16_ca.json"), cv("cv/K16ca.npy"))}
best = max(c, key=lambda k: (round(c[k][0], 3), c[k][1]))
import sys; print(json.dumps(c), file=sys.stderr); print(best)
PYX
)
echo "$(date -Is) sweep corpus: $pick"
export PYTHONPATH=$R/repo/src
g=0
for K in 12 14 18 20; do
  ( $PY -m task_networks cv $pick $K cv/sweep_K$K.npy --s 100 --restarts 2 --seed $SEED --devices cuda:$g > cv_sweep_K$K.log 2>&1
    $PY $P/code/zeroshot_tasks.py $pick - zeroshot_sweep_K$K.json $K cuda:$g 1 $SEED > zeroshot_sweep_K$K.log 2>&1 ) &
  g=$((g + 1))
done
wait
echo "SWEEP DONE $(date -Is)"
